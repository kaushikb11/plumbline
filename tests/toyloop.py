"""Toy two-model language-bus loop — shared test fixture (eng spec §15).

A deliberately small, fully deterministic-when-seeded stand-in for an OM1-style
loop that exercises all four seams (§3.1):

    sensor --> caption (StubCaptioner, model)         SENSOR_TO_CAPTION
           --> fuse   (fuse(), derived/no model call) CAPTION_TO_FUSE
           --> decide (StubDecider, model)            FUSE_TO_DECIDE
           --> act    (act(), derived/no model call)  DECIDE_TO_ACT

The two *models* (captioner, decider) carry controllable, seeded nondeterminism:
at temperature 0 they are deterministic; at temperature > 0 they sample from a
seeded `random.Random`, so two runs with different seeds diverge while a replay
that serves recorded responses must reproduce a recorded run byte-for-byte.

This module builds `SeamEvent`s directly (the job the proxy/interceptor does in
a real run) so the property tests can feed them to the substrate. It computes a
local content digest for `request_digest`; the real recorder computes this via
`plumbline.core.trace.canonicalize` (§5.2).
"""

import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize

DEFAULT_RULES: tuple[str, ...] = ("avoid obstacles", "keep humans safe")

# The two seams that are genuine model calls; "model I/O" is measured here (§15).
MODEL_SEAMS: tuple[Seam, ...] = (Seam.SENSOR_TO_CAPTION, Seam.FUSE_TO_DECIDE)

_DISTANCE_RE = re.compile(r"(\d+\.\d+)\s*m")


@dataclass(frozen=True)
class Frame:
    """One sensor reading with ground-truth scene state (sim-style, §7.3)."""

    tick: int
    obstacle_distance_m: float
    obstacle_bearing_deg: float
    scene: str


@dataclass
class StubCaptioner:
    """Model at SENSOR_TO_CAPTION. Seeded nondeterminism at temperature > 0.

    Every phrasing still contains the metric distance, so the decider downstream
    sees consistent task-relevant content across paraphrases.
    """

    rng: random.Random
    temperature: float
    model_id: str = "stub/captioner-v1"

    def caption(self, frame: Frame) -> str:
        canonical = (
            f"obstacle {frame.obstacle_distance_m:.2f} m at {frame.obstacle_bearing_deg:.0f} deg"
        )
        if self.temperature <= 0.0:
            return canonical
        variants = (
            canonical,
            f"there is an object {frame.obstacle_distance_m:.2f} m away "
            f"at {frame.obstacle_bearing_deg:.0f} deg",
            f"{frame.scene}: {frame.obstacle_distance_m:.2f} m ahead, "
            f"bearing {frame.obstacle_bearing_deg:.0f}",
        )
        if self.rng.random() < self.temperature:
            return variants[self.rng.randrange(len(variants))]
        return canonical


@dataclass
class StubDecider:
    """Model at FUSE_TO_DECIDE. Picks an action from a fixed set; seeded
    nondeterminism at temperature > 0 gives a known categorical distribution for
    the noise-floor calibration (§7.2, §15)."""

    rng: random.Random
    temperature: float
    actions: tuple[str, ...] = ("avoid", "advance", "stop")
    model_id: str = "stub/decider-v1"

    def decide(self, prompt: str) -> dict[str, JSONValue]:
        distance = _extract_distance(prompt)
        correct = "avoid" if (distance is not None and distance < 0.5) else "advance"
        if self.temperature <= 0.0 or self.rng.random() >= self.temperature:
            return {"action": correct, "args": {}}
        return {"action": self.rng.choice(self.actions), "args": {}}


def fuse(captions: Sequence[str], rules: Sequence[str]) -> str:
    """CAPTION_TO_FUSE: deterministic transform of captions + rules -> prompt."""
    observations = "; ".join(captions)
    return f"Rules: {', '.join(rules)}. Observations: {observations}. Decide the next action."


def act(plan: Mapping[str, JSONValue]) -> list[JSONValue]:
    """DECIDE_TO_ACT: deterministic transform of action plan -> HAL commands."""
    action = plan.get("action")
    if action == "avoid":
        return [{"cmd": "move", "x": -0.2, "y": 0.0, "yaw": 0.5}]
    if action == "stop":
        return [{"cmd": "stop"}]
    return [{"cmd": "move", "x": 0.3, "y": 0.0, "yaw": 0.0}]


def run_loop(
    frames: Sequence[Frame],
    captioner: StubCaptioner,
    decider: StubDecider,
    *,
    episode_id: str,
    rules: Sequence[str] = DEFAULT_RULES,
) -> tuple[SeamEvent, ...]:
    """Run the loop once, returning the ordered SeamEvents for all four seams."""
    events: list[SeamEvent] = []
    seq = 0
    for tick, frame in enumerate(frames):
        caption = captioner.caption(frame)
        events.append(
            _event(
                episode_id,
                seq,
                Seam.SENSOR_TO_CAPTION,
                tick,
                req={"frame": _frame_to_json(frame)},
                resp={"caption": caption},
                model_id=captioner.model_id,
                params={"temperature": captioner.temperature},
            )
        )
        seq += 1

        prompt = fuse([caption], rules)
        events.append(
            _event(
                episode_id,
                seq,
                Seam.CAPTION_TO_FUSE,
                tick,
                req={"captions": [caption], "rules": [*rules]},
                resp={"fused_prompt": prompt},
                model_id=None,
                params={},
            )
        )
        seq += 1

        plan = decider.decide(prompt)
        events.append(
            _event(
                episode_id,
                seq,
                Seam.FUSE_TO_DECIDE,
                tick,
                req={"prompt": prompt},
                resp={"action_plan": plan},
                model_id=decider.model_id,
                params={"temperature": decider.temperature},
            )
        )
        seq += 1

        commands = act(plan)
        events.append(
            _event(
                episode_id,
                seq,
                Seam.DECIDE_TO_ACT,
                tick,
                req={"action_plan": plan},
                resp={"hal_commands": commands},
                model_id=None,
                params={},
            )
        )
        seq += 1
    return tuple(events)


def canonical_bytes(payload: Payload) -> bytes:
    """Deterministic byte serialization of a payload's inline content, used to
    assert byte-identical model I/O (§15). Distinct from `request_digest`
    (a content hash); this keeps the readable inline bytes for comparison."""
    return json.dumps(payload.inline, sort_keys=True, separators=(",", ":")).encode("utf-8")


def model_io_bytes(events: Sequence[SeamEvent]) -> bytes:
    """Concatenated request+response bytes at the model seams, in order (§15)."""
    chunks: list[bytes] = []
    for event in events:
        if event.seam in MODEL_SEAMS:
            chunks.append(canonical_bytes(event.request))
            chunks.append(canonical_bytes(event.response))
    return b"\x00".join(chunks)


def default_matchers() -> dict[Seam, Matcher]:
    """Per-seam matchers: embedding distance for free text, exact for actions (§3.7)."""
    return {
        Seam.CAPTION_TO_FUSE: EmbeddingMatcher(threshold=0.2),
        Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=0.2),
        Seam.DECIDE_TO_ACT: ExactMatcher(),
    }


def make_frames() -> tuple[Frame, ...]:
    """A short scenario that stresses the obstacle-context case (the LiDAR-dog bug)."""
    return (
        Frame(tick=0, obstacle_distance_m=0.30, obstacle_bearing_deg=-15.0, scene="human_ahead"),
        Frame(tick=1, obstacle_distance_m=0.80, obstacle_bearing_deg=10.0, scene="open"),
        Frame(tick=2, obstacle_distance_m=0.20, obstacle_bearing_deg=5.0, scene="human_ahead"),
        Frame(tick=3, obstacle_distance_m=0.55, obstacle_bearing_deg=-30.0, scene="clutter"),
    )


def load_unimplemented(module: str, name: str) -> Any:
    """Fetch an as-yet-unbuilt public symbol from `module`.

    Raises AttributeError until the symbol is implemented — the intended failure
    for tests that pin down APIs in modules not yet scaffolded (fidelity, proxy).
    """
    return getattr(import_module(module), name)


def _frame_to_json(frame: Frame) -> dict[str, JSONValue]:
    return {
        "tick": frame.tick,
        "obstacle_distance_m": frame.obstacle_distance_m,
        "obstacle_bearing_deg": frame.obstacle_bearing_deg,
        "scene": frame.scene,
    }


def _extract_distance(prompt: str) -> float | None:
    match = _DISTANCE_RE.search(prompt)
    return float(match.group(1)) if match else None


def _event(
    episode_id: str,
    seq: int,
    seam: Seam,
    tick: int,
    *,
    req: JSONValue,
    resp: JSONValue,
    model_id: str | None,
    params: Mapping[str, JSONValue],
) -> SeamEvent:
    request = Payload(inline=req)
    response = Payload(inline=resp)
    return SeamEvent(
        episode_id=episode_id,
        seq=seq,
        seam=seam,
        logical_tick=tick,
        wall_ts=float(tick),  # synthetic; recorded wall time never drives replay (§3.4)
        request=request,
        response=response,
        model_id=model_id,
        params=params,
        request_digest=_digest(request),
        latency_ms=0.0,
    )


def _digest(payload: Payload) -> str:
    # Same request-identity convention the real recorder/proxy use (§5.2).
    return canonicalize(payload).digest
