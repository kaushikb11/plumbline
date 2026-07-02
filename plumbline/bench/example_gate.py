"""Example gate config + a small Go2 demo episode (engineering spec §8.4, §10).

Run it with:

    plumbline gate plumbline/bench/example_gate.py

`build()` records a toy golden episode, accepts it, and returns a GateSpec whose
candidate config swaps in a *compatible* captioner (a paraphrase within the
matcher threshold) — so the gate passes. Point the GitHub Action at your own
gate config (real golden traces + the candidate config you are testing).

The recorder and captioners are exported so tests and other configs can reuse
this as a stand-in golden episode.
"""

import itertools
from collections.abc import Callable, Mapping

from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.regression import Config, GateSpec, GoldenSet

EPISODE_ID = "go2-demo"
SCENES = ("scene_human", "scene_obstacle")

CAPTION: Mapping[str, str] = {
    "scene_human": "a person one meter directly ahead appears calm and clearly curious",
    "scene_obstacle": "a solid obstacle forty centimeters to the left side of the robot body",
}
COMPATIBLE_CAPTION: Mapping[str, str] = {  # paraphrase within the matcher threshold
    "scene_human": "a human one meter directly ahead appears calm and clearly curious",
    "scene_obstacle": "a solid obstacle forty centimeters to the left flank of the robot body",
}
INCOMPATIBLE_CAPTION = (
    "empty hallway clear flooring no hazards detected anywhere forward open space"
)
_ACTION: Mapping[str, JSONValue] = {
    "scene_human": {"commands": [{"type": "move", "x": 0.3, "y": 0.0, "yaw": 0.1}]},
    "scene_obstacle": {"commands": [{"type": "move", "x": 0.0, "y": 0.2, "yaw": -0.3}]},
}


def _event(seq: int, seam: Seam, tick: int, request: Payload, response: Payload) -> SeamEvent:
    return SeamEvent(
        episode_id=EPISODE_ID,
        seq=seq,
        seam=seam,
        logical_tick=tick,
        wall_ts=float(tick),
        request=request,
        response=response,
        model_id=None,
        params={},
        request_digest=canonicalize(request).digest,
        latency_ms=0.0,
    )


def record_demo_episode(store: TraceStore) -> str:
    """Record the toy Go2 golden episode into `store` and return its id."""
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode(EPISODE_ID, {"robot": "go2", "sim": "gazebo"})
    seq = itertools.count()
    for tick, scene in enumerate(SCENES):
        caption = CAPTION[scene]
        recorder.record(
            _event(
                next(seq),
                Seam.SENSOR_TO_CAPTION,
                tick,
                Payload(inline={"scene": scene}),
                Payload(inline={"caption": caption}),
            )
        )
        recorder.record(
            _event(
                next(seq),
                Seam.CAPTION_TO_FUSE,
                tick,
                Payload(inline={"captions": [caption]}),
                Payload(inline={"fused_prompt": caption}),
            )
        )
        recorder.record(
            _event(
                next(seq),
                Seam.FUSE_TO_DECIDE,
                tick,
                Payload(inline={"prompt": caption}),
                Payload(inline={"action_plan": _ACTION[scene]}),
            )
        )
        recorder.record(
            _event(
                next(seq),
                Seam.DECIDE_TO_ACT,
                tick,
                Payload(inline=_ACTION[scene]),
                Payload(inline={"executed": True}),
            )
        )
    recorder.close_episode(EPISODE_ID)
    return EPISODE_ID


def captioner(captions: Mapping[str, str] | str) -> Callable[[Payload], Payload]:
    """A swapped captioner: per-scene paraphrase (Mapping) or a constant (str)."""

    def override(request: Payload) -> Payload:
        if isinstance(captions, str):
            return Payload(inline={"caption": captions})
        inline = request.inline
        assert isinstance(inline, dict)
        scene = inline["scene"]
        assert isinstance(scene, str)
        return Payload(inline={"caption": captions[scene]})

    return override


def demo_matchers() -> dict[Seam, Matcher]:
    return {
        Seam.CAPTION_TO_FUSE: EmbeddingMatcher(threshold=0.2),
        Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=0.2),
        Seam.DECIDE_TO_ACT: ExactMatcher(),
    }


def build() -> GateSpec:
    store = TraceStore()
    episode_id = record_demo_episode(store)
    golden = GoldenSet(store)
    golden.add(episode_id)
    config = Config(
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: captioner(dict(COMPATIBLE_CAPTION))},
        matchers=demo_matchers(),
    )
    return GateSpec(store=store, golden=golden, config=config, drift_threshold=0.1)
