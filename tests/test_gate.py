"""The regression gate (engineering spec §8) — Experiment B in miniature.

A golden Go2 episode is recorded and accepted. Counterfactual-replaying it under a
*compatible* captioner swap reproduces the behavior and the gate passes; under an
*incompatible* swap (drops the obstacle context) the run diverges, the action
sequence can't be reproduced, and the gate fails — with the divergence attributed
to the seam. This is the "config change breaks the robot, gate goes red" property.
"""

import itertools
from collections.abc import Callable, Mapping

from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.regression import Config, FailurePolicy, GoldenSet, gate

_SCENES = ("scene_human", "scene_obstacle")
_CAPTION: Mapping[str, str] = {
    "scene_human": "a person one meter directly ahead appears calm and clearly curious",
    "scene_obstacle": "a solid obstacle forty centimeters to the left side of the robot body",
}
_COMPATIBLE: Mapping[str, str] = {  # paraphrase within the matcher threshold
    "scene_human": "a human one meter directly ahead appears calm and clearly curious",
    "scene_obstacle": "a solid obstacle forty centimeters to the left flank of the robot body",
}
_INCOMPATIBLE = "empty hallway clear flooring no hazards detected anywhere forward open space"
_ACTION: Mapping[str, JSONValue] = {
    "scene_human": {"commands": [{"type": "move", "x": 0.3, "y": 0.0, "yaw": 0.1}]},
    "scene_obstacle": {"commands": [{"type": "move", "x": 0.0, "y": 0.2, "yaw": -0.3}]},
}


def _event(
    episode_id: str, seq: int, seam: Seam, tick: int, request: Payload, response: Payload
) -> SeamEvent:
    return SeamEvent(
        episode_id=episode_id,
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


def _record_golden_episode() -> TraceStore:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("golden-1", {"robot": "go2", "sim": "gazebo"})
    seq = itertools.count()
    for tick, scene in enumerate(_SCENES):
        caption = _CAPTION[scene]
        recorder.record(
            _event(
                "golden-1",
                next(seq),
                Seam.SENSOR_TO_CAPTION,
                tick,
                Payload(inline={"scene": scene}),
                Payload(inline={"caption": caption}),
            )
        )
        recorder.record(
            _event(
                "golden-1",
                next(seq),
                Seam.CAPTION_TO_FUSE,
                tick,
                Payload(inline={"captions": [caption]}),
                Payload(inline={"fused_prompt": caption}),
            )
        )
        recorder.record(
            _event(
                "golden-1",
                next(seq),
                Seam.FUSE_TO_DECIDE,
                tick,
                Payload(inline={"prompt": caption}),
                Payload(inline={"action_plan": _ACTION[scene]}),
            )
        )
        recorder.record(
            _event(
                "golden-1",
                next(seq),
                Seam.DECIDE_TO_ACT,
                tick,
                Payload(inline=_ACTION[scene]),
                Payload(inline={"executed": True}),
            )
        )
    recorder.close_episode("golden-1")
    return store


def _captioner(captions: Mapping[str, str] | str) -> Callable[[Payload], Payload]:
    def override(request: Payload) -> Payload:
        if isinstance(captions, str):
            return Payload(inline={"caption": captions})
        inline = request.inline
        assert isinstance(inline, dict)
        scene = inline["scene"]
        assert isinstance(scene, str)
        return Payload(inline={"caption": captions[scene]})

    return override


def _matchers() -> dict[Seam, Matcher]:
    return {
        Seam.CAPTION_TO_FUSE: EmbeddingMatcher(threshold=0.2),
        Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=0.2),
        Seam.DECIDE_TO_ACT: ExactMatcher(),
    }


def _config(captions: Mapping[str, str] | str) -> Config:
    return Config(
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner(captions)},
        matchers=_matchers(),
    )


def test_golden_set_captures_behavior_and_versions() -> None:
    store = _record_golden_episode()
    golden = GoldenSet(store)
    golden.add("golden-1")
    assert len(golden.episodes()) == 1
    assert len(golden.episodes()[0].label.actions) == len(_SCENES)  # one action plan per tick
    assert len(golden.version()) == 64  # sha256 hex content hash


def test_gate_passes_on_a_compatible_config() -> None:
    store = _record_golden_episode()
    golden = GoldenSet(store)
    golden.add("golden-1")

    result = gate(store, golden, _config(dict(_COMPATIBLE)), drift_threshold=0.1)

    assert result.passed is True
    assert result.max_drift == 0.0
    assert result.diverged_fraction == 0.0


def test_gate_fails_on_an_injected_regression() -> None:
    store = _record_golden_episode()
    golden = GoldenSet(store)
    golden.add("golden-1")

    result = gate(store, golden, _config(_INCOMPATIBLE), drift_threshold=0.1)

    assert result.passed is False
    assert result.max_drift > 0.1
    assert result.diverged_fraction == 1.0
    assert result.per_episode[0].diverged is True
    assert result.per_episode[0].divergence_seam is Seam.CAPTION_TO_FUSE
    assert result.per_episode[0].divergence_distance is not None


def test_failure_policies() -> None:
    store = _record_golden_episode()
    golden = GoldenSet(store)
    golden.add("golden-1")
    config = _config(_INCOMPATIBLE)

    # A regression with drift 1.0 fails under every policy at threshold 0.1.
    for policy in FailurePolicy:
        assert gate(store, golden, config, 0.1, policy=policy).passed is False
    # ...and a threshold above the drift passes.
    assert gate(store, golden, config, 1.0, policy=FailurePolicy.ANY).passed is True


def test_gate_action_schema_matcher_tolerates_numeric_jitter() -> None:
    # The ActionSchema-derived behavior matcher (§14.6) lets a coordinate jitter a
    # real controller produces run-to-run pass the gate, where ExactMatcher fails.
    from plumbline.adapters import ActionSchemaMatcher, OM1ActionSchema
    from plumbline.regression.golden import BehaviorLabel

    store = _record_golden_episode()
    golden = GoldenSet(store)
    # Golden label = the recorded plans nudged within tolerance.
    golden.add(
        "golden-1",
        label=BehaviorLabel(
            actions=(
                Payload(inline={"commands": [{"type": "move", "x": 0.301, "y": 0.0, "yaw": 0.1}]}),
                Payload(inline={"commands": [{"type": "move", "x": 0.0, "y": 0.201, "yaw": -0.3}]}),
            )
        ),
    )
    config = Config(live_frontier=set(), overrides={}, matchers={})

    assert gate(store, golden, config, 0.1).passed is False  # ExactMatcher default -> drift
    tolerant = ActionSchemaMatcher(OM1ActionSchema(), atol=1e-2)
    assert gate(store, golden, config, 0.1, behavior_matcher=tolerant).passed is True
