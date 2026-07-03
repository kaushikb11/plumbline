"""The regression gate (engineering spec §8) — Experiment B in miniature.

A golden Go2 episode is recorded and accepted. Counterfactual-replaying it under a
*compatible* captioner swap reproduces the behavior and the gate passes; under an
*incompatible* swap (drops the obstacle context) the run diverges, the action
sequence can't be reproduced, and the gate fails — with the divergence attributed
to the seam. This is the "config change breaks the robot, gate goes red" property.
"""

import itertools
import json
from collections.abc import Callable, Mapping

from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.regression import Config, DecisionGate, FailurePolicy, GoldenSet, gate

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
    # The ActionSchema-derived behavior matcher (§14.6) lets a numeric jitter a real
    # controller produces run-to-run pass the gate, where ExactMatcher fails. Uses a
    # tool-call action with a numeric arg (the shape a real function-calling runtime
    # emits) parsed by GenericActionSchema.
    from plumbline.adapters import ActionSchemaMatcher, GenericActionSchema
    from plumbline.regression.golden import BehaviorLabel

    def tool_call(speed: float) -> Payload:
        arguments = json.dumps({"speed": speed})
        return Payload(
            inline={"tool_calls": [{"function": {"name": "move_forward", "arguments": arguments}}]}
        )

    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("jitter-1", {})
    recorder.record(
        _event("jitter-1", 0, Seam.DECIDE_TO_ACT, 0, tool_call(0.30), Payload(inline={"ok": True}))
    )
    recorder.close_episode("jitter-1")
    golden = GoldenSet(store)
    golden.add(
        "jitter-1", label=BehaviorLabel(actions=(tool_call(0.301),))
    )  # nudged within tolerance
    config = Config(live_frontier=set(), overrides={}, matchers={})

    assert gate(store, golden, config, 0.1).passed is False  # ExactMatcher default -> drift
    tolerant = ActionSchemaMatcher(GenericActionSchema(), atol=1e-2)
    assert gate(store, golden, config, 0.1, behavior_matcher=tolerant).passed is True


def _probe(context: str) -> Mapping[str, JSONValue]:
    return {"action": "avoid" if "obstacle" in context else "advance"}


def _obstacle_episode(store: TraceStore, swapped_caption: str) -> Config:
    golden_caption = "a solid obstacle forty centimeters to the left side of the robot"
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("obs", {})
    recorder.record(
        _event(
            "obs",
            0,
            Seam.SENSOR_TO_CAPTION,
            0,
            Payload(inline={"scene": "obstacle"}),
            Payload(inline={"caption": golden_caption}),
        )
    )
    recorder.record(
        _event(
            "obs",
            1,
            Seam.CAPTION_TO_FUSE,
            0,
            Payload(inline={"captions": [golden_caption]}),
            Payload(inline={"fused_prompt": golden_caption}),
        )
    )
    recorder.record(
        _event(
            "obs",
            2,
            Seam.FUSE_TO_DECIDE,
            0,
            Payload(inline={"prompt": golden_caption}),
            Payload(inline={"action_plan": _ACTION["scene_obstacle"]}),
        )
    )
    recorder.record(
        _event(
            "obs",
            3,
            Seam.DECIDE_TO_ACT,
            0,
            Payload(inline=_ACTION["scene_obstacle"]),
            Payload(inline={"executed": True}),
        )
    )
    recorder.close_episode("obs")

    def override(request: Payload) -> Payload:
        return Payload(inline={"caption": swapped_caption})

    return Config(
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: override},
        matchers=_matchers(),
    )


def test_decision_gate_catches_low_surface_flip_missed_by_surface_gate() -> None:
    # THE flagship contrast: a caption that drops the one decision-critical token but
    # stays surface-close. Same store, same config, opposite verdicts.
    store = TraceStore()
    config = _obstacle_episode(
        store, "a solid clear forty centimeters to the left side of the robot"
    )
    golden = GoldenSet(store)
    golden.add("obs")

    surface = gate(store, golden, config, 0.1)  # surface/structural mode
    assert surface.passed is True  # MISS: caption within threshold -> old decision served
    assert surface.max_drift == 0.0

    caught = gate(store, golden, config, 0.1, decision=DecisionGate(_probe, k=3.0))
    assert caught.passed is False  # CATCH: the decision flips avoid -> advance
    assert caught.per_episode[0].decision_divergence == 1.0
    assert caught.per_episode[0].sigma == 0.0
    assert caught.per_episode[0].divergence_seam is Seam.SENSOR_TO_CAPTION
    assert caught.threshold_units == "sigma"


def test_decision_gate_does_not_flag_a_benign_rephrasing() -> None:
    store = TraceStore()
    # keeps "obstacle" (the decision-critical token) -> decision unchanged
    config = _obstacle_episode(store, "an obstacle sitting forty centimeters to the robot left")
    golden = GoldenSet(store)
    golden.add("obs")
    result = gate(store, golden, config, 0.1, decision=DecisionGate(_probe, k=3.0))
    assert result.passed is True
    assert result.per_episode[0].decision_divergence == 0.0
