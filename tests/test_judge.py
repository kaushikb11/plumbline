"""Behavioral-equivalence judge (engineering spec §7.5).

Structural comparison of typed action plans, and the LLM-as-judge routed through
the proxy so the judgment is recorded and replayed exactly like any other model
call — the eval is as reproducible as the thing it evaluates.
"""

import random

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload
from plumbline.fidelity import (
    judge_noise_floor,
    semantic_equivalence,
    structural_equivalence,
)
from plumbline.proxy import RecordingProxy, ReplayingProxy

_CTX = Context(episode_id="judge-ep", model_id="judge/model-v1", params={"temperature": 0.7})


def _plan(action: str, **args: float) -> Payload:
    typed_args: dict[str, JSONValue] = dict(args)
    return Payload(inline={"action": action, "args": typed_args})


def _stub_judge(request: Payload) -> Payload:
    """A deterministic LLM judge stand-in: equivalent iff the two sequences match.

    Deterministic in the prompt, so record -> faithful replay reproduces it.
    """
    inline = request.inline
    assert isinstance(inline, dict)
    verdict = (
        "EQUIVALENT" if inline.get("sequence_a") == inline.get("sequence_b") else "NOT EQUIVALENT"
    )
    return Payload(inline={"choices": [{"message": {"content": verdict}}]})


def test_structural_equivalence_penalizes_mismatch_and_length() -> None:
    base = [_plan("move", x=0.3, yaw=0.1), _plan("stop")]

    identical = structural_equivalence(base, [_plan("move", x=0.3, yaw=0.1), _plan("stop")])
    assert identical.equivalent is True
    assert identical.distance == 0.0

    one_step_differs = structural_equivalence(base, [_plan("move", x=0.3, yaw=0.1), _plan("turn")])
    assert one_step_differs.equivalent is False
    assert one_step_differs.distance == 0.5  # 1 of 2 aligned steps

    shorter = structural_equivalence(base, [_plan("move", x=0.3, yaw=0.1)])
    assert shorter.equivalent is False
    assert shorter.distance == 0.5  # length gap of 1 over 2 steps, penalized

    longer = structural_equivalence(base, [*base, _plan("speak")])
    assert longer.equivalent is False
    assert longer.distance > 0.0  # the extra action is not free


def test_semantic_judge_is_recorded_and_replayable_through_the_proxy() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    proxy = RecordingProxy(_stub_judge, recorder)

    def record_judge(request: Payload) -> Payload:
        return proxy.forward(request, _CTX)

    same_a, same_b = [_plan("move", x=0.3)], [_plan("move", x=0.3)]
    diff_a, diff_b = [_plan("move", x=0.3)], [_plan("stop")]

    recorded_same = semantic_equivalence(same_a, same_b, record_judge)
    recorded_diff = semantic_equivalence(diff_a, diff_b, record_judge)
    proxy.close(_CTX.episode_id)
    assert recorded_same.equivalent is True
    assert recorded_diff.equivalent is False
    assert recorded_same.method == "semantic"

    # Replay: serve the recorded judge responses; verdicts must reproduce exactly.
    replay = ReplayingProxy(store, _CTX.episode_id)

    def replay_judge(request: Payload) -> Payload:
        return replay.faithful(request, _CTX)

    assert semantic_equivalence(same_a, same_b, replay_judge).equivalent is True
    assert semantic_equivalence(diff_a, diff_b, replay_judge).equivalent is False


def test_judge_noise_floor_detects_self_disagreement() -> None:
    seq_a, seq_b = [_plan("move", x=0.3)], [_plan("move", x=0.3)]

    flips = random.Random(0)

    def noisy_judge(_request: Payload) -> Payload:
        verdict = "EQUIVALENT" if flips.random() < 0.5 else "NOT EQUIVALENT"
        return Payload(inline={"choices": [{"message": {"content": verdict}}]})

    assert judge_noise_floor(seq_a, seq_b, noisy_judge, 200) > 0.0

    def steady_judge(_request: Payload) -> Payload:
        return Payload(inline={"choices": [{"message": {"content": "EQUIVALENT"}}]})

    assert judge_noise_floor(seq_a, seq_b, steady_judge, 200) == 0.0


def test_parse_equivalent_bare_late_negation_is_conservative() -> None:
    # A judge that hedges then reverses ("...equivalent, but no.") must NOT pass as
    # EQUIVALENT — a bare trailing negation counts as a difference signal. (Framework
    # review, fidelity finding: the one parse that landed un-conservatively.)
    from plumbline.core.trace import Payload
    from plumbline.fidelity.judge import _parse_equivalent

    def _p(text: str) -> Payload:
        return Payload(inline=text)

    assert _parse_equivalent(_p("They look equivalent at first, but no.")) is False
    assert _parse_equivalent(_p("same behavior overall; not.")) is False
    # And the good cases still parse correctly:
    assert _parse_equivalent(_p("EQUIVALENT")) is True
    assert _parse_equivalent(_p("NOT EQUIVALENT")) is False
    assert _parse_equivalent(_p("they do not diverge; identical")) is True
    assert _parse_equivalent(_p("not fully equivalent")) is False
