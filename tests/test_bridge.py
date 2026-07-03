"""Fidelity bridge — decision distributions and σ from RECORDED seams (§7;
limitations item "fidelity not wired to recorded seams")."""

import random
from collections.abc import Callable

from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import Payload, SeamEvent, canonicalize
from plumbline.fidelity.bridge import (
    default_decision_label,
    recorded_decision_drift,
    recorded_distribution,
    sample_recorded_decisions,
    samples_episode_id,
)

EPISODE = "ep"


def _tool_response(action: str) -> Payload:
    return Payload(
        inline={
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {
                                    "name": "Move",
                                    "arguments": f'{{"action": "{action}"}}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )


def _record_episode(store: TraceStore, ticks: int = 2) -> None:
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode(EPISODE, {})
    for tick in range(ticks):
        request = Payload(inline={"messages": [{"role": "user", "content": f"tick {tick}"}]})
        recorder.record(
            SeamEvent(
                episode_id=EPISODE,
                seq=tick,
                seam=Seam.FUSE_TO_DECIDE,
                logical_tick=tick,
                wall_ts=float(tick),
                request=request,
                response=_tool_response("move forwards"),
                model_id="cortex",
                params={},
                request_digest=canonicalize(request).digest,
                latency_ms=1.0,
            )
        )
    recorder.close_episode(EPISODE)


def _noisy_post(p_forward: float, seed: int = 7) -> Callable[[Payload], Payload]:
    rng = random.Random(seed)

    def post(request: Payload) -> Payload:
        action = "move forwards" if rng.random() < p_forward else "stand still"
        return _tool_response(action)

    return post


def test_sampler_writes_sibling_episode_and_leaves_original_untouched() -> None:
    store = TraceStore()
    _record_episode(store)
    original = (store.root / "episodes" / EPISODE / "events.jsonl").read_bytes()

    sibling = sample_recorded_decisions(store, EPISODE, _noisy_post(1.0), n=4)
    assert sibling == samples_episode_id(EPISODE)
    assert (store.root / "episodes" / EPISODE / "events.jsonl").read_bytes() == original
    samples = store.load_episode(sibling).events
    assert len(samples) == 2 * 4  # n per FUSE tick
    assert all(e.seam is Seam.FUSE_TO_DECIDE for e in samples)
    # Samples reuse the recorded request identity (same digest -> same context).
    assert {e.request_digest for e in samples} == {
        e.request_digest
        for e in store.load_episode(EPISODE).events
        if e.seam is Seam.FUSE_TO_DECIDE
    }


def test_recorded_distribution_reflects_sampled_decider() -> None:
    store = TraceStore()
    _record_episode(store)
    sample_recorded_decisions(store, EPISODE, _noisy_post(1.0), n=8)
    dist = recorded_distribution(store, EPISODE, tick=0)
    assert len(dist) == 1  # deterministic decider -> point mass
    assert max(dist.values()) == 1.0


def test_drift_flags_flip_beyond_sigma_and_not_identical_candidate() -> None:
    store = TraceStore()
    _record_episode(store)
    sample_recorded_decisions(store, EPISODE, _noisy_post(0.9), n=16)

    flipped = recorded_decision_drift(
        store, EPISODE, tick=0, candidate_responses=[_tool_response("move back")] * 8
    )
    assert flipped.divergence > flipped.sigma  # a real flip clears the floor
    assert flipped.excess > 0.5

    same = recorded_decision_drift(
        store, EPISODE, tick=0, candidate_responses=[_tool_response("move forwards")] * 8
    )
    assert same.excess < flipped.excess  # matching candidate is not charged like a flip


def test_default_decision_label_shapes() -> None:
    assert "move forwards" in default_decision_label(_tool_response("move forwards"))
    assert default_decision_label(_tool_response("a")) != default_decision_label(
        _tool_response("b")
    )
    text = Payload(inline={"choices": [{"message": {"content": "just text"}}]})
    assert default_decision_label(text) == "just text"
    bare = Payload(inline={"anything": 1})
    assert default_decision_label(bare) == '{"anything":1}'


def _scripted_post(actions: "list[str]") -> Callable[[Payload], Payload]:
    it = iter(actions)

    def post(request: Payload) -> Payload:
        return _tool_response(next(it))

    return post


def test_drift_is_seed_stable_and_size_matched() -> None:
    # Math-review F1-redux: sigma and the numerator are both size-M//2 AND averaged
    # over `trials` half-draws, so `excess` is a stable estimate, not the seed-noise
    # a single M//2 half produced (which flipped sign between seed=0 and seed=1).
    #
    # Pool: 24 "move forwards" / 8 "stand still" (75%). A REAL flip candidate (all
    # "move back") diverges ~1.0 >> sigma -> large excess, robust across seeds. A
    # same-distribution candidate sits at the floor -> excess ~ 0.
    store = TraceStore()
    _record_episode(store, ticks=1)
    sample_recorded_decisions(
        store, EPISODE, _scripted_post(["move forwards"] * 23 + ["stand still"] * 8), n=31
    )

    flip = [_tool_response("move back")] * 16
    e0 = recorded_decision_drift(store, EPISODE, tick=0, candidate_responses=flip, seed=0).excess
    e1 = recorded_decision_drift(store, EPISODE, tick=0, candidate_responses=flip, seed=1).excess
    assert e0 > 0.6 and e1 > 0.6  # a real flip is caught...
    assert abs(e0 - e1) < 0.05  # ...and the value is seed-stable (the F1-redux fix)

    # A same-distribution candidate is not charged as drift, across seeds.
    same = [_tool_response("move forwards")] * 12 + [_tool_response("stand still")] * 4
    s0 = recorded_decision_drift(store, EPISODE, tick=0, candidate_responses=same, seed=0).excess
    s1 = recorded_decision_drift(store, EPISODE, tick=0, candidate_responses=same, seed=1).excess
    assert s0 < 0.1 and s1 < 0.1
