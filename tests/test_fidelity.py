"""Fidelity metric properties (engineering spec §7).

The noise floor's convergence is covered by test_noise_floor. These tests pin the
divergences and the caption/fusion-loss math against a *deterministic* probe
decision-maker, so the §7 formulas are checked for correctness, not just that
they run. A deterministic decider has sigma = 0, which makes the loss values
exactly predictable.
"""

from collections.abc import Callable, Iterator, Mapping

import pytest
from plumbline.core.trace import JSONValue
from plumbline.fidelity import (
    caption_loss,
    decision_distribution,
    decision_stability,
    fusion_loss,
    jensen_shannon,
    salient_artifact,
    total_variation,
)


def _probe(context: str) -> dict[str, JSONValue]:
    """A deterministic decision-maker: 'obstacle' in context -> avoid, else advance."""
    return {"action": "avoid" if "obstacle" in context else "advance", "args": {}}


def test_total_variation() -> None:
    assert total_variation({"a": 1.0}, {"a": 1.0}) == 0.0
    assert total_variation({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert total_variation({"a": 0.5, "b": 0.5}, {"a": 1.0}) == pytest.approx(0.5)


def test_jensen_shannon() -> None:
    assert jensen_shannon({"a": 1.0}, {"a": 1.0}) == 0.0
    assert jensen_shannon({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)  # bits, disjoint = max
    js = jensen_shannon({"a": 0.7, "b": 0.3}, {"a": 0.4, "b": 0.6})
    assert 0.0 < js < 1.0


def test_deterministic_decider_has_zero_noise_floor() -> None:
    assert decision_stability(_probe, "obstacle at 0.3 m", 64) == 0.0


def test_decision_binning_is_pluggable_and_canonical_default_is_lossless() -> None:
    # The §14.6 action-equivalence call: the default canonical label is lossless,
    # so distinct continuous move plans each form their own class (degenerate).
    moves: list[dict[str, JSONValue]] = [
        {"action": "move", "x": index / 10.0, "y": 0.0, "yaw": 0.0} for index in range(10)
    ]

    def mover(seq: Iterator[Mapping[str, JSONValue]]) -> Callable[[str], Mapping[str, JSONValue]]:
        return lambda _context: next(seq)

    exact = decision_distribution(mover(iter(moves)), "ctx", 10)
    assert len(exact) == 10  # all singletons under exact-canonical binning

    # A coarser caller-supplied label (here: action type) collapses them into one
    # behavioral class — the fix continuous action spaces need (§14.6).
    def by_type(plan: Mapping[str, JSONValue]) -> str:
        action = plan.get("action")
        return action if isinstance(action, str) else "?"

    typed = decision_distribution(mover(iter(moves)), "ctx", 10, label=by_type)
    assert typed == {"move": 1.0}


def test_caption_loss_is_zero_when_decision_preserved_and_high_when_flipped() -> None:
    oracle = "obstacle at 0.3 m, 15 deg left"  # render(G): decision is "avoid"

    # A caption that preserves the decision-relevant fact -> no loss.
    assert caption_loss(_probe, "an obstacle is right ahead", oracle, 32) == 0.0

    # A caption that drops it (the LiDAR-dog failure) -> maximal loss.
    assert caption_loss(_probe, "the path looks clear", oracle, 32) == pytest.approx(1.0)


def test_salient_artifact_guard_is_zero_for_faithful_re_emphasis() -> None:
    # A content-only decider: re-emphasizing a caption already present in the
    # fused prompt does not change the decision, so the salient introduces no
    # artifact and fusion_loss would not overstate the Fuser's loss.
    artifact = salient_artifact(_probe, "an obstacle is right ahead", "obstacle ahead", 32)
    assert artifact == 0.0


def test_salient_artifact_guard_flags_a_flattering_salient_decider_pair() -> None:
    # A decider sensitive to a token the default salient introduces ("observation")
    # flips its decision for no informational reason. The guard catches this: with
    # this (salient, decider) pair, fusion_loss would OVERSTATE the loss, so the
    # salient must be fixed before fusion_loss is trusted.
    def phrasing_sensitive(context: str) -> dict[str, JSONValue]:
        return {"action": "stop" if "observation" in context else "advance", "args": {}}

    artifact = salient_artifact(phrasing_sensitive, "the path is clear", "the path is clear", 32)
    assert artifact > 0.0


def test_fusion_loss_flags_dropped_task_relevant_caption() -> None:
    fused = "the path looks clear"  # fused prompt -> "advance"

    # Re-emphasizing a dropped obstacle caption flips the decision -> fusion loss.
    assert fusion_loss(_probe, fused, ["an obstacle is 0.4 m away"], 32) == pytest.approx(1.0)

    # A caption that would not change the decision contributes no loss.
    assert fusion_loss(_probe, fused, ["the weather is sunny"], 32) == 0.0


def test_decision_drift_flip_and_preserved() -> None:
    from plumbline.fidelity import decision_drift

    flip = decision_drift(_probe, "obstacle ahead", "clear ahead", 16)
    assert flip.divergence == 1.0
    assert flip.sigma == 0.0  # deterministic probe -> zero noise floor
    assert flip.excess == 1.0

    preserved = decision_drift(_probe, "obstacle ahead", "an obstacle just ahead", 16)
    assert preserved.divergence == 0.0
    assert preserved.excess == 0.0
