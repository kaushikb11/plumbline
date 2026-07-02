"""Experiment A — caption verbosity/fidelity curve (engineering spec §4, §7.6).

The thesis as a checkable result: as a caption is degraded, decision fidelity
cliffs while a surface text-similarity metric declines smoothly and stays high.
Uses a deterministic content-only probe decider (sigma = 0) so values are exact.
"""

from collections.abc import Mapping

from plumbline.bench.leaderboard import LabeledScene
from plumbline.bench.verbosity import (
    BandwidthCurve,
    linspace,
    run_verbosity_sweep,
    token_dice,
    truncate,
)
from plumbline.core.trace import JSONValue

# The task-relevant word "obstacle" is the LAST token, so truncation drops it early
# (at a low level) while most of the caption's surface form still survives.
_CAPTION = "the hallway is calm and mostly clear except one obstacle"
_SCENE = LabeledScene("hall", "data:,x", "there is an obstacle directly ahead")


def _probe(context: str) -> Mapping[str, JSONValue]:
    return {"action": "stop" if "obstacle" in context else "move_forward"}


def _captioner(_scene: LabeledScene) -> str:
    return _CAPTION


def test_truncate_is_monotone_and_nested() -> None:
    prev_tokens = _CAPTION.split()
    for level in linspace(0.0, 1.0, 11):
        tokens = truncate(_CAPTION, level).split()
        assert len(tokens) <= len(prev_tokens)
        assert tokens == prev_tokens[: len(tokens)]  # nested prefix
        prev_tokens = tokens
    assert truncate(_CAPTION, 0.0) == _CAPTION
    assert truncate(_CAPTION, 1.0) == ""


def test_token_dice_exact() -> None:
    assert token_dice("a b c", "a b c") == 1.0
    assert token_dice("a b", "c d") == 0.0
    assert token_dice("a b c d", "a b") == 2 * 2 / (4 + 2)  # 0.666...
    assert token_dice("", "") == 1.0
    assert token_dice("a", "") == 0.0
    assert token_dice("a b", "b a") == 1.0  # symmetric over multisets


def test_decision_fidelity_cliffs_while_surface_stays_high() -> None:
    curve = run_verbosity_sweep([_SCENE], _captioner, _probe, truncate, linspace(0.0, 1.0, 11), n=8)
    fidelities = [p.decision_fidelity for p in curve.points]
    # Monotone non-increasing (equal-length slices for strict zip).
    assert all(b <= a for a, b in zip(fidelities[:-1], fidelities[1:], strict=True))
    # Full caption preserves the decision; degradation eventually destroys it.
    assert curve.points[0].decision_fidelity == 1.0
    assert curve.points[-1].decision_fidelity == 0.0
    # At the first level where the decision breaks, the surface metric is still high
    # and strictly above the decision fidelity — the whole point of Experiment A.
    broken = next(p for p in curve.points if p.decision_fidelity == 0.0)
    assert broken.surface_similarity >= 0.6
    assert broken.surface_similarity > broken.decision_fidelity
    assert curve.divergence > 0.5


def test_zero_degradation_is_lossless() -> None:
    point = run_verbosity_sweep([_SCENE], _captioner, _probe, truncate, [0.0], n=8).points[0]
    assert point.mean_caption_loss == 0.0
    assert point.decision_fidelity == 1.0
    assert point.surface_similarity == 1.0


def test_full_degradation_empties_the_caption() -> None:
    point = run_verbosity_sweep([_SCENE], _captioner, _probe, truncate, [1.0], n=8).points[0]
    assert point.surface_similarity == 0.0  # empty vs the non-empty full caption
    assert 0.0 <= point.mean_caption_loss <= 1.0


def test_empty_caption_edge() -> None:
    scene = LabeledScene("empty", "data:,x", "the path is clear")
    curve = run_verbosity_sweep([scene], lambda _s: "", _probe, truncate, [0.0, 0.5, 1.0], n=4)
    assert token_dice("", "") == 1.0
    assert all(0.0 <= p.mean_caption_loss <= 1.0 for p in curve.points)


def test_oracle_sampled_once_per_scene_across_levels() -> None:
    calls: dict[str, int] = {}

    def counting(context: str) -> Mapping[str, JSONValue]:
        calls[context] = calls.get(context, 0) + 1
        return {"action": "stop" if "obstacle" in context else "move_forward"}

    n, levels = 4, [0.0, 0.5, 1.0]
    run_verbosity_sweep([_SCENE], _captioner, counting, truncate, levels, n=n)
    oracle_calls = calls.get(_SCENE.render_g, 0)
    caption_calls = sum(count for context, count in calls.items() if context != _SCENE.render_g)
    # 3n per scene TOTAL (n for D(render_g) + 2n for the floor), NOT multiplied by levels.
    assert oracle_calls == 3 * n
    # n per (scene, level) for the degraded-caption decision.
    assert caption_calls == n * len(levels)


def test_divergence_and_knee() -> None:
    curve = run_verbosity_sweep([_SCENE], _captioner, _probe, truncate, linspace(0.0, 1.0, 11), n=8)
    assert curve.divergence == max(p.surface_similarity - p.decision_fidelity for p in curve.points)
    knee = curve.knee(threshold=0.5)
    assert knee is not None and knee.decision_fidelity < 0.5
    # A caption that always preserves the decision has no knee.
    steady = run_verbosity_sweep(
        [LabeledScene("s", "data:,x", "clear path")],
        lambda _s: "clear path clear path",
        _probe,
        truncate,
        [0.0, 0.5],
        n=4,
    )
    assert steady.knee() is None


def test_as_table_renders() -> None:
    curve: BandwidthCurve = run_verbosity_sweep(
        [_SCENE], _captioner, _probe, truncate, [0.0, 1.0], n=4
    )
    table = curve.as_table()
    assert "decision_fidelity" in table and "surface_similarity" in table
    assert len(table.splitlines()) == 2
