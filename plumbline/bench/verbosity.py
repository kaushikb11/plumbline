"""Experiment A — the caption verbosity / fidelity curve (engineering spec §7.6).

Sweeps a caption from full detail down to nothing and plots downstream DECISION
fidelity (`1 - caption_loss`, §7.3) against a surface text-similarity metric. The
point (engineering spec §7 preamble, §7.6): a surface metric is BLIND TO WHICH
words carry the decision — two captions with identical surface similarity can have
opposite decision fidelity — so surface caption quality is a poor proxy for
decision preservation, and only the decision-scored metric sees the break.

HONEST SCOPE: the *magnitude* of the divergence is NOT a universal constant — it
depends on the caption's structure and the degradation knob. `truncate` drops
trailing tokens, so where fidelity breaks depends on WHERE the task-relevant words
sit (task word last -> early break, large divergence; task word first -> no break
until the caption is empty, divergence ~0). What is robust and knob-independent is
the blindness itself (see `test_surface_metric_is_blind_to_which_word_is_lost`);
this module illustrates that blindness, not a fixed cliff.

Mirrors `bench/leaderboard.py` (Experiment C): reuses `caption_loss`'s ingredients
and shares the per-scene oracle across the whole sweep. No robot, no simulator —
ground truth is the dataset label (`render(G)`, §14.5, supplied by the caller and
kept caption-agnostic).
"""

import math
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from plumbline.bench.leaderboard import Captioner, LabeledScene
from plumbline.fidelity import (
    DeciderFn,
    DecisionLabel,
    Distribution,
    Divergence,
    canonical_label,
    decision_distribution,
    decision_stability,
    total_variation,
)

# A degradation knob: (caption, level in [0,1]) -> a less-informative caption.
Degradation = Callable[[str, float], str]
# A surface text-similarity metric: (a, b) -> similarity in [0,1].
SurfaceMetric = Callable[[str, str], float]
# A reference selector for the surface metric: (scene, full_caption) -> reference text.
Reference = Callable[["LabeledScene", str], str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def truncate(caption: str, level: float) -> str:
    """Default verbosity knob: keep a `(1 - level)` prefix fraction of the caption's
    whitespace tokens, dropping trailing detail.

    Monotone and nested by construction (tokens kept at a higher level are a subset
    of those at any lower level), so information only ever decreases as `level`
    rises. Deterministic. `level<=0` -> unchanged; `level>=1` -> empty.
    """
    tokens = caption.split()
    keep = math.ceil((1.0 - _clamp01(level)) * len(tokens))
    return " ".join(tokens[:keep])


def token_dice(a: str, b: str) -> float:
    """Sørensen–Dice overlap over token multisets: `2·|A∩B| / (|A|+|B|)`.

    The 'BLEU-ish' surface metric the thesis argues against: symmetric, in [0,1],
    it counts how many words survive and is blind to whether the surviving words
    are the decision-relevant ones. Dependency-free. Two empty strings -> 1.0.
    """
    counts_a, counts_b = Counter(a.lower().split()), Counter(b.lower().split())
    denominator = sum(counts_a.values()) + sum(counts_b.values())
    if denominator == 0:
        return 1.0
    return 2.0 * sum((counts_a & counts_b).values()) / denominator


def linspace(start: float, stop: float, num: int) -> tuple[float, ...]:
    if num <= 1:
        return (start,)
    step = (stop - start) / (num - 1)
    return tuple(start + step * index for index in range(num))


DEFAULT_LEVELS: tuple[float, ...] = linspace(0.0, 1.0, 11)


@dataclass(frozen=True)
class SweepPoint:
    level: float
    decision_fidelity: float  # 1 - mean caption_loss at this level
    mean_caption_loss: float  # E[caption_loss] over scenes (§7.3)
    surface_similarity: float  # mean surface metric (degraded vs reference) over scenes
    per_scene_loss: Mapping[str, float]


@dataclass(frozen=True)
class BandwidthCurve:
    scene_ids: tuple[str, ...]
    points: tuple[SweepPoint, ...]  # ordered by increasing degradation level

    @property
    def divergence(self) -> float:
        """The largest gap where the surface metric says 'still the same caption'
        while the decisions say 'broken'. NOTE: its magnitude is structure-dependent
        (see the module docstring) — the robust finding is the blindness, not this
        number."""
        return max((p.surface_similarity - p.decision_fidelity for p in self.points), default=0.0)

    def knee(self, *, threshold: float = 0.5) -> SweepPoint | None:
        """The first level where decision fidelity drops below `threshold` — the
        bandwidth knee where dropping words starts flipping decisions (§7.6)."""
        return next((p for p in self.points if p.decision_fidelity < threshold), None)

    def as_table(self) -> str:
        return "\n".join(
            f"level={p.level:.2f}  decision_fidelity={p.decision_fidelity:.3f}  "
            f"surface_similarity={p.surface_similarity:.3f}  "
            f"(caption_loss={p.mean_caption_loss:.3f})"
            for p in self.points
        )


def run_verbosity_sweep(
    scenes: Sequence[LabeledScene],
    captioner: Captioner,
    decider: DeciderFn,
    degrade: Degradation = truncate,
    levels: Sequence[float] = DEFAULT_LEVELS,
    *,
    n: int = 32,
    surface: SurfaceMetric = token_dice,
    reference: Reference | None = None,
    label: DecisionLabel = canonical_label,
    divergence: Divergence = total_variation,
) -> BandwidthCurve:
    """Sweep `degrade` over `levels`, scoring decision fidelity (via `caption_loss`,
    inlined) against a surface-similarity baseline (§4 Experiment A).

    The oracle decision distribution and noise floor depend only on the scene, not
    the level, so they are sampled once per scene and shared across the sweep — the
    same optimization `run_captioner_leaderboard` makes, and what keeps a real-model
    run affordable.

    By default `surface` compares the degraded caption to the FULL caption ("how
    much of the original caption's text survived"); pass `reference` to compare
    against something else — e.g. `lambda scene, _full: scene.render_g` to measure
    similarity to ground truth instead, which is a stricter, less flattering baseline.
    """
    select_reference: Reference = (
        reference if reference is not None else (lambda _scene, full: full)
    )

    full_caption: dict[str, str] = {}
    oracle: dict[str, tuple[Distribution, float]] = {}
    for scene in scenes:
        full_caption[scene.scene_id] = captioner(scene)
        oracle[scene.scene_id] = (
            decision_distribution(decider, scene.render_g, n, label=label),
            decision_stability(decider, scene.render_g, n, label=label, divergence=divergence),
        )

    points: list[SweepPoint] = []
    for level in levels:
        per_scene: dict[str, float] = {}
        similarities: list[float] = []
        for scene in scenes:
            full = full_caption[scene.scene_id]
            degraded = degrade(full, level)
            d_caption = decision_distribution(decider, degraded, n, label=label)
            d_oracle, sigma = oracle[scene.scene_id]
            per_scene[scene.scene_id] = max(0.0, divergence(d_caption, d_oracle) - sigma)
            similarities.append(surface(degraded, select_reference(scene, full)))
        mean_loss = statistics.fmean(per_scene.values()) if per_scene else 0.0
        mean_similarity = statistics.fmean(similarities) if similarities else 1.0
        points.append(SweepPoint(level, 1.0 - mean_loss, mean_loss, mean_similarity, per_scene))

    return BandwidthCurve(tuple(scene.scene_id for scene in scenes), tuple(points))
