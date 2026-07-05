"""The regression gate — CI for robot behavior (engineering spec §8.2–§8.5).

For each golden episode, counterfactual-replay under a candidate config (the
swapped model / edited prompt / changed rule, expressed as seam overrides),
compute behavioral drift from the accepted behavior, and fail per policy.

Drift is the §7.5 structural behavioral-equivalence distance between the replayed
action sequence and the golden action sequence — alignment then per-step distance
with a length penalty (§8.3). A candidate config that diverges an episode halts
before the action seam, leaving the replayed sequence short; the alignment
penalizes that, so a config change that breaks reproduction surfaces as high
drift *and* is attributed to the seam that diverged (§6.4).

Honest positioning (§8.5): eval-gated CI, golden cases, and fail-on-drift are
established LLM-regression-testing practice. What is new is the target (embodied
robot decisions) and the determinism the replay substrate provides — you gate on
*reproduced* behavior, not re-rolled samples.
"""

import enum
import logging
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import ExactMatcher, Matcher
from plumbline.core.replayer import DivergencePolicy, Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload
from plumbline.fidelity import (
    DeciderFn,
    DecisionDrift,
    DecisionLabel,
    Divergence,
    canonical_label,
    decision_drift,
    structural_equivalence,
)
from plumbline.fidelity.decision import total_variation
from plumbline.regression.golden import GoldenSet, action_sequence

_log = logging.getLogger("plumbline.regression")
_EXACT_MATCHER: Matcher = ExactMatcher()


def _text_leaves(value: JSONValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _text_leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _text_leaves(item)]
    return []


def _default_context(response: Payload) -> str:
    """Concatenate the string leaves of a seam response — a runtime-neutral default
    caption context. Injectable (like render(G)/salient), because the caption field is
    runtime-specific (§14.5/§14.6)."""
    return " ".join(_text_leaves(response.inline)).strip()


@dataclass(frozen=True)
class Config:
    """A candidate config change, expressed as the counterfactual to run (§8.2).

    The swapped model / edited prompt / changed rule is captured as seam
    `overrides`; an adapter translates a real config diff into these.
    """

    live_frontier: set[Seam]
    overrides: Mapping[Seam, Callable[[Payload], Payload]]
    matchers: Mapping[Seam, Matcher]
    on_divergence: DivergencePolicy = DivergencePolicy.HALT


class FailurePolicy(enum.Enum):
    ANY = "any"  # fail if any episode's drift exceeds the threshold
    AGGREGATE = "aggregate"  # fail if the mean drift exceeds the threshold
    QUANTILE = "quantile"  # fail if the q-quantile drift exceeds the threshold


@dataclass(frozen=True)
class EpisodeDrift:
    episode_id: str
    drift: float  # behavioral distance from golden (§8.3); in σ-units under a decision gate
    diverged: bool
    divergence_seam: Seam | None  # per-seam attribution (§6.4)
    divergence_distance: float | None
    decision_divergence: float | None = None  # div(D(candidate), D(golden)), decision mode
    sigma: float | None = None  # the decider's noise floor, decision mode
    scored: bool = True  # False if no frontier-seam event was scored (nothing to gate, §Q20)


@dataclass(frozen=True)
class DecisionGate:
    """Opt-in decision-divergence gate mode (§7, §14.6). Scores drift as the decider's
    decision-distribution divergence between the golden and counterfactual caption,
    corrected by the noise floor σ, and fails iff it exceeds `k`·σ. This CATCHES a
    low-surface decision flip the surface matcher misses, and does NOT flag a benign
    rephrasing that doesn't move the decision.

    Scope: it runs the supplied `decider` on the counterfactual caption — it does NOT
    re-run a stateful fuser or the recorded Cortex decider (that needs a runtime
    re-drive). `decider` must be live / temperature-sampling (not a by-digest replay).
    """

    decider: DeciderFn
    n: int = 32
    # Two thresholding modes (§14.6 HUMAN REVIEW; docs/math-review-section7.md):
    #  - alpha (DEFAULT, distribution-free): fail iff the permutation p-value of the
    #    observed divergence against the golden's own split-half null is < alpha.
    #    Calibrated and assumption-free — the right tool for a metric that clusters
    #    near 0/1, where a sigma-multiple's normality assumption fails (arXiv 2412.12148).
    #    (With the default trials=32 the smallest achievable p is 1/33 ≈ 0.03, so a
    #    real flip fires at alpha=0.05; raise `trials` for a stricter alpha.)
    #  - k (legacy, opt in with alpha=None): fail iff excess/max(sigma, 1/n) > k. A
    #    sigma-multiple placeholder; k=3 is uncalibrated. The sigma floor at 1/n (the
    #    estimator resolution) removes the old sigma=0 -> infinity hair-trigger.
    # alpha takes precedence when set (the default); pass alpha=None to use k.
    alpha: float | None = 0.05
    k: float = 3.0
    label: DecisionLabel = canonical_label
    divergence: Divergence = total_variation
    context_seam: Seam | None = None  # default: the single live_frontier seam
    context_of: Callable[[Payload], str] = _default_context


@dataclass(frozen=True)
class GateResult:
    passed: bool
    threshold: float
    policy: FailurePolicy
    per_episode: tuple[EpisodeDrift, ...]
    threshold_units: str = "distance"  # "sigma" when a decision gate is active

    @property
    def diverged_fraction(self) -> float:
        if not self.per_episode:
            return 0.0
        return sum(episode.diverged for episode in self.per_episode) / len(self.per_episode)

    @property
    def max_drift(self) -> float:
        return max((episode.drift for episode in self.per_episode), default=0.0)


@dataclass
class GateSpec:
    """Everything the gate needs, bundled — the contract a CLI gate-config module's
    `build()` returns (§8.4). The candidate config is Python because a seam swap is
    inherently code (it re-runs a seam), so a config module is how CI declares it.
    """

    store: TraceStore
    golden: GoldenSet
    config: Config
    drift_threshold: float
    policy: FailurePolicy = FailurePolicy.ANY
    quantile: float = 0.95  # used only when policy is QUANTILE
    # The §14.6 per-step action-equivalence matcher (sets the false-positive rate).
    # Threaded through to gate() so a CI config can override the strict default.
    behavior_matcher: Matcher = _EXACT_MATCHER
    # Opt-in decision-divergence gate (catches low-surface decision flips the matcher
    # misses); None -> the surface/structural path.
    decision: "DecisionGate | None" = None


def gate(
    store: TraceStore,
    golden: GoldenSet,
    config: Config,
    drift_threshold: float,
    *,
    behavior_matcher: Matcher = _EXACT_MATCHER,
    policy: FailurePolicy = FailurePolicy.ANY,
    quantile: float = 0.95,
    decision: "DecisionGate | None" = None,
) -> GateResult:
    """Run the gate over the golden set under `config` (§8.2).

    NOTE (§14.6, HUMAN REVIEW): `behavior_matcher` is the per-step action
    equivalence and it sets the gate's false-positive rate. ExactMatcher is the
    strict default; `recommended_behavior_matcher` (typed, tolerant, reorder-
    insensitive) is the open choice.

    With `decision` supplied, the gate uses the decision-divergence mode (§7): drift
    is scored in σ-units and the run fails iff it exceeds `decision.k`·σ — catching a
    low-surface decision flip the surface matcher misses. Without it, the surface/
    structural path runs (the honest fallback: pure-trace replay can't re-run a
    decider).
    """
    if decision is not None:
        # Foot-gun guard (math-review Q13): in decision mode the threshold is
        # `decision.k` σ-units, NOT `drift_threshold` (distance) — the latter is
        # silently unused. Warn if a caller passed a meaningful one expecting it to
        # apply, rather than letting a 0.1 quietly become a 3.0-σ gate.
        if drift_threshold != 0.0:
            _log.warning(
                "gate: drift_threshold=%s is IGNORED in decision mode; the threshold is "
                "decision.k=%s (sigma-units). Pass drift_threshold=0.0 to silence this.",
                drift_threshold,
                decision.k,
            )
        return _decision_gate(store, golden, config, decision, policy, quantile)
    drifts: list[EpisodeDrift] = []
    for episode in golden.episodes():
        replayer = Replayer(store, VirtualClock(), config.matchers)
        result = replayer.counterfactual(
            episode.episode_id,
            config.live_frontier,
            config.overrides,
            config.on_divergence,
        )
        candidate = action_sequence(result.events)
        verdict = structural_equivalence(episode.label.actions, candidate, matcher=behavior_matcher)
        drifts.append(
            EpisodeDrift(
                episode_id=episode.episode_id,
                drift=verdict.distance,
                diverged=result.diverged,
                divergence_seam=result.divergence_seam,
                divergence_distance=result.divergence_distance,
            )
        )
    # An empty golden set cannot certify "no regression" — fail rather than pass
    # vacuously (a mis-loaded corpus must not silently green a CI gate).
    passed = bool(drifts) and _passes([d.drift for d in drifts], drift_threshold, policy, quantile)
    return GateResult(
        passed=passed, threshold=drift_threshold, policy=policy, per_episode=tuple(drifts)
    )


def _passes(
    drifts: Sequence[float], threshold: float, policy: FailurePolicy, quantile: float
) -> bool:
    if not drifts:
        return True
    if policy is FailurePolicy.ANY:
        return max(drifts) <= threshold
    if policy is FailurePolicy.AGGREGATE:
        return statistics.fmean(drifts) <= threshold
    ordered = sorted(drifts)
    # Nearest-rank q-quantile: the smallest value at or above the q fraction. The
    # old int(q*n) was one too high for n>=~20, collapsing P95 into ANY.
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index] <= threshold


def _single_frontier(config: Config) -> Seam:
    if len(config.live_frontier) != 1:
        raise ValueError("the decision gate requires a single-seam live_frontier")
    return next(iter(config.live_frontier))


def _decision_gate(
    store: TraceStore,
    golden: GoldenSet,
    config: Config,
    decision: DecisionGate,
    policy: FailurePolicy,
    quantile: float,
) -> GateResult:
    """Score drift as decision divergence, bypassing the surface matcher/halt: run the
    decider on the recorded (golden) vs override (counterfactual) caption at the
    frontier seam. In p-value mode (`alpha` set) fail iff the permutation p-value <
    alpha; in legacy σ-unit mode fail iff excess/max(σ, 1/n) > k.

    Both statistics are normalized to a HIGHER-IS-WORSE `stat` so the ANY/AGGREGATE/
    QUANTILE policy machinery is shared: p-value → `1 − p` (threshold `1 − alpha`),
    σ-mode → `excess/max(σ, 1/n)` (threshold `k`, σ floored at the estimator
    resolution 1/n to kill the σ=0 → ∞ hair-trigger)."""
    seam = decision.context_seam or _single_frontier(config)
    override = config.overrides.get(seam)
    if override is None:
        raise ValueError(f"the decision gate needs an override for the frontier seam {seam.value}")
    use_pvalue = decision.alpha is not None
    threshold = (1.0 - decision.alpha) if decision.alpha is not None else decision.k
    sigma_floor = 1.0 / decision.n
    drifts: list[EpisodeDrift] = []
    for episode in golden.episodes():
        worst: DecisionDrift | None = None
        worst_stat = 0.0
        stats: list[float] = []
        for event in store.load_episode(episode.episode_id).events:
            if event.seam is not seam:
                continue
            golden_ctx = decision.context_of(event.response)
            candidate_ctx = decision.context_of(override(event.request))
            score = decision_drift(
                decision.decider,
                golden_ctx,
                candidate_ctx,
                decision.n,
                label=decision.label,
                divergence=decision.divergence,
            )
            if use_pvalue:
                stat = 1.0 - score.p_value  # significance; higher = worse
            else:
                stat = score.excess / max(score.sigma, sigma_floor)  # n_sigma, floored
            stats.append(stat)
            if worst is None or stat > worst_stat:
                worst, worst_stat = score, stat
        episode_stat = max(stats, default=0.0)
        if not stats:
            # Q20: an episode that never hit the frontier seam has nothing to gate —
            # a pass here is "not scored", not "scored and clean". Flag it distinctly
            # (and log it) so it can't masquerade as a clean pass.
            _log.warning(
                "decision gate: episode %r had no %s events to score (reported unscored)",
                episode.episode_id,
                seam.value,
            )
        drifts.append(
            EpisodeDrift(
                episode_id=episode.episode_id,
                drift=episode_stat,
                diverged=episode_stat > threshold,
                divergence_seam=seam if episode_stat > threshold else None,
                divergence_distance=worst.divergence if worst else None,
                decision_divergence=worst.divergence if worst else None,
                sigma=worst.sigma if worst else None,
                scored=bool(stats),
            )
        )
    passed = bool(drifts) and _passes([d.drift for d in drifts], threshold, policy, quantile)
    return GateResult(
        passed=passed,
        threshold=threshold,
        policy=policy,
        per_episode=tuple(drifts),
        threshold_units="1 - p_value" if use_pvalue else "sigma",
    )
