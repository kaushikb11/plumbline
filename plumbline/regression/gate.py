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
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import ExactMatcher, Matcher
from plumbline.core.replayer import DivergencePolicy, Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import Payload
from plumbline.fidelity import structural_equivalence
from plumbline.regression.golden import GoldenSet, action_sequence

_EXACT_MATCHER: Matcher = ExactMatcher()


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
    drift: float  # behavioral distance from golden (§8.3), 0.0 .. 1.0
    diverged: bool
    divergence_seam: Seam | None  # per-seam attribution (§6.4)
    divergence_distance: float | None


@dataclass(frozen=True)
class GateResult:
    passed: bool
    threshold: float
    policy: FailurePolicy
    per_episode: tuple[EpisodeDrift, ...]

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


def gate(
    store: TraceStore,
    golden: GoldenSet,
    config: Config,
    drift_threshold: float,
    *,
    behavior_matcher: Matcher = _EXACT_MATCHER,
    policy: FailurePolicy = FailurePolicy.ANY,
    quantile: float = 0.95,
) -> GateResult:
    """Run the gate over the golden set under `config` (§8.2).

    NOTE (§14.6, HUMAN REVIEW): `behavior_matcher` is the per-step action
    equivalence and it sets the gate's false-positive rate. ExactMatcher is the
    strict default; a NumericToleranceMatcher (for pose/coordinate plans) or the
    adapter's ActionSchema-derived matcher is the open choice.
    """
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
