"""Decision distributions and the noise floor (engineering spec §7.1, §7.2).

For an input context `x`, the decision-maker (the Cortex LLM, or a fixed probe
function in controlled experiments) induces a distribution `D(x)` over decisions
— action-plan classes — because it samples at temperature. `D(x)` is estimated by
drawing N samples and binning each by a `label` (its action-equivalence class).

Decisions are compared by a divergence `div(P, Q)`: total variation is the
default for discrete typed action plans; Jensen-Shannon is available for soft
distributions (§7.1).

The decision-stability noise floor `sigma(x)` is the decider's self-disagreement
across samples (§7.2): the divergence between two independent N-sample estimates of
`D(x)`. As N grows both converge to `D(x)`, so `sigma -> 0` and scales as
~1/sqrt(N). Every fidelity gap is credited only beyond `sigma`. `decision_stability`
draws 2N samples and splits them into two N-halves so the floor is measured at the
SAME sample size as the caption/fusion-loss numerator (two full-N distributions).

REPLAY CAVEAT: these metrics need N *independent* temperature samples, so they are
record-mode / live-mode. Under by-request-digest faithful replay, N identical
prompts collapse to one recorded response, so `D(x)` becomes a point mass and
`sigma -> 0` — do not compute fidelity metrics against a faithfully-replayed
decider (the same caveat as the judge's noise floor, judge.py).

JUDGMENT CALLS (CLAUDE.md short-leash, §7 — flagged, not silent):
  - Binning is the §14.6 action-equivalence question (OPEN in the spec). The
    class of a plan is decided by an injected `label`. The default,
    `canonical_label`, is lossless — the generic core never bins lossily on its
    own. Continuous action plans (e.g. move(x,y,yaw)) need a runtime-specific
    equivalence (type + numeric tolerance via the adapter's ActionSchema),
    supplied by the caller that owns §14.6; exact-canonical binning is degenerate
    for them (every distinct float becomes its own class).
  - Divergence: total_variation default per §7.1.
  - sigma: §7.2's `E[div(half1, half2)]` is an expectation; it is estimated by
    averaging the split-half divergence over `trials` seeded random partitions,
    not one arbitrary split — this is what makes the ~1/sqrt(N) convergence clean.
"""

import math
import random
from collections.abc import Callable, Mapping, Sequence

from plumbline.core.trace import JSONValue, canonical_dumps

# A normalized {action-plan class label: probability} distribution.
Distribution = Mapping[str, float]
Divergence = Callable[[Distribution, Distribution], float]
# The decision-maker: a context (e.g. the fused prompt) -> an action plan.
DeciderFn = Callable[[str], Mapping[str, JSONValue]]
# The action-equivalence class of a decision (the §14.6 binning).
DecisionLabel = Callable[[Mapping[str, JSONValue]], str]


def canonical_label(decision: Mapping[str, JSONValue]) -> str:
    """Default, lossless action-equivalence class: the full canonical action plan.

    NOTE (§14.6): degenerate for continuous action plans — inject a coarser
    `label` (type + numeric tolerance from the runtime's ActionSchema) for those.
    """
    return canonical_dumps(dict(decision))


# --- divergences (§7.1) -----------------------------------------------------


def total_variation(p: Distribution, q: Distribution) -> float:
    """Total variation distance: 0.0 (identical) .. 1.0 (disjoint support)."""
    keys = p.keys() | q.keys()
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def jensen_shannon(p: Distribution, q: Distribution) -> float:
    """Jensen-Shannon divergence in bits: 0.0 (identical) .. 1.0 (disjoint)."""
    keys = p.keys() | q.keys()
    m: dict[str, float] = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(a: Distribution, b: Distribution) -> float:
    total = 0.0
    for key, a_prob in a.items():
        if a_prob > 0.0:
            b_prob = b.get(key, 0.0)
            if b_prob > 0.0:
                total += a_prob * math.log2(a_prob / b_prob)
    return total


# --- decision distribution (§7.1) -------------------------------------------


def histogram(labels: Sequence[str]) -> dict[str, float]:
    """Normalized empirical distribution over class labels."""
    n = len(labels)
    if n == 0:
        return {}
    counts: dict[str, float] = {}
    for item in labels:
        counts[item] = counts.get(item, 0.0) + 1.0
    return {key: count / n for key, count in counts.items()}


def sample_labels(
    decider: DeciderFn, context: str, n: int, *, label: DecisionLabel = canonical_label
) -> list[str]:
    return [label(decider(context)) for _ in range(n)]


def decision_distribution(
    decider: DeciderFn, context: str, n: int, *, label: DecisionLabel = canonical_label
) -> Distribution:
    """Estimate `D(context)` from N samples of the decider (§7.1)."""
    return histogram(sample_labels(decider, context, n, label=label))


# --- noise floor (§7.2) -----------------------------------------------------


def self_divergence(
    labels: Sequence[str],
    *,
    divergence: Divergence = total_variation,
    trials: int = 32,
    seed: int = 0,
) -> float:
    """Split-half self-divergence of a fixed label sample — the §7.2 estimator core.

    Averages `div(half1, half2)` over `trials` seeded random partitions to estimate
    the expectation. Reused for the decision-stability floor (§7.2) and the
    behavioral judge's own noise floor (§7.5).
    """
    half = len(labels) // 2
    if half == 0:
        return 0.0
    rng = random.Random(seed)
    acc = 0.0
    for _ in range(trials):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        acc += divergence(histogram(shuffled[:half]), histogram(shuffled[half:]))
    return acc / trials


def decision_stability(
    decider: DeciderFn,
    context: str,
    n: int,
    *,
    label: DecisionLabel = canonical_label,
    divergence: Divergence = total_variation,
    trials: int = 32,
    seed: int = 0,
) -> float:
    """Estimate the noise floor `sigma(context)` at decision-sample size N (§7.2).

    Draws 2N samples and splits them into two independent N-halves, so sigma
    estimates E[div(D_N, D_N)] — the divergence between two same-distribution
    N-samples. That matches the sample size of the caption/fusion-loss numerator
    (which compares two full-N distributions); estimating sigma from N/2 halves of
    a single N sample would make the floor ~sqrt(2) too large and under-report
    small real losses.
    """
    labels = sample_labels(decider, context, 2 * n, label=label)
    return self_divergence(labels, divergence=divergence, trials=trials, seed=seed)
