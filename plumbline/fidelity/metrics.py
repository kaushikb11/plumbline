"""Caption and fusion fidelity (engineering spec §7.3, §7.4).

Both metrics are anchored to downstream decision success and corrected for the
decision-maker's own sampling noise (the §7.2 floor) — never surface text
similarity. The action-equivalence binning (§14.6) is threaded through as `label`
so caption/fusion loss bin decisions exactly as the noise floor does.

================================  HUMAN REVIEW  ================================
This module contains the two open decisions §14.5 and §14.6, both of which can
*flatter the metric* if chosen carelessly. They are surfaced here, not buried,
and require human review before Experiment A/C depend on them:

  * render(G) — the oracle context for caption_loss (§7.3, §14.5). This module
    does NOT construct render(G); the caller (the sim adapter) supplies it. That
    is deliberate: a render(G) phrased the way a *good captioner* would describe
    the scene makes caption_loss artificially small (the caption "agrees" with an
    oracle written in caption-like prose). An honest render(G) is a caption-
    agnostic structured description of ground truth (object poses, agent state),
    identical regardless of which captioner is under test. The metric cannot
    enforce this — it is the adapter's responsibility and a review gate.

  * salient(C_i) and weights — the fusion re-emphasis (§7.4, §14.6). The default
    `default_salient` plainly restates the caption; `weights` are uniform. Risks:
    a salient that flips the decision via emphasis/repetition/position artifacts
    rather than genuine information re-injection OVERSTATES fusion loss; a salient
    too weak to surface dropped content UNDERSTATES it; non-uniform weights that
    downweight inconvenient captions flatter the result. Both are injectable so
    the choice is explicit at the call site, not hidden in the metric.
===============================================================================
"""

from collections.abc import Callable, Sequence

from plumbline.fidelity.decision import (
    DeciderFn,
    DecisionLabel,
    Divergence,
    canonical_label,
    decision_distribution,
    decision_stability,
    total_variation,
)


def default_salient(caption: str) -> str:
    """Plainly restate a caption for re-emphasis (§7.4).

    Deliberately minimal: it re-injects the caption's content without rephrasing
    it into a decision. Replace it at the call site if your decider needs a
    different re-emphasis — and review whether the replacement flips decisions for
    reasons other than the caption's information (§14.6).
    """
    return f"Additionally, note this observation: {caption}"


def caption_loss(
    decider: DeciderFn,
    caption: str,
    oracle_context: str,
    n: int,
    *,
    label: DecisionLabel = canonical_label,
    divergence: Divergence = total_variation,
) -> float:
    """Caption fidelity loss (§7.3):

        caption_loss(C) = max(0, div(D(C), D(render(G))) - sigma)

    How much acting on the caption diverges from acting on ground truth, beyond
    the decision-maker's own noise. `oracle_context` IS render(G) and is supplied
    by the caller (§14.5 — see the module HUMAN REVIEW banner); sigma is computed
    at the oracle input, per §7.3.
    """
    d_caption = decision_distribution(decider, caption, n, label=label)
    d_oracle = decision_distribution(decider, oracle_context, n, label=label)
    sigma = decision_stability(decider, oracle_context, n, label=label, divergence=divergence)
    return max(0.0, divergence(d_caption, d_oracle) - sigma)


def fusion_loss(
    decider: DeciderFn,
    fused_prompt: str,
    captions: Sequence[str],
    n: int,
    *,
    salient: Callable[[str], str] = default_salient,
    weights: Sequence[float] | None = None,
    label: DecisionLabel = canonical_label,
    divergence: Divergence = total_variation,
) -> float:
    """Fusion fidelity loss (§7.4):

        fusion_loss = sum_i weight_i * max(0, div(D(F), D(F + salient(C_i))) - sigma)

    with weight_i = 1/k by default (uniform AND normalized), so the result is the
    MEAN per-caption fidelity loss, bounded in [0, 1] and comparable across episodes
    with different caption counts. (This resolves §7.4's "uniform by default" to
    normalized; the raw unbounded sum is available by passing explicit `weights`.)

    If re-adding C_i beyond the noise floor changes the decision, the Fuser
    dropped task-relevant information from C_i. `salient` and `weights` are the
    §14.6 judgment calls (see the module HUMAN REVIEW banner); both injectable.
    """
    if weights is not None and len(weights) != len(captions):
        raise ValueError(f"weights length {len(weights)} != captions length {len(captions)}")
    count = len(captions)
    if count == 0:
        return 0.0
    d_f = decision_distribution(decider, fused_prompt, n, label=label)
    sigma = decision_stability(decider, fused_prompt, n, label=label, divergence=divergence)
    total = 0.0
    for index, caption in enumerate(captions):
        weight = (1.0 / count) if weights is None else weights[index]
        d_augmented = decision_distribution(
            decider, f"{fused_prompt} {salient(caption)}", n, label=label
        )
        total += weight * max(0.0, divergence(d_f, d_augmented) - sigma)
    return total


def salient_artifact(
    decider: DeciderFn,
    fused_prompt_with_caption: str,
    caption: str,
    n: int,
    *,
    salient: Callable[[str], str] = default_salient,
    label: DecisionLabel = canonical_label,
    divergence: Divergence = total_variation,
) -> float:
    """Self-consistency guard for the salient operation (§7.4, §14.6 — HUMAN REVIEW).

    Run with a fused prompt that ALREADY contains `caption`'s information. Returns
    the decision change attributable to re-emphasizing it, beyond the noise floor:

        max(0, div(D(F), D(F + salient(caption))) - sigma)

    It should be ~0: re-injecting information already present must not change the
    decision. A positive value means `salient` flips decisions via emphasis,
    repetition, or position/vocabulary artifacts rather than genuine information —
    which makes fusion_loss OVERSTATE the Fuser's loss (a flattering metric).
    Run this against the real Cortex decider for each (salient, decider) pair
    before trusting fusion_loss; fix the salient if it is non-zero.

    Computationally this is a single-caption fusion_loss term where the caption is
    known to already be present, so any positive result is pure salient artifact,
    not dropped information.
    """
    d_f = decision_distribution(decider, fused_prompt_with_caption, n, label=label)
    sigma = decision_stability(
        decider, fused_prompt_with_caption, n, label=label, divergence=divergence
    )
    d_augmented = decision_distribution(
        decider, f"{fused_prompt_with_caption} {salient(caption)}", n, label=label
    )
    return max(0.0, divergence(d_f, d_augmented) - sigma)
