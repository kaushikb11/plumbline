"""Fidelity metrics (engineering spec §7) — the scientific core.

Everything is anchored to downstream decision success and corrected for the
decision-maker's own sampling noise (the decision-stability floor), never to
caption surface similarity. Action-plan binning (the §14.6 action-equivalence
question) is an injected `label`, lossless by default.

§14.5 (render(G)) and §14.6 (salient, action equivalence) are open decisions that
can flatter the metric; they are surfaced (not hidden) in `metrics.py` and
`judge.py` and require human review.
"""

from plumbline.fidelity.decision import (
    DeciderFn,
    DecisionLabel,
    Distribution,
    Divergence,
    canonical_label,
    decision_distribution,
    decision_stability,
    histogram,
    jensen_shannon,
    sample_labels,
    self_divergence,
    total_variation,
)
from plumbline.fidelity.judge import (
    JudgeModel,
    JudgeVerdict,
    behavioral_equivalence_prompt,
    judge_noise_floor,
    semantic_equivalence,
    structural_equivalence,
)
from plumbline.fidelity.metrics import (
    DecisionDrift,
    caption_loss,
    decision_drift,
    default_salient,
    fusion_loss,
    salient_artifact,
)

__all__ = [
    "DeciderFn",
    "DecisionDrift",
    "DecisionLabel",
    "Distribution",
    "Divergence",
    "JudgeModel",
    "JudgeVerdict",
    "behavioral_equivalence_prompt",
    "canonical_label",
    "caption_loss",
    "decision_drift",
    "decision_distribution",
    "decision_stability",
    "default_salient",
    "fusion_loss",
    "histogram",
    "jensen_shannon",
    "judge_noise_floor",
    "salient_artifact",
    "sample_labels",
    "self_divergence",
    "semantic_equivalence",
    "structural_equivalence",
    "total_variation",
]
