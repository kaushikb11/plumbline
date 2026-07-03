"""Regression gate — CI for robot behavior (engineering spec §8)."""

from plumbline.regression.gate import (
    Config,
    DecisionGate,
    EpisodeDrift,
    FailurePolicy,
    GateResult,
    GateSpec,
    gate,
)
from plumbline.regression.golden import (
    BehaviorLabel,
    GoldenEpisode,
    GoldenSet,
    action_sequence,
)

__all__ = [
    "BehaviorLabel",
    "Config",
    "DecisionGate",
    "EpisodeDrift",
    "FailurePolicy",
    "GateResult",
    "GateSpec",
    "GoldenEpisode",
    "GoldenSet",
    "action_sequence",
    "gate",
]
