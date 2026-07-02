"""Observability (engineering spec §11) — baseline comparison for Experiment B."""

from plumbline.observability.baselines import (
    BaselineComparison,
    MonitorVerdict,
    compare_against_baselines,
    generic_tracer_monitor,
    latency_monitor,
    plumbline_behavior_monitor,
)
from plumbline.observability.feed import (
    baseline_feed,
    episode_telemetry,
    gate_feed,
    write_feed,
)
from plumbline.observability.otlp import episode_to_otlp, event_to_otlp_span, write_otlp
from plumbline.observability.trace_diff import (
    StepDiff,
    StepStatus,
    TraceDiff,
    diff_episodes,
    diff_traces,
)

__all__ = [
    "BaselineComparison",
    "MonitorVerdict",
    "StepDiff",
    "StepStatus",
    "TraceDiff",
    "baseline_feed",
    "compare_against_baselines",
    "diff_episodes",
    "diff_traces",
    "episode_telemetry",
    "episode_to_otlp",
    "event_to_otlp_span",
    "gate_feed",
    "generic_tracer_monitor",
    "latency_monitor",
    "plumbline_behavior_monitor",
    "write_feed",
    "write_otlp",
]
