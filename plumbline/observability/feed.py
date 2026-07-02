"""Flattened dashboard-feed builders (engineering spec §11).

The Grafana dashboards bind to these JSON rollups via the Infinity datasource (reads
JSON files/HTTP, no backend, no collector) — the default zero-service path. Three
feeds cover the two data families: `episode_telemetry` (per-seam / per-tick rollups
from a recorded episode), `gate_feed` (drift/divergence from a GateResult), and
`baseline_feed` (the Experiment-B green/red verdicts from a BaselineComparison).

JSON only (`canonical_dumps` / `json`), no pickle (invariant 3). Feeds carry
digests and rollups, never raw request payloads.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from plumbline.core.trace import Episode, JSONValue, SeamEvent, canonical_dumps
from plumbline.observability.baselines import BaselineComparison
from plumbline.proxy.otel import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    seam_event_attributes,
)
from plumbline.regression.gate import GateResult


def episode_telemetry(episode: Episode) -> dict[str, JSONValue]:
    """Per-seam latency + token rollups and per-tick seam counts for one episode."""
    by_seam: dict[str, list[SeamEvent]] = {}
    for event in episode.events:
        by_seam.setdefault(event.seam.value, []).append(event)
    seams: list[JSONValue] = []
    for seam_value, events in by_seam.items():
        latencies = sorted(event.latency_ms for event in events)
        row: dict[str, JSONValue] = {
            "seam": seam_value,
            "count": len(events),
            "latency_mean_ms": sum(latencies) / len(latencies),
            "latency_p95_ms": _percentile(latencies, 0.95),
        }
        input_tokens = _sum_tokens(events, GEN_AI_USAGE_INPUT_TOKENS)
        output_tokens = _sum_tokens(events, GEN_AI_USAGE_OUTPUT_TOKENS)
        if input_tokens is not None:  # present only when the recording carried usage
            row["input_tokens"] = input_tokens
        if output_tokens is not None:
            row["output_tokens"] = output_tokens
        seams.append(row)
    tick_counts: dict[int, int] = {}
    for event in episode.events:
        tick_counts[event.logical_tick] = tick_counts.get(event.logical_tick, 0) + 1
    ticks: list[JSONValue] = [
        {"logical_tick": tick, "seam_count": count} for tick, count in sorted(tick_counts.items())
    ]
    return {"episode_id": episode.episode_id, "seams": seams, "ticks": ticks}


def gate_feed(result: GateResult) -> dict[str, JSONValue]:
    """Drift / divergence rows from a GateResult for the regression dashboard."""
    return {
        "passed": result.passed,
        "threshold": result.threshold,
        "max_drift": result.max_drift,
        "diverged_fraction": result.diverged_fraction,
        "episodes": [
            {
                "episode_id": drift.episode_id,
                "drift": drift.drift,
                "diverged": drift.diverged,
                "divergence_seam": drift.divergence_seam.value if drift.divergence_seam else None,
                "divergence_distance": drift.divergence_distance,
            }
            for drift in result.per_episode
        ],
    }


def baseline_feed(comparison: BaselineComparison) -> dict[str, JSONValue]:
    """Experiment-B verdict rows (green/red) for the contrast panel."""
    return {
        "verdicts": [
            {
                "name": verdict.name,
                "healthy": verdict.healthy,
                "status": "green" if verdict.healthy else "red",
                "detail": verdict.detail,
            }
            for verdict in comparison.verdicts
        ],
        "caught_by": list(comparison.caught_by),
        "missed_by": list(comparison.missed_by),
    }


def write_feed(feed: Mapping[str, JSONValue], path: str | Path) -> None:
    Path(path).write_text(canonical_dumps(dict(feed)), encoding="utf-8")


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    # Nearest-rank, matching the gate's quantile convention (gate.py::_passes).
    index = min(len(sorted_values) - 1, max(0, math.ceil(quantile * len(sorted_values)) - 1))
    return sorted_values[index]


def _sum_tokens(events: Sequence[SeamEvent], attr_key: str) -> int | None:
    total = 0
    found = False
    for event in events:
        value = seam_event_attributes(event).get(attr_key)
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
            found = True
    return total if found else None
