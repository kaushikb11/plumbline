"""Trace-diff viewer (engineering spec §11).

Given two runs — two episodes, or a faithful and a counterfactual run of the same
episode — show where they diverged and which seam introduced the divergence. The
gate report already names the diverging seam; this is the side-by-side view behind
it, and the visual the Experiment-B demo uses.

Runs are aligned by (logical_tick, seam) — not by seq, which differs between runs —
so a swapped seam shows as CHANGED and the downstream seams a halted counterfactual
never reached show as ONLY_A (present in the golden, missing in the candidate).
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from plumbline.core.seam import Seam
from plumbline.core.trace import Episode, SeamEvent

_SEAM_ORDER: dict[Seam, int] = {seam: index for index, seam in enumerate(Seam)}


class StepStatus(enum.Enum):
    SAME = "same"
    CHANGED = "changed"  # present in both, but request and/or response differ
    ONLY_A = "only_a"  # present in the first run, missing in the second
    ONLY_B = "only_b"  # present in the second run, missing in the first


@dataclass(frozen=True)
class StepDiff:
    logical_tick: int
    seam: Seam
    status: StepStatus
    request_changed: bool
    response_changed: bool


@dataclass(frozen=True)
class TraceDiff:
    steps: tuple[StepDiff, ...]

    @property
    def identical(self) -> bool:
        return all(step.status is StepStatus.SAME for step in self.steps)

    @property
    def first_divergence(self) -> StepDiff | None:
        """The first seam (in tick/pipeline order) where the runs differ (§6.4)."""
        return next((step for step in self.steps if step.status is not StepStatus.SAME), None)

    def as_text(self) -> str:
        first = self.first_divergence
        lines: list[str] = []
        for step in self.steps:
            marker = "  <- first divergence" if step is first else ""
            detail = ""
            if step.status is StepStatus.CHANGED:
                parts = [
                    p
                    for p, c in (
                        ("request", step.request_changed),
                        ("response", step.response_changed),
                    )
                    if c
                ]
                detail = f" ({' & '.join(parts)} changed)"
            status = step.status.value.upper()
            lines.append(
                f"tick {step.logical_tick}  {step.seam.value:<18} {status}{detail}{marker}"
            )
        if not lines:
            return "both runs empty"
        return "\n".join(lines)


def diff_traces(a: Sequence[SeamEvent], b: Sequence[SeamEvent]) -> TraceDiff:
    """Align two event sequences by (logical_tick, seam) and diff them step by step."""
    index_a = _index(a)
    index_b = _index(b)
    keys = sorted(set(index_a) | set(index_b), key=lambda key: (key[0], _SEAM_ORDER[key[1]]))

    steps: list[StepDiff] = []
    for tick, seam in keys:
        events_a = index_a.get((tick, seam), [])
        events_b = index_b.get((tick, seam), [])
        for occurrence in range(max(len(events_a), len(events_b))):
            event_a = events_a[occurrence] if occurrence < len(events_a) else None
            event_b = events_b[occurrence] if occurrence < len(events_b) else None
            steps.append(_step(tick, seam, event_a, event_b))
    return TraceDiff(tuple(steps))


def diff_episodes(a: Episode, b: Episode) -> TraceDiff:
    return diff_traces(a.events, b.events)


def _index(events: Sequence[SeamEvent]) -> dict[tuple[int, Seam], list[SeamEvent]]:
    by_key: dict[tuple[int, Seam], list[SeamEvent]] = {}
    for event in events:
        by_key.setdefault((event.logical_tick, event.seam), []).append(event)
    return by_key


def _step(tick: int, seam: Seam, event_a: SeamEvent | None, event_b: SeamEvent | None) -> StepDiff:
    if event_a is None:
        return StepDiff(tick, seam, StepStatus.ONLY_B, False, False)
    if event_b is None:
        return StepDiff(tick, seam, StepStatus.ONLY_A, False, False)
    request_changed = event_a.request != event_b.request
    response_changed = event_a.response != event_b.response
    status = StepStatus.CHANGED if (request_changed or response_changed) else StepStatus.SAME
    return StepDiff(tick, seam, status, request_changed, response_changed)
