"""Golden episodes (engineering spec §8.1).

A versioned set of recorded episodes whose behavior has been accepted as good,
stored as full traces (in the TraceStore) plus an accepted-behavior summary — the
action sequence and any success label.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonical_dumps, canonicalize


def action_sequence(events: Sequence[SeamEvent]) -> tuple[Payload, ...]:
    """An episode's behavior: its ordered DECIDE_TO_ACT action plans (§8.1)."""
    return tuple(event.request for event in events if event.seam is Seam.DECIDE_TO_ACT)


@dataclass(frozen=True)
class BehaviorLabel:
    """Accepted behavior for a golden episode (§8.1): the action sequence, plus an
    optional task-success label."""

    actions: tuple[Payload, ...]
    success: bool | None = None


@dataclass(frozen=True)
class GoldenEpisode:
    episode_id: str
    label: BehaviorLabel


class GoldenSet:
    """A versioned set of episodes accepted as good (§8.1)."""

    def __init__(self, store: TraceStore) -> None:
        self._store = store
        self._episodes: dict[str, GoldenEpisode] = {}

    def add(
        self,
        episode_id: str,
        *,
        success: bool | None = None,
        label: BehaviorLabel | None = None,
    ) -> None:
        """Accept an episode. Its behavior is captured from the recorded trace's
        DECIDE_TO_ACT seam unless an explicit `label` is supplied."""
        if label is None:
            actions = action_sequence(self._store.load_episode(episode_id).events)
            label = BehaviorLabel(actions=actions, success=success)
        self._episodes[episode_id] = GoldenEpisode(episode_id, label)

    def episodes(self) -> tuple[GoldenEpisode, ...]:
        return tuple(self._episodes[key] for key in sorted(self._episodes))

    def version(self) -> str:
        """Content hash of the set — episode ids + accepted behavior + labels (§8.1)."""
        summary: list[JSONValue] = []
        for episode in self.episodes():
            # Hash the canonical digest (inline AND content-addressed blobs), not
            # just inline, so a blob-only difference cannot collide to one version.
            actions: list[JSONValue] = [
                canonicalize(action).digest for action in episode.label.actions
            ]
            summary.append([episode.episode_id, actions, episode.label.success])
        return hashlib.sha256(canonical_dumps(summary).encode("utf-8")).hexdigest()
