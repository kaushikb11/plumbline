"""The Replayer interface and replay-result types (engineering spec §3.6, §6).

FROZEN (CLAUDE.md invariant 1): the `Replayer` method signatures and the
`ReplayResult`/`DivergencePolicy` types are the contract; the bodies are WS1.

Two modes: faithful replay serves every seam from the trace and must reproduce
behavior bit-identically; counterfactual replay re-executes a declared live
frontier and serves the rest, handling divergence per §6.

Halt-on-divergence is the default and divergence is a result, not an error
(CLAUDE.md invariant 5, §6.4): never silently serve a stale recorded response
past a divergence.
"""

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import ExactMatcher, Matcher
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import Payload, SeamEvent

# The language-bus loop is a linear pipeline (§3.1): each seam's input is the
# previous seam's output. Counterfactual replay relies on this ordering to know
# which seam first sees a changed input after a swap.
_PIPELINE: tuple[Seam, ...] = (
    Seam.SENSOR_TO_CAPTION,
    Seam.CAPTION_TO_FUSE,
    Seam.FUSE_TO_DECIDE,
    Seam.DECIDE_TO_ACT,
)


class DivergencePolicy(enum.Enum):
    """What to do when a downstream seam's live request diverges (§6.3)."""

    HALT = "halt"  # default: stop, mark episode diverged, report seam + distance
    GO_LIVE = "go_live"  # re-execute this seam and everything downstream live
    RECORD_NEW = "record_new"  # go live AND record a new trace branch (re-baselining)


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a replay run (§6.4).

    Records, per episode, the first seam where the live frontier diverged from
    the trace and by how much; the regression gate consumes this as attribution.

    NOTE: §6.4 mandates only the first-divergence seam and its distance; the
    `events` field (the reproduced seam-event sequence) is added as the natural
    output payload and is the field most likely to be refined.
    """

    episode_id: str
    diverged: bool
    divergence_seam: Seam | None  # first seam where live diverged from trace
    divergence_distance: float | None  # matcher distance at that seam
    events: tuple[SeamEvent, ...]  # reproduced seam events


class Replayer:
    def __init__(
        self,
        store: TraceStore,
        clock: VirtualClock,
        matchers: Mapping[Seam, Matcher],
    ) -> None:
        self._store = store
        self._clock = clock
        self._matchers = matchers

    def faithful(self, episode_id: str) -> ReplayResult:
        """Return the recorded event sequence — the serve-everything-from-trace
        reproduction baseline (§3.6).

        This method does not re-execute a runtime; it loads the trace. The
        determinism *guarantee* — that re-driving a runtime while serving each
        recorded response by request_digest reproduces the same decisions — is
        exercised by the proxy's per-request serving (`plumbline.proxy`) and
        verified end-to-end in `tests/test_reexecution.py`. A divergence here is
        impossible by construction; `diverged` is always False.
        """
        episode = self._store.load_episode(episode_id)
        self._clock.bind_replay(episode)
        return ReplayResult(
            episode_id=episode_id,
            diverged=False,
            divergence_seam=None,
            divergence_distance=None,
            events=episode.events,
        )

    def counterfactual(
        self,
        episode_id: str,
        live_frontier: set[Seam],
        # NOTE: §3.6 types overrides as `dict[Seam, Callable]`. The override re-
        # executes a seam — given a (live) request Payload it returns a response
        # Payload — so the element type is interpreted as Callable[[Payload],
        # Payload]. Mapping is used for immutability.
        overrides: Mapping[Seam, Callable[[Payload], Payload]],
        on_divergence: DivergencePolicy,
    ) -> ReplayResult:
        """Re-execute the live frontier; serve the rest from the trace, handling
        divergence per §6.

        For each tick, seams are walked in pipeline order. A seam in the live
        frontier is re-executed via its override. When a re-executed seam's output
        differs from the recording, the next downstream seam's input has changed;
        that seam's matcher compares the changed (live vs recorded) content. A
        mismatch is a divergence: under HALT (the default) the run stops, records
        the seam and distance, and serves nothing past it (invariant 5, §6.4).

        NOTE: comparing the swapped seam's live-vs-recorded *output* with the
        downstream seam's matcher is the linear-chain stand-in for comparing the
        downstream seam's live request — faithful for the toy loop where the
        caption flows directly into the fuse seam. Full downstream re-execution
        (GO_LIVE / RECORD_NEW past the first diverged seam) requires the runtime
        adapter to re-drive the loop, which pure-trace replay cannot do; those
        policies re-execute a seam only where an override exists, else serve the
        recorded content — bounded (§6.5). Even when they continue past a
        divergence, the FIRST divergence is still recorded in the result
        (invariant 5): a non-HALT run is never reported clean.

        Events within a tick are processed in recorded (seq) order, which is
        pipeline order (`_PIPELINE`); all events are served, including repeated
        calls at the same seam in one tick (e.g. multi-camera / multi-caption).
        """
        episode = self._store.load_episode(episode_id)
        self._clock.bind_replay(episode)

        ticks, by_tick = _group_by_tick(episode.events)
        served: list[SeamEvent] = []
        first_seam: Seam | None = None
        first_distance: float | None = None

        for tick in ticks:
            changed_live: Payload | None = None
            changed_recorded: Payload | None = None

            for recorded_event in by_tick[tick]:
                seam = recorded_event.seam

                if seam in live_frontier:
                    override = overrides.get(seam)
                    live_response = (
                        override(recorded_event.request)
                        if override is not None
                        else recorded_event.response
                    )
                    served.append(replace(recorded_event, response=live_response))
                    if live_response != recorded_event.response:
                        changed_live, changed_recorded = live_response, recorded_event.response
                    else:
                        changed_live = changed_recorded = None
                    continue

                # Seam served from trace. If an upstream output changed, validate
                # that this seam's recorded input still applies.
                if changed_live is not None and changed_recorded is not None:
                    matcher: Matcher = self._matchers.get(seam, ExactMatcher())
                    verdict = matcher.matches(changed_live, changed_recorded)
                    if not verdict.is_match:
                        if first_seam is None:  # record the first divergence, always
                            first_seam, first_distance = seam, verdict.distance
                        if on_divergence is DivergencePolicy.HALT:
                            return ReplayResult(
                                episode_id=episode_id,
                                diverged=True,
                                divergence_seam=seam,
                                divergence_distance=verdict.distance,
                                events=tuple(served),
                            )
                        # GO_LIVE / RECORD_NEW (bounded): re-execute via an override
                        # if one exists, else serve recorded; continue past.
                        downstream = overrides.get(seam)
                        if downstream is not None:
                            new_response = downstream(recorded_event.request)
                            served.append(replace(recorded_event, response=new_response))
                            if new_response != recorded_event.response:
                                changed_live, changed_recorded = (
                                    new_response,
                                    recorded_event.response,
                                )
                            else:
                                changed_live = changed_recorded = None
                        else:
                            served.append(recorded_event)
                            changed_live = changed_recorded = None
                    else:
                        # Recorded input still applies within tolerance.
                        served.append(recorded_event)
                        changed_live = changed_recorded = None
                else:
                    served.append(recorded_event)

        return ReplayResult(
            episode_id=episode_id,
            diverged=first_seam is not None,
            divergence_seam=first_seam,
            divergence_distance=first_distance,
            events=tuple(served),
        )


def _group_by_tick(
    events: tuple[SeamEvent, ...],
) -> tuple[list[int], dict[int, list[SeamEvent]]]:
    """Group events by logical tick, preserving tick order and, within a tick, the
    recorded (seq) order of every event — including repeated calls at one seam."""
    order: list[int] = []
    by_tick: dict[int, list[SeamEvent]] = {}
    for event in events:
        if event.logical_tick not in by_tick:
            by_tick[event.logical_tick] = []
            order.append(event.logical_tick)
        by_tick[event.logical_tick].append(event)
    return order, by_tick
