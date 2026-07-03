"""The virtual clock (engineering spec §3.4).

FROZEN (CLAUDE.md invariant 1): the method signatures are the frozen contract;
the bodies are WS1.

Determinism envelope (§3.4, CLAUDE.md invariant 4): Plumbline guarantees that on
replay every model call receives the recorded request and returns the recorded
response, so the sequence of decisions and actions is reproduced. It controls
the runtime's internal scheduler ONLY if an adapter exposes a clock hook. Absent
that hook, loop timing may vary while model I/O does not. This is deterministic
model-I/O replay, NOT deterministic wall-clock scheduling.
"""

from plumbline.core.trace import Episode


class VirtualClock:
    # NOTE: the frozen interface declares no constructor; this no-arg __init__
    # only initializes internal counters and does not change the contract.
    def __init__(self) -> None:
        self._tick = 0
        # When bound to an episode, ticks are served from the recording rather
        # than from wall-derived advancement (§3.4).
        self._replay: tuple[int, ...] | None = None
        self._cursor = 0

    def now_tick(self) -> int:
        if self._replay is not None and self._cursor < len(self._replay):
            return self._replay[self._cursor]
        return self._tick

    def advance(self) -> int:
        if self._replay is not None:
            tick = self.now_tick()
            self._cursor += 1
            return tick
        self._tick += 1
        return self._tick

    def bind_replay(self, episode: Episode) -> None:
        """Serve recorded ticks from `episode` during replay (§3.4).

        NOTE (honesty): the Replayer *binds* the clock but reconstructs a tick's
        seam grouping from `SeamEvent.logical_tick` + the seq-sort in
        `TraceStore.load_episode`, not by calling `advance()`/`now_tick()` here.
        So on the current replay paths this object is bound but not stepped —
        `now_tick`/`advance` exist for a runtime driver that wants to *read* the
        recorded tick stream, and are the seam a clock-hook adapter would use.
        Correct ordering does not depend on this object being stepped."""
        self._replay = tuple(event.logical_tick for event in episode.events)
        self._cursor = 0
