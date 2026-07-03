"""Automatic logical-tick sourcing for the recording proxy (§6).

An out-of-process runtime (e.g. the OM1 Go binary) does not send the
`x-plumbline-tick` header, so without a tick source every recorded event lands at
tick 0 and the counterfactual/gate lose the per-tick grouping they need. This module
supplies a `TickPolicy` the proxy calls per recorded seam to derive `logical_tick`
from the seam sequence itself — no runtime cooperation required. An explicit header
override always wins (surfaced via `ctx.params[_TICK_OVERRIDE_KEY]`).

The tick is a *logical* cycle index, not wall time (invariant 4).
"""

import threading
from dataclasses import dataclass, field
from typing import Protocol

from plumbline.core.seam import Seam

# Namespaced control key: an explicit per-request tick override, carried in
# ctx.params and stripped before the event is recorded (like _HTTP_STATUS_KEY).
_TICK_OVERRIDE_KEY = "plumbline.tick_override"


class TickPolicy(Protocol):
    """Derives the logical tick for a recorded seam. `override` is the explicit
    header value when present (and must win), else None."""

    def next_tick(self, seam: Seam, override: int | None) -> int: ...


@dataclass
class BoundaryTickPolicy:
    """Advance `logical_tick` when a tick-boundary seam starts a new perception →
    action cycle (default boundary: SENSOR_TO_CAPTION).

    Consecutive boundary seams share a tick (multi-camera / multi-caption). A loop
    with no boundary seam (e.g. a pure decide loop) collapses to one tick — configure
    a different `boundary_seam` or send the `x-plumbline-tick` header for such
    runtimes. An explicit override wins and resyncs the counter.
    """

    boundary_seam: Seam = Seam.SENSOR_TO_CAPTION
    _tick: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)
    _prev_was_boundary: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def next_tick(self, seam: Seam, override: int | None) -> int:
        with self._lock:
            if override is not None:
                self._tick = override
                self._started = True
                self._prev_was_boundary = seam == self.boundary_seam
                return override
            if seam == self.boundary_seam:
                if not self._started:
                    self._started = True
                    self._prev_was_boundary = True
                    return self._tick  # first cycle is tick 0
                if not self._prev_was_boundary:
                    self._tick += 1  # a new perception cycle
                self._prev_was_boundary = True
                return self._tick
            # A non-boundary seam stays within the current cycle.
            self._started = True
            self._prev_was_boundary = False
            return self._tick
