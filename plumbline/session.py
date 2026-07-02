"""Recording session — coordinate all four seams into one episode (§3.5, §4, §9).

A real OM1 run records the model seams through the HTTP proxy and the action seam
through the Zenoh tap. Independently they collide: each producer assigns its own
`seq`, so the merged episode has no coherent ordering. `RecordingSession` is the
single owner of the episode — it hands every producer a globally-monotonic `seq`
and the current loop `logical_tick`, so the proxy, the tap, and the adapter's
CAPTION_TO_FUSE reconstruction all write one coherent trace (§3.2; the shared tick
is what lets counterfactual replay group a tick's seams, §6).

It IS a `Recorder` (so it drops straight into the proxy) with three changes:
the episode lifecycle is idempotent (the proxy's auto-open won't truncate a trace
the tap is already writing), `record` reassigns a shared `seq`, and it carries the
loop tick. A lock guards the shared counter because the tap fires on a Zenoh
thread while the proxy runs on the event loop.

Usage (in-process; the caller drives the loop):

    session = RecordingSession(store, episode_id="go2-001", metadata={"robot": "go2"})
    session.open()
    proxy = RecordingProxy(model_call, session)          # session IS the recorder
    adapter.bus_tap().subscribe(session.record_bus_sample)  # action seam
    for index, frame in enumerate(run):
        session.set_tick(index)
        caption = proxy.forward(vision_request, session.context(model_id="openai/vlm"))
        session.record(adapter.reconstruct_caption_to_fuse(...))   # CAPTION_TO_FUSE
        proxy.forward(fused_prompt, session.context(model_id="openai/cortex"))
        # the action plan arrives on the bus -> record_bus_sample (DECIDE_TO_ACT)
    session.close()

For the zero-touch HTTP proxy, the runtime supplies the tick via the
`x-plumbline-tick` header instead of `context()`; `record` still assigns the
shared seq, so ordering stays coherent.
"""

import itertools
import threading
from collections.abc import Mapping
from dataclasses import replace

from plumbline.adapters.base import BusSample
from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize


class RecordingSession(Recorder):
    def __init__(
        self,
        store: TraceStore,
        *,
        episode_id: str,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> None:
        super().__init__(store, VirtualClock())
        self._episode_id = episode_id
        self._metadata: Mapping[str, JSONValue] = metadata if metadata is not None else {}
        self._seq = itertools.count()
        self._tick = 0
        self._opened = False
        self._closed = False
        self._lock = threading.Lock()

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def logical_tick(self) -> int:
        return self._tick

    # --- loop tick (the caller drives it once per iteration) -----------------

    def set_tick(self, tick: int) -> None:
        with self._lock:
            self._tick = tick

    def advance_tick(self) -> int:
        with self._lock:
            self._tick += 1
            return self._tick

    def context(
        self,
        *,
        model_id: str | None = None,
        params: Mapping[str, JSONValue] | None = None,
    ) -> Context:
        """A Context stamped with this session's episode id and current tick, for
        the in-process proxy."""
        return Context(
            episode_id=self._episode_id,
            model_id=model_id,
            params=params if params is not None else {},
            logical_tick=self._tick,
        )

    # --- lifecycle (idempotent: the session owns the one episode) -------------

    def open(self) -> None:
        self.open_episode(self._episode_id, self._metadata)

    def close(self) -> None:
        self.close_episode(self._episode_id)

    def open_episode(self, episode_id: str, metadata: Mapping[str, JSONValue]) -> None:
        # A producer's episode_id/metadata are ignored: the session owns the
        # episode, and opening is done once so an auto-opening proxy cannot
        # truncate a trace the tap is already writing.
        with self._lock:
            if not self._opened:
                super().open_episode(self._episode_id, self._metadata)
                self._opened = True

    def close_episode(self, episode_id: str) -> None:
        with self._lock:
            if self._opened and not self._closed:
                super().close_episode(self._episode_id)
                self._closed = True

    # --- recording: every producer routes here for a shared seq --------------

    def record(self, event: SeamEvent) -> None:
        with self._lock:
            if self._closed:
                # A bus sample in flight after close() (the tap fires on a Zenoh
                # thread): drop it rather than crash the tap thread on a sealed
                # episode. The late action is intentionally not recorded. A record
                # BEFORE open() is a caller-ordering bug and is left to surface, not
                # silently dropped (which would hide a lost first action).
                return
            super().record(replace(event, seq=next(self._seq)))

    def record_bus_sample(self, sample: BusSample, *, seam: Seam = Seam.DECIDE_TO_ACT) -> None:
        """Record an observed bus message (an action plan) as a seam event —
        wire this to `BusTap.subscribe` (§4.3)."""
        request = Payload(inline=sample.payload)
        self.record(
            SeamEvent(
                episode_id=self._episode_id,
                seq=0,  # reassigned by record()
                seam=seam,
                logical_tick=self._tick,
                wall_ts=sample.wall_ts,
                request=request,
                response=request,  # the bus message is the action as issued
                model_id=None,
                params={},
                request_digest=canonicalize(request).digest,
                latency_ms=0.0,
            )
        )
