"""The Recorder (engineering spec §3.5).

FROZEN (CLAUDE.md invariant 1): the method signatures are the contract; the
bodies are WS1 implementation. The recorder writes to the store append-only per
episode.

NOTE on §3.5: the spec says the recorder "assigns seq and logical_tick" and
"canonicalizes payloads". When `record` receives a fully-formed `SeamEvent` (the
HTTP proxy / interceptor builds it from the live model I/O, including its own
digest and its `logical_tick` from the loop driver's `Context`), the recorder
must NOT alter that captured model I/O — doing so would break the zero-touch and
byte-identical-replay guarantees. So `record` persists the event as captured;
seq/tick assignment belongs to the event's producer. The virtual clock is the
*replay*-time tick source (`bind_replay`); record-time ticks originate from the
loop driver (`Context.logical_tick`, §3.4, §6), so the recorder does not advance
it. The `clock` dependency is retained for the frozen constructor contract.
"""

from collections.abc import Mapping

from plumbline.core.clock import VirtualClock
from plumbline.core.store import TraceStore
from plumbline.core.trace import (
    ConfigSnapshot,
    DigestMismatch,
    EpisodeManifest,
    JSONValue,
    SeamEvent,
    canonicalize,
)


class Recorder:
    def __init__(self, store: TraceStore, clock: VirtualClock) -> None:
        self._store = store
        self._clock = clock

    def record(self, event: SeamEvent) -> None:
        # Validate, never recompute: the recorder must not alter captured model I/O
        # (§3.5, zero-touch). A digest that does not match its request records fine
        # but is silently unreplayable, so reject it loudly here rather than let it
        # rot in the trace. The proxy computes digests correctly, so real recordings
        # pass; hand-built events should use make_seam_event (which computes it).
        expected = canonicalize(event.request).digest
        if event.request_digest != expected:
            raise DigestMismatch(
                episode_id=event.episode_id,
                seq=event.seq,
                recorded=event.request_digest,
                expected=expected,
            )
        self._store.append_event(event.episode_id, event)

    def open_episode(self, episode_id: str, metadata: Mapping[str, JSONValue]) -> None:
        # Empty config snapshot for now; the OM1 adapter populates runtime config
        # and model versions (§9.2). put_config content-addresses it under config/.
        config_hash = self._store.put_config(
            ConfigSnapshot(config_hash="", runtime_config={}, model_versions={})
        )
        self._store.open_episode(
            EpisodeManifest(
                episode_id=episode_id,
                metadata=metadata,
                config_hash=config_hash,
                seam_index=(),
            )
        )

    def close_episode(self, episode_id: str) -> None:
        self._store.close_episode(episode_id)
