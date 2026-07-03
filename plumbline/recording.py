"""Integrated recording coordinator — produce a full four-seam episode from a driven
run (§4, §9; limitations gap #3).

The zero-touch HTTP proxy records only the model seams it sees per call
(SENSOR_TO_CAPTION / FUSE_TO_DECIDE). `RecordingCoordinator` plugs in as the proxy's
recorder and, using the adapter's `reconstruct_*` hooks, fills in the two derived
seams around each Cortex call — CAPTION_TO_FUSE (before) and DECIDE_TO_ACT (after) —
in correct per-tick pipeline order, so the counterfactual/gate can run on an episode
the recorder itself produced (not hand-built fixtures).

It IS a `RecordingSession` (so it drops straight into the proxy and owns the shared
seq/tick), with `record()` overridden to interleave the reconstructions.
"""

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent
from plumbline.session import RecordingSession


class ReconstructingAdapter(Protocol):
    """The reconstruction hooks the coordinator needs (OM1Adapter / G1Adapter /
    GenericAgentAdapter satisfy it). Defined here, not in the frozen base.py."""

    def reconstruct_caption_to_fuse(
        self,
        *,
        episode_id: str,
        seq: int,
        logical_tick: int,
        captions: Sequence[JSONValue],
        fused_prompt: JSONValue,
        wall_ts: float = 0.0,
    ) -> SeamEvent: ...

    def reconstruct_decide_to_act(
        self,
        *,
        episode_id: str,
        seq: int,
        logical_tick: int,
        decision_response: Payload,
        wall_ts: float = 0.0,
    ) -> SeamEvent: ...


CaptionExtractor = Callable[[Payload], JSONValue]


def default_caption_text(response: Payload) -> JSONValue:
    """Pull the caption from an OpenAI-shaped VLM response (choices[0].message.content),
    else a `caption` field, else the whole inline content."""
    inline = response.inline
    if isinstance(inline, dict):
        choices = inline.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and "content" in message:
                    return message["content"]
        if "caption" in inline:
            return inline["caption"]
    return inline


class RecordingCoordinator(RecordingSession):
    """A RecordingSession that reconstructs the CAPTION_TO_FUSE and DECIDE_TO_ACT seams
    around each FUSE_TO_DECIDE model call, yielding a full four-seam episode."""

    def __init__(
        self,
        store: TraceStore,
        *,
        episode_id: str,
        adapter: ReconstructingAdapter,
        metadata: Mapping[str, JSONValue] | None = None,
        caption_extractor: CaptionExtractor = default_caption_text,
        reconstruct_action: bool = True,
    ) -> None:
        super().__init__(store, episode_id=episode_id, metadata=metadata)
        self._adapter = adapter
        self._caption_extractor = caption_extractor
        self._reconstruct_action = reconstruct_action
        self._captions: dict[int, list[JSONValue]] = {}
        self._buffer_lock = threading.Lock()  # separate from the session lock (non-reentrant)
        self._store_ref = store

    @property
    def store(self) -> TraceStore:
        return self._store_ref

    def record(self, event: SeamEvent) -> None:
        # Mirror the proxy-stamped tick so a bus sample on the tap thread inherits it.
        self.set_tick(event.logical_tick)
        if event.seam is Seam.SENSOR_TO_CAPTION:
            with self._buffer_lock:
                self._captions.setdefault(event.logical_tick, []).append(
                    self._caption_extractor(event.response)
                )
            super().record(event)
        elif event.seam is Seam.FUSE_TO_DECIDE:
            with self._buffer_lock:
                captions = self._captions.pop(event.logical_tick, None)
            # CAPTION_TO_FUSE is emitted BEFORE the FUSE event so per-tick seq order is
            # pipeline order (SENSOR -> CAPTION -> FUSE -> DECIDE).
            if captions:
                super().record(
                    self._adapter.reconstruct_caption_to_fuse(
                        episode_id=self.episode_id,
                        seq=0,  # reassigned by RecordingSession.record
                        logical_tick=event.logical_tick,
                        captions=captions,
                        fused_prompt=event.request.inline,
                        wall_ts=event.wall_ts,
                    )
                )
            super().record(event)
            if self._reconstruct_action:
                super().record(
                    self._adapter.reconstruct_decide_to_act(
                        episode_id=self.episode_id,
                        seq=0,
                        logical_tick=event.logical_tick,
                        decision_response=event.response,
                        wall_ts=event.wall_ts,
                    )
                )
        else:
            # DECIDE_TO_ACT from a real bus tap, or anything else: record as-is.
            super().record(event)
