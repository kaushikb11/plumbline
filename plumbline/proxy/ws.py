"""WebSocket record/replay capture (§4.2; limitations gap #1).

OM1's reference perception delivers caption/transcript RESULTS over a WebSocket
(wss://api.openmind.com), invisible to the HTTP-only proxy. This captures that WS
result stream: each inbound caption frame becomes one SENSOR_TO_CAPTION SeamEvent, so
per-caption fidelity and the captioner-swap counterfactual keep working. Frames are
relayed UNALTERED in record mode (the zero-touch invariant); replay serves the
recorded frames in seq order without an upstream.

Transport-agnostic: the WS client is injected as a `WsTransport` (like `AsyncTransport`
for HTTP), so the substrate carries no `websockets` dependency. Binary frames use the
content-addressed blob path (no pickle). The RTSP video UPLOAD is a separate media
transport and out of scope (docs/limitations.md gap #1).
"""

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import BlobKind, JSONValue, Payload, SeamEvent, canonicalize

_WS_ENDPOINT_KEY = "plumbline.ws_endpoint"
_WS_DIRECTION_KEY = "plumbline.ws_direction"
_WS_OPCODE_KEY = "plumbline.ws_opcode"
_WS_FRAME_SEQ_KEY = "plumbline.ws_frame_seq"
_WS_TEXT_KEY = "plumbline.ws_text"
_WS_BINARY_KEY = "plumbline.ws_binary"


@dataclass(frozen=True)
class WsFrame:
    kind: Literal["text", "bytes", "close"]
    text: str | None = None
    data: bytes | None = None
    code: int | None = None  # set for kind == "close"


class WsConnection(Protocol):
    """Both the client side and an upstream connection satisfy this."""

    async def send(self, frame: WsFrame) -> None: ...
    async def recv(self) -> WsFrame: ...  # returns a kind=="close" frame at end of stream
    async def close(self, code: int = 1000) -> None: ...


class WsTransport(Protocol):
    """The real WS client the proxy dials upstream through (record mode) — injected
    so the substrate carries no `websockets` dependency."""

    async def connect(
        self, url: str, *, subprotocols: Sequence[str], headers: Mapping[str, str]
    ) -> WsConnection: ...


class _NoUpstreamWsTransport:
    """Replay transport: never connects (replay serves recorded frames, no upstream)."""

    async def connect(
        self, url: str, *, subprotocols: Sequence[str], headers: Mapping[str, str]
    ) -> WsConnection:
        raise RuntimeError("replay ws server does not connect to an upstream")


class AsyncWSProxy:
    def __init__(
        self,
        *,
        transport: WsTransport,
        recorder: Recorder,
        store: TraceStore,
        episode_metadata: Mapping[str, JSONValue] | None = None,
    ) -> None:
        self._transport = transport
        self._recorder = recorder
        self._store = store
        self._episode_metadata: Mapping[str, JSONValue] = episode_metadata or {}
        self._opened: set[str] = set()
        self._seq: dict[str, int] = {}
        self._frame_ordinal: dict[str, int] = {}

    def _ensure_open(self, episode_id: str) -> None:
        if episode_id not in self._opened:
            self._recorder.open_episode(episode_id, self._episode_metadata)
            self._opened.add(episode_id)
            self._seq[episode_id] = 0
            self._frame_ordinal[episode_id] = 0

    async def record(
        self,
        client: WsConnection,
        ctx: Context,
        *,
        upstream_url: str,
        endpoint: str,
        seam: Seam = Seam.SENSOR_TO_CAPTION,
        subprotocols: Sequence[str] = (),
        headers: Mapping[str, str] | None = None,
        model_id: str | None = None,
    ) -> None:
        """Relay frames between client and a freshly-dialed upstream, recording each.
        Frames reach the client UNALTERED (zero-touch)."""
        self._ensure_open(ctx.episode_id)
        upstream = await self._transport.connect(
            upstream_url, subprotocols=subprotocols, headers=headers or {}
        )

        async def client_to_upstream() -> None:
            while True:
                frame = await client.recv()
                if frame.kind == "close":
                    await upstream.close(frame.code or 1000)
                    return
                await upstream.send(frame)
                self._record_frame(ctx, endpoint, seam, frame, "outbound", model_id)

        async def upstream_to_client() -> None:
            while True:
                frame = await upstream.recv()
                await client.send(frame)  # UNALTERED
                if frame.kind == "close":
                    return
                self._record_frame(ctx, endpoint, seam, frame, "inbound", model_id)

        up = asyncio.create_task(client_to_upstream())
        down = asyncio.create_task(upstream_to_client())
        done, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()  # surface any exception

    async def replay(
        self,
        client: WsConnection,
        ctx: Context,
        *,
        endpoint: str,
        seam: Seam = Seam.SENSOR_TO_CAPTION,
    ) -> None:
        """Serve the recorded frame sequence for this endpoint in seq order; an
        outbound event consumes one client frame (its subscribe/keepalive)."""
        events = [
            event
            for event in self._store.load_episode(ctx.episode_id).events
            if event.seam is seam and _endpoint_of(event) == endpoint
        ]
        events.sort(key=lambda event: event.seq)
        for event in events:
            if event.params.get(_WS_DIRECTION_KEY) == "outbound":
                await client.recv()  # consume the runtime's subscribe/keepalive
            else:
                await client.send(self._response_to_frame(event.response))
        await client.close(1000)

    def _record_frame(
        self,
        ctx: Context,
        endpoint: str,
        seam: Seam,
        frame: WsFrame,
        direction: str,
        model_id: str | None,
    ) -> None:
        ordinal = self._frame_ordinal[ctx.episode_id]
        self._frame_ordinal[ctx.episode_id] = ordinal + 1
        # WS frames are server-pushed with no per-frame request; a synthetic, unique
        # request keeps each frame individually addressable and un-collapsible.
        request = Payload(
            inline={
                _WS_ENDPOINT_KEY: endpoint,
                _WS_DIRECTION_KEY: direction,
                _WS_FRAME_SEQ_KEY: ordinal,
            }
        )
        seq = self._seq[ctx.episode_id]
        self._seq[ctx.episode_id] = seq + 1
        self._recorder.record(
            SeamEvent(
                episode_id=ctx.episode_id,
                seq=seq,
                seam=seam,
                logical_tick=ctx.logical_tick,
                wall_ts=time.time(),
                request=request,
                response=self._frame_to_response(frame),
                model_id=model_id,
                params={
                    _WS_DIRECTION_KEY: direction,
                    _WS_ENDPOINT_KEY: endpoint,
                    _WS_OPCODE_KEY: frame.kind,
                },
                request_digest=canonicalize(request).digest,
                latency_ms=0.0,
            )
        )

    def _frame_to_response(self, frame: WsFrame) -> Payload:
        if frame.kind == "text":
            # Stored VERBATIM. Parsing to JSON and re-serializing on replay would
            # alter the frame's bytes wherever the server's formatting differs from
            # json.dumps defaults (compact separators, key order), breaking
            # byte-identical replay. Consumers parse the text themselves.
            return Payload(inline={_WS_TEXT_KEY: frame.text or ""})
        ref = self._store.put_blob(frame.data or b"", BlobKind.BIN)  # content-addressed, no pickle
        return Payload(inline={_WS_BINARY_KEY: f"blob:{ref.sha256}"}, blobs=(ref,))

    def _response_to_frame(self, response: Payload) -> WsFrame:
        inline = response.inline
        assert isinstance(inline, dict)
        if _WS_TEXT_KEY in inline:
            value = inline[_WS_TEXT_KEY]
            return WsFrame(kind="text", text=value if isinstance(value, str) else json.dumps(value))
        return WsFrame(kind="bytes", data=self._store.get_blob(response.blobs[0]))


def _endpoint_of(event: SeamEvent) -> JSONValue:
    endpoint = event.params.get(_WS_ENDPOINT_KEY)
    if endpoint is not None:
        return endpoint
    inline = event.request.inline
    return inline.get(_WS_ENDPOINT_KEY) if isinstance(inline, dict) else None
