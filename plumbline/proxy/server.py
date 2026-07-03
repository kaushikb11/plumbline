"""Real HTTP proxy server (engineering spec §4.2, §14.2).

The transport-agnostic `AsyncHTTPProxy` (`plumbline.proxy.http`) gets a concrete
network client here:

  - `HttpxTransport` forwards a captured request to the real provider over HTTPS
    via httpx and captures the response (including SSE streams).
  - `make_asgi_app` exposes the proxy (record mode) as an ASGI application the
    runtime points a base URL at (`OPENAI_BASE_URL=http://localhost:8900/v1`).

No TLS interception is needed in this reverse-proxy model, because the base URL is
redirected by config (§4.2); MITM with a locally trusted CA (§14.2) is the
documented fallback for runtimes that cannot change their base URL.

Requires the optional `httpx` dependency:  pip install "plumbline[proxy]".
The rest of `plumbline.proxy` does not import httpx, so the core stays light.
"""

import importlib
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from typing import Any

import httpx

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue
from plumbline.proxy.http import AsyncHTTPProxy, HTTPRequest, HTTPResponse
from plumbline.proxy.streaming import CapturedStream, split_sse
from plumbline.proxy.tick import _TICK_OVERRIDE_KEY
from plumbline.proxy.ws import AsyncWSProxy, WsConnection, WsFrame, _NoUpstreamWsTransport

_JSON_CONTENT_TYPE = "application/json"

ASGIScope = MutableMapping[str, Any]
ASGIMessage = MutableMapping[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]

# Hop-by-hop / length / encoding headers we let the transport recompute rather
# than forward verbatim (a forwarded content-length or gzip content-encoding would
# not match the re-emitted body).
_TICK_HEADER = "x-plumbline-tick"
_DROP_REQUEST_HEADERS = frozenset(
    # ...plus the internal loop-tick header, so it is never forwarded upstream.
    {"host", "content-length", "connection", "transfer-encoding", "accept-encoding", _TICK_HEADER}
)
_DROP_RESPONSE_HEADERS = frozenset({"content-length", "content-encoding", "transfer-encoding"})


class HttpxTransport:
    """An `AsyncTransport` backed by httpx — forwards to the real upstream (§4.2)."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _DROP_REQUEST_HEADERS
        }
        upstream = self._client.build_request(
            request.method, request.url, headers=headers, content=request.body
        )
        response = await self._client.send(upstream, stream=True)
        try:
            forwarded = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in _DROP_RESPONSE_HEADERS
            }
            raw = await response.aread()
            if "text/event-stream" in response.headers.get("content-type", ""):
                # errors="replace": a non-utf-8 byte in the SSE stream must not raise
                # (that would 500 the runtime — a zero-touch leak). The framing is
                # captured for replay; the raw bytes are preserved on the HTTPResponse.
                stream = CapturedStream(split_sse(raw.decode("utf-8", errors="replace")))
                return HTTPResponse(response.status_code, forwarded, raw, stream)
            return HTTPResponse(response.status_code, forwarded, raw, None)
        finally:
            await response.aclose()


def make_asgi_app(proxy: AsyncHTTPProxy, *, upstream: str, episode_id: str) -> ASGIApp:
    """Expose `proxy` (record mode) as an ASGI app. Point the runtime's base URL at
    it; it forwards to `upstream` (the scheme+host of the real provider).

    The loop-iteration index is read from the `x-plumbline-tick` request header if
    present, so counterfactual replay can group seams by tick (§6); otherwise 0. A
    real adapter stamps it per loop iteration.
    """
    base = upstream.rstrip("/")

    async def app(scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        # ASGI servers open a "lifespan" scope at startup and may open "websocket";
        # only "http" is handled. Ignore the rest rather than raising, which would
        # noise up (or disable) server startup.
        if scope["type"] != "http":
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        query = scope["query_string"].decode("latin-1")
        url = base + scope["path"] + (f"?{query}" if query else "")
        request = HTTPRequest(
            method=scope["method"],
            url=url,
            headers=headers,
            body=await _read_body(receive),
        )
        # The tick header is an explicit OVERRIDE (carried in params); when absent the
        # proxy's tick_policy derives the tick from the seam sequence. `None` means
        # "no override" — distinct from an explicit tick 0.
        override = _parse_tick(headers.get(_TICK_HEADER))
        params: dict[str, JSONValue] = {} if override is None else {_TICK_OVERRIDE_KEY: override}
        ctx = Context(
            episode_id=episode_id,
            model_id=None,
            params=params,
            logical_tick=override if override is not None else 0,
        )
        await _send_response(send, await proxy.record(request, ctx))

    return app


class _NoUpstreamTransport:
    """Placeholder transport for a replay-only server: it never forwards, because
    replay serves recorded responses from the trace and must not hit an upstream."""

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        raise RuntimeError("replay server does not forward to an upstream")


def make_replay_asgi_app(store: TraceStore, *, episode_id: str) -> ASGIApp:
    """Expose faithful replay as an ASGI app: point the runtime's base URL at it and
    it serves the recorded response for each request (matched by request_digest),
    never forwarding upstream. A request with no recorded match returns 404 — it
    does not fabricate a response.
    """
    proxy = AsyncHTTPProxy(
        transport=_NoUpstreamTransport(), recorder=Recorder(store, VirtualClock()), store=store
    )

    async def app(scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope["type"] != "http":
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        query = scope["query_string"].decode("latin-1")
        request = HTTPRequest(
            method=scope["method"],
            url=scope["path"] + (f"?{query}" if query else ""),
            headers=headers,
            body=await _read_body(receive),
        )
        ctx = Context(
            episode_id=episode_id,
            model_id=None,
            params={},
            logical_tick=_parse_tick(headers.get(_TICK_HEADER)) or 0,  # unused on replay
        )
        try:
            response = await proxy.replay(request, ctx)
        except KeyError:
            response = HTTPResponse(
                status=404,
                headers={"content-type": _JSON_CONTENT_TYPE},
                body=b'{"error":"no recorded response for this request"}',
                stream=None,
            )
        await _send_response(send, response)

    return app


def _parse_tick(value: str | None) -> int | None:
    """Parse the loop-tick header defensively. None = header absent (no override);
    a malformed value also yields None (fall back to the tick policy)."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


async def _read_body(receive: ASGIReceive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        chunk = message.get("body", b"")
        if chunk:
            chunks.append(chunk)
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _send_response(send: ASGISend, response: HTTPResponse) -> None:
    header_list = [
        (key.encode("latin-1"), value.encode("latin-1")) for key, value in response.headers.items()
    ]
    start: ASGIMessage = {
        "type": "http.response.start",
        "status": response.status,
        "headers": header_list,
    }
    await send(start)
    # An empty stream (chunks == ()) must still send exactly one terminal body
    # message, or the ASGI response is incomplete — hence the `and .chunks` guard.
    if response.stream is not None and response.stream.chunks:
        chunks = response.stream.chunks
        for index, chunk in enumerate(chunks):
            body: ASGIMessage = {
                "type": "http.response.body",
                "body": chunk.encode("utf-8"),
                "more_body": index < len(chunks) - 1,
            }
            await send(body)
    else:
        await send({"type": "http.response.body", "body": response.body})


# --- WebSocket server (§4.2; limitations gap #1) ----------------------------


class _ASGIClientSocket:
    """Adapts an ASGI websocket (receive/send) to the WsConnection protocol. The
    connect handshake (accept) is done by the app before this wraps the channel."""

    def __init__(self, receive: ASGIReceive, send: ASGISend) -> None:
        self._receive = receive
        self._send = send

    async def send(self, frame: WsFrame) -> None:
        if frame.kind == "text":
            await self._send({"type": "websocket.send", "text": frame.text})
        elif frame.kind == "bytes":
            await self._send({"type": "websocket.send", "bytes": frame.data})
        else:
            await self._send({"type": "websocket.close", "code": frame.code or 1000})

    async def recv(self) -> WsFrame:
        message = await self._receive()
        if message["type"] == "websocket.receive":
            text = message.get("text")
            if text is not None:
                return WsFrame(kind="text", text=text)
            return WsFrame(kind="bytes", data=message.get("bytes") or b"")
        return WsFrame(kind="close", code=message.get("code", 1000))  # websocket.disconnect

    async def close(self, code: int = 1000) -> None:
        await self._send({"type": "websocket.close", "code": code})


class WebsocketsTransport:  # pragma: no cover - thin `websockets` wrapper
    """Concrete WsTransport: dials an upstream WS via the `websockets` library
    (lazily imported, the only place it is used — keeps it an optional dep)."""

    async def connect(
        self, url: str, *, subprotocols: Sequence[str], headers: Mapping[str, str]
    ) -> WsConnection:
        # Dynamic import so mypy --strict passes identically whether or not the
        # optional `websockets` extra is installed (a static import needs a
        # type-ignore that becomes an unused-ignore error once it IS installed).
        websockets: Any = importlib.import_module("websockets")

        connection = await websockets.connect(
            url, subprotocols=list(subprotocols) or None, additional_headers=dict(headers)
        )
        return _WebsocketsConnection(connection)


class _WebsocketsConnection:  # pragma: no cover - thin `websockets` wrapper
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def send(self, frame: WsFrame) -> None:
        if frame.kind == "text":
            await self._connection.send(frame.text)
        elif frame.kind == "bytes":
            await self._connection.send(frame.data)
        else:
            await self._connection.close(frame.code or 1000)

    async def recv(self) -> WsFrame:
        try:
            message = await self._connection.recv()
        except Exception:
            return WsFrame(kind="close", code=1000)
        if isinstance(message, str):
            return WsFrame(kind="text", text=message)
        return WsFrame(kind="bytes", data=message)

    async def close(self, code: int = 1000) -> None:
        await self._connection.close(code)


async def _accept_ws(receive: ASGIReceive, send: ASGISend) -> None:
    await receive()  # websocket.connect
    await send({"type": "websocket.accept"})


def make_ws_asgi_app(
    proxy: AsyncWSProxy,
    *,
    upstream: str,
    episode_id: str,
    seam: Seam = Seam.SENSOR_TO_CAPTION,
    model_id: str | None = None,
) -> ASGIApp:
    """Record a WebSocket: accept the client, then relay to/from `upstream` while
    capturing each frame (§4.2). Non-websocket scopes are ignored."""

    async def app(scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") != "websocket":
            return
        await _accept_ws(receive, send)
        client = _ASGIClientSocket(receive, send)
        ctx = Context(episode_id=episode_id, model_id=None, params={}, logical_tick=0)
        await proxy.record(
            client,
            ctx,
            upstream_url=upstream,
            endpoint=scope.get("path", ""),
            seam=seam,
            model_id=model_id,
        )

    return app


def make_ws_replay_asgi_app(
    store: TraceStore, *, episode_id: str, seam: Seam = Seam.SENSOR_TO_CAPTION
) -> ASGIApp:
    """Replay a recorded WebSocket caption stream to the client; never connects
    upstream (`_NoUpstreamWsTransport`)."""
    proxy = AsyncWSProxy(
        transport=_NoUpstreamWsTransport(), recorder=Recorder(store, VirtualClock()), store=store
    )

    async def app(scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") != "websocket":
            return
        await _accept_ws(receive, send)
        client = _ASGIClientSocket(receive, send)
        ctx = Context(episode_id=episode_id, model_id=None, params={}, logical_tick=0)
        await proxy.replay(client, ctx, endpoint=scope.get("path", ""), seam=seam)

    return app


def make_dispatch_asgi_app(*, http: ASGIApp, ws: ASGIApp) -> ASGIApp:
    """One ASGI app that routes http scopes to `http` and websocket scopes to `ws`."""

    async def app(scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") == "websocket":
            await ws(scope, receive, send)
        else:
            await http(scope, receive, send)

    return app
