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

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import httpx

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.proxy.http import AsyncHTTPProxy, HTTPRequest, HTTPResponse
from plumbline.proxy.streaming import CapturedStream, split_sse

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
                stream = CapturedStream(split_sse(raw.decode("utf-8")))
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
        ctx = Context(
            episode_id=episode_id,
            model_id=None,
            params={},
            logical_tick=_parse_tick(headers.get(_TICK_HEADER)),
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
            logical_tick=_parse_tick(headers.get(_TICK_HEADER)),
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


def _parse_tick(value: str | None) -> int:
    """Parse the loop-tick header defensively; a malformed value falls back to 0."""
    try:
        return int(value) if value is not None else 0
    except ValueError:
        return 0


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
