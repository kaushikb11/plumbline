"""SSE streaming pass-through in the record proxy (engineering spec §4.2, §14.3).

The record proxy must forward a streamed SSE completion to the runtime chunk-by-chunk
(preserving time-to-first-token for a robot consuming a streamed decision) WITHOUT
sacrificing byte-identity of the recording or the zero-touch invariant. These tests
drive `make_asgi_app` in-process with a fake streaming transport that yields SSE chunks
one at a time, capturing the ASGI `send` messages, and prove:

  (a) STREAMS        - the client receives one body message per chunk as it ARRIVES
                       (more_body=True), then a terminal more_body=False; not one blob.
  (b) BYTE-IDENTICAL - the recorded response is identical to a buffered recording of the
                       same chunks (same assembled body + `split_sse` framing).
  (c) REPLAY         - faithful replay re-serves the recorded SSE framing.
  (d) ZERO-TOUCH     - a failing recorder never robs the client of its chunks.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.proxy.http import AsyncHTTPProxy, HTTPRequest, HTTPResponse
from plumbline.proxy.server import ASGIMessage, make_asgi_app, make_replay_asgi_app
from plumbline.proxy.streaming import CapturedStream, split_sse

_SSE_CHUNKS = (
    b'data: {"choices":[{"delta":{"content":"m"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"ove"}}]}\n\n',
    b"data: [DONE]\n\n",
)
_REQ_BODY = json.dumps(
    {"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "?"}]}
).encode("utf-8")


class _FakeStreamingTransport:
    """Yields the SSE chunks ONE AT A TIME from `stream()` (the streaming path), and
    also serves the whole buffered response from `send()` (so an equivalent buffered
    recording can be produced for the byte-identity comparison)."""

    def __init__(
        self, chunks: tuple[bytes, ...], *, content_type: str = "text/event-stream"
    ) -> None:
        self._chunks = chunks
        self._content_type = content_type

    def _captured(self, raw: bytes) -> CapturedStream | None:
        if "text/event-stream" in self._content_type:
            return CapturedStream(split_sse(raw.decode("utf-8", errors="replace")))
        return None

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        raw = b"".join(self._chunks)
        return HTTPResponse(200, {"content-type": self._content_type}, raw, self._captured(raw))

    async def stream(
        self, request: HTTPRequest
    ) -> tuple[int, Mapping[str, str], AsyncIterator[bytes]]:
        chunks = self._chunks

        async def gen() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        return 200, {"content-type": self._content_type}, gen()


class _BufferingTransport:
    """Has NO `stream()` method, so the proxy falls back to the buffered path."""

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        raw = b"".join(self._chunks)
        stream = CapturedStream(split_sse(raw.decode("utf-8", errors="replace")))
        return HTTPResponse(200, {"content-type": "text/event-stream"}, raw, stream)


async def _drive(app: object, body: bytes = _REQ_BODY) -> list[ASGIMessage]:
    """Drive an ASGI app in-process, capturing every `send` message."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"x-plumbline-tick", b"0")],
    }
    inbound = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[ASGIMessage] = []

    async def receive() -> ASGIMessage:
        return inbound.pop(0) if inbound else {"type": "http.disconnect"}

    async def send(message: ASGIMessage) -> None:
        sent.append(message)

    await app(scope, receive, send)  # type: ignore[operator]
    return sent


def _bodies(sent: list[ASGIMessage]) -> list[ASGIMessage]:
    return [m for m in sent if m["type"] == "http.response.body"]


def _new_proxy(transport: object, store: TraceStore) -> AsyncHTTPProxy:
    return AsyncHTTPProxy(
        transport=transport,  # type: ignore[arg-type]
        recorder=Recorder(store, VirtualClock()),
        store=store,
    )


# --- (a) STREAMS: chunks reach the client as they arrive --------------------


def test_client_receives_chunks_as_a_stream_not_one_blob() -> None:
    store = TraceStore()
    proxy = _new_proxy(_FakeStreamingTransport(_SSE_CHUNKS), store)
    app = make_asgi_app(proxy, upstream="http://api.openai.test", episode_id="ep")

    sent = asyncio.run(_drive(app))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200
    bodies = _bodies(sent)
    # One body message per upstream chunk (each more_body=True), then a terminal
    # empty more_body=False message: N + 1, NOT a single buffered blob.
    assert len(bodies) == len(_SSE_CHUNKS) + 1
    assert [m["body"] for m in bodies[:-1]] == list(_SSE_CHUNKS)
    assert all(m["more_body"] is True for m in bodies[:-1])
    assert bodies[-1]["body"] == b""
    assert bodies[-1]["more_body"] is False
    # And the runtime saw the full byte stream, in order.
    assert b"".join(m["body"] for m in bodies) == b"".join(_SSE_CHUNKS)


# --- (b) BYTE-IDENTICAL: streamed recording == buffered recording -----------


def test_streamed_recording_is_byte_identical_to_buffered_recording() -> None:
    stream_store = TraceStore()
    stream_proxy = _new_proxy(_FakeStreamingTransport(_SSE_CHUNKS), stream_store)
    stream_app = make_asgi_app(stream_proxy, upstream="http://api.openai.test", episode_id="ep")
    asyncio.run(_drive(stream_app))

    # An equivalent BUFFERED recording of the same chunks: a transport without
    # `stream()` routes through the original `record()` path unchanged.
    buffered_store = TraceStore()
    buffered_proxy = _new_proxy(_BufferingTransport(_SSE_CHUNKS), buffered_store)
    buffered_app = make_asgi_app(buffered_proxy, upstream="http://api.openai.test", episode_id="ep")
    asyncio.run(_drive(buffered_app))

    streamed = stream_store.load_episode("ep").events
    buffered = buffered_store.load_episode("ep").events
    assert len(streamed) == len(buffered) == 1
    # Identical assembled body AND identical split_sse framing.
    assert streamed[0].response.inline == buffered[0].response.inline
    assert streamed[0].request_digest == buffered[0].request_digest
    # The framing is exactly split_sse over the joined raw, wire-chunking notwithstanding.
    raw = b"".join(_SSE_CHUNKS).decode("utf-8")
    assert streamed[0].response.inline["chunks"] == list(split_sse(raw))  # type: ignore[index,call-overload]


# --- (c) REPLAY: faithful replay re-serves the recorded SSE framing ---------


def test_faithful_replay_restreams_recorded_framing() -> None:
    store = TraceStore()
    proxy = _new_proxy(_FakeStreamingTransport(_SSE_CHUNKS), store)
    app = make_asgi_app(proxy, upstream="http://api.openai.test", episode_id="ep")
    asyncio.run(_drive(app))
    proxy.close("ep")

    replay_app = make_replay_asgi_app(store, episode_id="ep")
    sent = asyncio.run(_drive(replay_app))

    assert sent[0]["type"] == "http.response.start"
    bodies = _bodies(sent)
    raw = b"".join(_SSE_CHUNKS).decode("utf-8")
    framing = split_sse(raw)
    # Replay emits one body message per recorded SSE event block (the split_sse framing),
    # and the concatenation reproduces the recorded stream byte-for-byte.
    assert [m["body"] for m in bodies] == [block.encode("utf-8") for block in framing]
    assert b"".join(m["body"] for m in bodies).decode("utf-8") == raw


# --- (d) ZERO-TOUCH: a failing recorder never robs the client ---------------


class _FailingRecordProxy(AsyncHTTPProxy):
    """A proxy whose recording step raises, to prove the streaming path serves the
    client regardless and never propagates the fault to the ASGI server."""

    async def record_prefetched(
        self, request: HTTPRequest, ctx: Context, response: HTTPResponse, latency_ms: float
    ) -> None:
        raise RuntimeError("simulated recorder failure")


def test_client_gets_all_chunks_even_if_recording_raises() -> None:
    store = TraceStore()
    proxy = _FailingRecordProxy(
        transport=_FakeStreamingTransport(_SSE_CHUNKS),
        recorder=Recorder(store, VirtualClock()),
        store=store,
    )
    app = make_asgi_app(proxy, upstream="http://api.openai.test", episode_id="ep")

    # No exception propagates out of the ASGI app...
    sent = asyncio.run(_drive(app))

    # ...and the client still received every chunk, in order, fully framed.
    bodies = _bodies(sent)
    assert [m["body"] for m in bodies[:-1]] == list(_SSE_CHUNKS)
    assert bodies[-1]["more_body"] is False
    assert b"".join(m["body"] for m in bodies) == b"".join(_SSE_CHUNKS)


# --- non-SSE through the streaming transport stays buffered/byte-identical ---


def test_non_sse_through_streaming_transport_is_single_message_and_identical() -> None:
    body = json.dumps({"id": "r", "choices": [{"message": {"content": "avoid"}}]}).encode("utf-8")
    store = TraceStore()
    proxy = _new_proxy(_FakeStreamingTransport((body,), content_type="application/json"), store)
    app = make_asgi_app(proxy, upstream="http://api.openai.test", episode_id="ep")

    sent = asyncio.run(_drive(app))

    bodies = _bodies(sent)
    # Non-SSE is buffered into a single body message (byte-identity is trivial).
    assert len(bodies) == 1
    assert bodies[0]["body"] == body
    events = store.load_episode("ep").events
    assert len(events) == 1
    assert "avoid" in json.dumps(events[0].response.inline)
