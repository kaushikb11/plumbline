"""The real httpx-backed proxy server (engineering spec §4.2).

Drives the full pipeline in-process with no network and no real provider:

    runtime httpx client ── ASGITransport ──▶ proxy ASGI app ──▶ HttpxTransport
                                                                      │
                                       ASGITransport ◀── fake upstream ASGI app

Both ends are httpx `ASGITransport`s, so every real code path runs (httpx client,
ASGI request/response handling, forwarding, capture, recording) without a socket.
"""

import asyncio
import json
from typing import Any

import httpx
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.proxy.http import AsyncHTTPProxy
from plumbline.proxy.server import (
    ASGIApp,
    ASGIReceive,
    ASGIScope,
    ASGISend,
    HttpxTransport,
    make_asgi_app,
)

_UPSTREAM_BODY = {
    "id": "cmpl-1",
    "model": "gpt-4o",
    "choices": [{"message": {"role": "assistant", "content": "avoid"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
}


def _fake_upstream(received: list[bytes]) -> ASGIApp:
    async def app(scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        assert scope["type"] == "http"
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message.get("body"):
                chunks.append(message["body"])
            if not message.get("more_body", False):
                break
        received.append(b"".join(chunks))
        body = json.dumps(_UPSTREAM_BODY).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app


async def _run() -> tuple[Any, list[bytes], TraceStore]:
    received: list[bytes] = []
    upstream_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_upstream(received)))
    store = TraceStore()
    proxy = AsyncHTTPProxy(
        transport=HttpxTransport(upstream_client),
        recorder=Recorder(store, VirtualClock()),
        store=store,
    )
    proxy_app = make_asgi_app(proxy, upstream="http://api.openai.test", episode_id="ep")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app),
        base_url="http://proxy",
    ) as runtime:
        response = await runtime.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "go?"}],
            },
            headers={"x-plumbline-tick": "0"},
        )
    await upstream_client.aclose()
    return response.json(), received, store


def test_httpx_proxy_records_and_is_zero_touch() -> None:
    returned, received, store = asyncio.run(_run())

    # Zero-touch: the runtime receives exactly the upstream body.
    assert returned == _UPSTREAM_BODY
    # The proxy forwarded the request to the upstream, unaltered, exactly once.
    assert len(received) == 1
    assert json.loads(received[0])["model"] == "gpt-4o"
    # ...and recorded it as the Cortex decision seam, provider-tagged.
    events = store.load_episode("ep").events
    assert len(events) == 1
    assert events[0].seam is Seam.FUSE_TO_DECIDE
    assert events[0].model_id == "openai/gpt-4o"


def _sse_upstream() -> ASGIApp:
    # A real text/event-stream upstream, so the wire-detection at server.py's
    # HttpxTransport (content-type -> split_sse) is exercised end to end.
    chunks = (
        b'data: {"choices":[{"delta":{"content":"m"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ove"}}]}\n\n',
        b"data: [DONE]\n\n",
    )

    async def app(scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for i, chunk in enumerate(chunks):
            await send(
                {"type": "http.response.body", "body": chunk, "more_body": i < len(chunks) - 1}
            )

    return app


def test_sse_stream_captured_end_to_end_through_the_asgi_stack() -> None:
    async def run() -> TraceStore:
        upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=_sse_upstream()))
        store = TraceStore()
        proxy = AsyncHTTPProxy(
            transport=HttpxTransport(upstream),
            recorder=Recorder(store, VirtualClock()),
            store=store,
        )
        app = make_asgi_app(proxy, upstream="http://api.openai.test", episode_id="ep")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as runtime:
            resp = await runtime.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": "?"}],
                },
                headers={"x-plumbline-tick": "0"},
            )
        # Zero-touch: the runtime received the exact SSE bytes.
        assert b"[DONE]" in resp.content
        await upstream.aclose()
        return store

    store = asyncio.run(run())
    # The stream was captured (not dropped) — the assembled OpenAI content survives.
    events = store.load_episode("ep").events
    assert len(events) == 1
    text = json.dumps(events[0].response.inline)
    assert "move" in text  # the reassembled delta stream ("m" + "ove")
