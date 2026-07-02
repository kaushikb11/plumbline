"""AsyncHTTPProxy serve-from-trace replay (engineering spec §4.2, §14.3).

The deployed replay path (faithful-by-digest and counterfactual-HALT), HTTP status
capture/reconstruction, non-JSON error bodies not crashing record(), and SSE
chunk-framing round-trip. Driven via asyncio.run (no pytest-asyncio). Previously
untested — this is the actual serve-from-trace path a live replay uses.
"""

import asyncio
import json
from collections.abc import Mapping

import pytest
from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy
from plumbline.core.store import TraceStore
from plumbline.proxy import (
    AsyncHTTPProxy,
    CapturedStream,
    HTTPRequest,
    HTTPResponse,
    ProxyDivergence,
)

_REQ = {"model": "gpt-4o", "messages": [{"role": "user", "content": "go?"}]}
_RESP = {"id": "r", "choices": [{"message": {"role": "assistant", "content": "avoid"}}]}


class _FakeTransport:
    def __init__(self, response: HTTPResponse) -> None:
        self.response = response

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        return self.response


def _request(body: Mapping[str, object]) -> HTTPRequest:
    return HTTPRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=json.dumps(body).encode("utf-8"),
    )


def _record(upstream: HTTPResponse) -> tuple[AsyncHTTPProxy, Context, HTTPRequest]:
    store = TraceStore()
    proxy = AsyncHTTPProxy(
        transport=_FakeTransport(upstream), recorder=Recorder(store, VirtualClock()), store=store
    )
    ctx = Context(episode_id="ep", model_id=None, params={})
    request = _request(_REQ)
    asyncio.run(proxy.record(request, ctx))
    proxy.close("ep")
    return proxy, ctx, request


def _json_response(body: Mapping[str, object], status: int = 200) -> HTTPResponse:
    return HTTPResponse(
        status=status,
        headers={"content-type": "application/json"},
        body=json.dumps(body).encode("utf-8"),
    )


def test_faithful_replay_by_digest_returns_recorded_response() -> None:
    proxy, ctx, request = _record(_json_response(_RESP))
    replayed = asyncio.run(proxy.replay(request, ctx))
    assert replayed.status == 200
    assert json.loads(replayed.body) == _RESP


def test_replay_reconstructs_non_200_status() -> None:
    proxy, ctx, request = _record(_json_response({"error": "rate limited"}, status=429))
    replayed = asyncio.run(proxy.replay(request, ctx))
    assert replayed.status == 429  # captured, not hardcoded to 200


def test_record_does_not_crash_on_non_json_body() -> None:
    upstream = HTTPResponse(
        status=502, headers={"content-type": "text/html"}, body=b"<html>502 Bad Gateway</html>"
    )
    proxy, _ctx, _request = _record(upstream)
    # record() completed without raising; the event is stored.
    assert len(proxy._store.load_episode("ep").events) == 1  # noqa: SLF001


def test_sse_replay_reuses_recorded_chunk_framing() -> None:
    chunks = ('data: {"choices":[{"delta":{"content":"a"}}]}\n\n', "data: [DONE]\n\n")
    raw = "".join(chunks)
    upstream = HTTPResponse(
        status=200,
        headers={"content-type": "text/event-stream"},
        body=raw.encode("utf-8"),
        stream=CapturedStream(chunks),
    )
    proxy, ctx, request = _record(upstream)
    replayed = asyncio.run(proxy.replay(request, ctx))
    assert replayed.stream is not None
    assert replayed.stream.raw == raw
    assert replayed.stream.chunks == chunks  # exact framing preserved, not re-derived


def test_counterfactual_replay_halts_on_divergence() -> None:
    proxy, ctx, _ = _record(_json_response(_RESP))
    diverging = _request(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "DIFFERENT"}]}
    )
    with pytest.raises(ProxyDivergence):
        asyncio.run(proxy.replay(diverging, ctx, mode=DivergencePolicy.HALT))
