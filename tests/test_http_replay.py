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


class _SequenceTransport:
    """Returns a different recorded response on each successive forward, so the SAME
    request can be recorded twice with distinct responses (a static scene sampled at
    temperature > 0)."""

    def __init__(self, responses: list[HTTPResponse]) -> None:
        self._responses = responses
        self._index = 0

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        response = self._responses[self._index]
        self._index += 1
        return response


class _CountingStore(TraceStore):
    """A TraceStore that counts load_episode() calls, to prove faithful replay reparses
    the trace ONCE (delegating to a cached ReplayingProxy), not once per request."""

    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0

    def load_episode(self, episode_id: str):  # type: ignore[no-untyped-def]
        self.load_calls += 1
        return super().load_episode(episode_id)


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


def _record_repeated(
    responses: list[HTTPResponse], store: TraceStore | None = None
) -> tuple[AsyncHTTPProxy, Context, HTTPRequest]:
    """Record the SAME request once per response, in order."""
    store = store or TraceStore()
    proxy = AsyncHTTPProxy(
        transport=_SequenceTransport(list(responses)),
        recorder=Recorder(store, VirtualClock()),
        store=store,
    )
    ctx = Context(episode_id="ep", model_id=None, params={})
    request = _request(_REQ)
    for _ in responses:
        asyncio.run(proxy.record(request, ctx))
    proxy.close("ep")
    return proxy, ctx, request


def test_faithful_replay_serves_repeated_request_in_record_order() -> None:
    # Blocker 1 (false-green): the same request recorded twice with DIFFERENT
    # responses must replay as [LEFT, RIGHT] in record order — not [LEFT, LEFT], the
    # first-wins bug of the old ad-hoc per-request scan.
    proxy, ctx, request = _record_repeated(
        [_json_response({"turn": "LEFT"}), _json_response({"turn": "RIGHT"})]
    )
    first = asyncio.run(proxy.replay(request, ctx))
    second = asyncio.run(proxy.replay(request, ctx))
    assert json.loads(first.body) == {"turn": "LEFT"}
    assert json.loads(second.body) == {"turn": "RIGHT"}


def test_faithful_replay_over_consumption_raises_key_error() -> None:
    # An occurrence beyond what was recorded is a divergence: ReplayMiss (a KeyError
    # subclass -> the server maps it to 404), never a silently reused stale response.
    proxy, ctx, request = _record_repeated([_json_response({"turn": "LEFT"})])
    asyncio.run(proxy.replay(request, ctx))  # consumes the one recorded occurrence
    with pytest.raises(KeyError):
        asyncio.run(proxy.replay(request, ctx))


def test_faithful_replay_reparses_trace_once_not_per_request() -> None:
    # Scale sanity: the deployed replay path must not reparse the whole episode on
    # every request (the old O(n^2) store.load_episode() per call). A cached
    # ReplayingProxy indexes the episode ONCE.
    store = _CountingStore()
    proxy, ctx, request = _record_repeated(
        [_json_response({"turn": str(i)}) for i in range(5)], store=store
    )
    store.load_calls = 0  # ignore any load during recording; count only replay
    for _ in range(5):
        asyncio.run(proxy.replay(request, ctx))
    assert store.load_calls == 1


def test_faithful_replay_missing_request_returns_key_error() -> None:
    # A request that was never recorded is still a miss (KeyError -> 404), unchanged.
    proxy, ctx, _ = _record(_json_response(_RESP))
    never_recorded = _request(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "unseen"}]}
    )
    with pytest.raises(KeyError):
        asyncio.run(proxy.replay(never_recorded, ctx))
