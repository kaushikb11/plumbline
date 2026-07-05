"""Proxy hardening: zero-touch on episode-open failure, and the redaction hook.

Two production blockers (engineering spec §4.2 zero-touch / invariant 4, and §5.1
trace-body security):

  * A store/open fault (read-only fs, full disk) must be logged and DROPPED — the
    runtime always receives the upstream response it earned, even on the FIRST call
    where the episode is opened. Recording is best-effort; forwarding is not.
  * An optional redactor scrubs sensitive keys from what is STORED, while the
    response returned to the runtime stays byte-identical (zero-touch).
"""

import asyncio
import functools
import json
from collections.abc import Mapping

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.core.trace import EpisodeManifest, Payload
from plumbline.proxy import (
    AsyncHTTPProxy,
    HTTPRequest,
    HTTPResponse,
    RecordingProxy,
)
from plumbline.proxy.recording import redact_json_keys

_UPSTREAM = Payload(inline={"caption": "a red ball", "secret": "sk-live-DEADBEEF"})


class _OpenFailsStore(TraceStore):
    """A store whose episode-open fails (mimics a read-only / full disk). Everything
    else behaves; only open_episode raises, so the fault surfaces where recording
    begins — the exact spot the zero-touch hole was."""

    def open_episode(self, manifest: EpisodeManifest) -> None:
        raise OSError("read-only filesystem")


def _ctx() -> Context:
    return Context(episode_id="ep", model_id=None, params={})


# --- Blocker 2: zero-touch survives an episode-open failure ------------------


def test_recording_proxy_returns_upstream_when_episode_open_fails() -> None:
    # open_episode raises on the FIRST forward. Because _ensure_open now lives inside
    # the zero-touch guard, the fault is logged and dropped and the runtime still gets
    # the upstream response — a healthy model call must not throw because recording
    # could not open the episode.
    recorder = Recorder(_OpenFailsStore(), VirtualClock())
    proxy = RecordingProxy(upstream=lambda request: _UPSTREAM, recorder=recorder)

    returned = proxy.forward(Payload(inline={"kind": "caption"}), _ctx())

    assert returned is _UPSTREAM  # unaltered, despite the open failure


def test_async_http_proxy_returns_upstream_when_episode_open_fails() -> None:
    upstream_response = HTTPResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8"),
    )
    proxy = AsyncHTTPProxy(
        transport=_FixedTransport(upstream_response),
        recorder=Recorder(_OpenFailsStore(), VirtualClock()),
        store=_OpenFailsStore(),
    )
    returned = asyncio.run(proxy.record(_http_request({"model": "m", "messages": []}), _ctx()))
    assert returned is upstream_response  # forwarded unaltered despite the open failure


# --- Blocker 3: redaction scrubs the stored trace, never the runtime response --


def test_recording_proxy_redacts_stored_event_but_not_returned_response() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    redactor = functools.partial(redact_json_keys, keys={"secret"})
    proxy = RecordingProxy(upstream=lambda request: _UPSTREAM, recorder=recorder, redactor=redactor)

    returned = proxy.forward(Payload(inline={"kind": "caption"}), _ctx())
    proxy.close("ep")

    # Zero-touch: the runtime receives the untouched upstream response.
    assert returned is _UPSTREAM
    assert isinstance(returned.inline, dict)
    assert returned.inline["secret"] == "sk-live-DEADBEEF"

    # The STORED response has the secret blanked.
    stored = store.load_episode("ep").events[0]
    assert isinstance(stored.response.inline, dict)
    assert stored.response.inline["secret"] == "[REDACTED]"
    assert stored.response.inline["caption"] == "a red ball"  # non-secret preserved


def test_async_http_proxy_redacts_stored_event_but_not_returned_response() -> None:
    store = TraceStore()
    secret_body = {"choices": [{"message": {"content": "hi"}}], "api_key": "sk-live-XYZ"}
    upstream_response = HTTPResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps(secret_body).encode("utf-8"),
    )
    proxy = AsyncHTTPProxy(
        transport=_FixedTransport(upstream_response),
        recorder=Recorder(store, VirtualClock()),
        store=store,
        redactor=functools.partial(redact_json_keys, keys={"api_key"}),
    )
    returned = asyncio.run(proxy.record(_http_request({"model": "m", "messages": []}), _ctx()))
    proxy.close("ep")

    # Runtime response is byte-identical to the upstream.
    assert returned is upstream_response
    assert json.loads(returned.body)["api_key"] == "sk-live-XYZ"

    stored = store.load_episode("ep").events[0]
    assert isinstance(stored.response.inline, dict)
    assert stored.response.inline["api_key"] == "[REDACTED]"


# --- helpers ----------------------------------------------------------------


class _FixedTransport:
    def __init__(self, response: HTTPResponse) -> None:
        self._response = response

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        return self._response


def _http_request(body: Mapping[str, object]) -> HTTPRequest:
    return HTTPRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=json.dumps(body).encode("utf-8"),
    )
