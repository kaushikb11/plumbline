"""Async HTTP recording/replaying proxy (engineering spec §4.2).

The runtime is pointed at this proxy by setting the provider base URL; the proxy
sits between the runtime and the real model endpoints. It is transport-agnostic:
the actual network client is injected as an `AsyncTransport`, so the substrate
stays light and free of a hard HTTP-client dependency (a small httpx/asyncio
reverse proxy or mitmproxy-as-a-library satisfies the Protocol in deployment).

Record mode: forward the request to the real endpoint via the transport, capture
and normalize request/response (including SSE streams, §14.3), record a
SeamEvent, and return the upstream HTTP response *unaltered* (the zero-touch
invariant). Replay mode: do not forward; match the request to the trace and
reconstruct the recorded HTTP response, preserving SSE chunk framing.
"""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from plumbline.core.interceptor import Context
from plumbline.core.matcher import ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import BlobKind, JSONValue, Payload, SeamEvent
from plumbline.proxy.normalizers import DEFAULT_NORMALIZERS, Normalizer
from plumbline.proxy.recording import ProxyDivergence
from plumbline.proxy.streaming import (
    CapturedStream,
    assemble_openai,
    payload_to_stream,
    stream_to_payload,
)

_SSE_CONTENT_TYPE = "text/event-stream"
_JSON_CONTENT_TYPE = "application/json"
# The upstream HTTP status, stashed in the event's params so a non-200 (rate limit,
# error) is captured and reconstructed on replay. Namespaced to avoid colliding with
# model request params (temperature, etc.); SeamEvent is frozen, so no new field.
_HTTP_STATUS_KEY = "plumbline.http_status"


@dataclass(frozen=True)
class HTTPRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    stream: CapturedStream | None = None  # set when the response is an SSE stream


class AsyncTransport(Protocol):
    """The real network client the proxy forwards through (record mode)."""

    async def send(self, request: HTTPRequest) -> HTTPResponse: ...


class AsyncHTTPProxy:
    def __init__(
        self,
        *,
        transport: AsyncTransport,
        recorder: Recorder,
        store: TraceStore,
        normalizers: tuple[Normalizer, ...] = DEFAULT_NORMALIZERS,
        episode_metadata: Mapping[str, JSONValue] | None = None,
    ) -> None:
        self._transport = transport
        self._recorder = recorder
        self._store = store
        self._normalizers = normalizers
        self._episode_metadata: Mapping[str, JSONValue] = episode_metadata or {}
        self._opened: set[str] = set()
        self._seq: dict[str, int] = {}

    async def record(self, request: HTTPRequest, ctx: Context) -> HTTPResponse:
        self._ensure_open(ctx.episode_id)
        normalizer = self._select(request.url)
        normalized_request = normalizer.normalize_request(_decode_json(request.body))

        started = time.perf_counter()
        response = await self._transport.send(request)  # zero-touch forward
        latency_ms = (time.perf_counter() - started) * 1000.0

        if response.stream is not None:
            response_payload = stream_to_payload(response.stream, assemble_openai(response.stream))
            response_blobs: Mapping[str, bytes] = {}
        else:
            normalized_response = normalizer.normalize_response(_decode_json(response.body))
            response_payload = normalized_response.payload
            response_blobs = normalized_response.blobs

        for data in {**normalized_request.blobs, **response_blobs}.values():
            self._store.put_blob(data, BlobKind.BIN)  # content-addressed (§5.3)

        params = dict(normalized_request.params or ctx.params)
        params[_HTTP_STATUS_KEY] = response.status

        seq = self._seq[ctx.episode_id]
        self._seq[ctx.episode_id] = seq + 1
        self._recorder.record(
            SeamEvent(
                episode_id=ctx.episode_id,
                seq=seq,  # monotonic call order
                seam=normalized_request.seam,
                logical_tick=ctx.logical_tick,  # loop-iteration index from the loop driver (§6)
                wall_ts=time.time(),
                request=normalized_request.payload,
                response=response_payload,
                model_id=normalized_request.model_id or ctx.model_id,
                params=params,
                request_digest=normalized_request.digest_key,
                latency_ms=latency_ms,
            )
        )
        return response  # the runtime receives the upstream response unaltered

    async def replay(
        self,
        request: HTTPRequest,
        ctx: Context,
        *,
        mode: DivergencePolicy | None = None,
        matchers: Mapping[Seam, Matcher] | None = None,
    ) -> HTTPResponse:
        """Serve a recorded response without forwarding.

        With `mode` None this is faithful replay (match by request_digest). With a
        DivergencePolicy and matchers it is counterfactual: a mismatch raises
        ProxyDivergence under HALT (no stale response served, §6.4).
        """
        normalizer = self._select(request.url)
        normalized = normalizer.normalize_request(_decode_json(request.body))
        episode = self._store.load_episode(ctx.episode_id)

        if mode is None:
            for event in episode.events:
                if event.request_digest == normalized.digest_key:
                    return _response_from_payload(event.response, _status_of(event))
            raise KeyError(f"no recorded response for request {normalized.digest_key}")

        # NOTE (bounded): this counterfactual path matches against the FIRST recorded
        # event at the seam — it has no positional cursor, so for a seam that recurs
        # across ticks use `Replayer.counterfactual` / `ReplayingProxy` (which advance
        # a per-seam cursor and can go live). The deployed replay server uses only the
        # faithful path above; this branch is for in-process/counterfactual callers.
        matchers = matchers or {}
        for event in episode.events:
            if event.seam is not normalized.seam:
                continue
            # Default to ExactMatcher for an unconfigured seam (matching
            # ReplayingProxy.counterfactual) so it still halts, not silently skips.
            matcher = matchers.get(normalized.seam, ExactMatcher())
            verdict = matcher.matches(normalized.payload, event.request)
            if verdict.is_match:
                return _response_from_payload(event.response, _status_of(event))
            if mode is DivergencePolicy.HALT:
                raise ProxyDivergence(normalized.seam, verdict.distance, event.request_digest)
        raise KeyError(f"no matching recorded call at {normalized.seam.value}")

    def close(self, episode_id: str) -> None:
        if episode_id in self._opened:
            self._recorder.close_episode(episode_id)
            self._opened.discard(episode_id)

    def _ensure_open(self, episode_id: str) -> None:
        if episode_id not in self._opened:
            self._recorder.open_episode(episode_id, self._episode_metadata)
            self._opened.add(episode_id)
            self._seq[episode_id] = 0

    def _select(self, url: str) -> Normalizer:
        for normalizer in self._normalizers:
            if normalizer.handles(url):
                return normalizer
        return self._normalizers[0]  # default to the first (OpenAI-compatible)


def _decode_json(body: bytes) -> JSONValue:
    """Best-effort JSON decode. A non-JSON body (an HTML/text error page, binary)
    is preserved as raw text rather than crashing record() after the upstream
    response has already been consumed.

    Fidelity note: a non-JSON body is NOT byte-faithful on replay — it is wrapped as
    `{"plumbline.raw_body": <text>}` and re-emitted as JSON. The determinism envelope
    is model-I/O JSON (invariant 4); a runtime that branches on a recorded error
    page's raw bytes would see different bytes on replay."""
    if not body:
        return {}
    try:
        parsed: JSONValue = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"plumbline.raw_body": body.decode("utf-8", "replace")}
    return parsed


def _status_of(event: SeamEvent) -> int:
    status = event.params.get(_HTTP_STATUS_KEY, 200)
    return status if isinstance(status, int) and not isinstance(status, bool) else 200


def _response_from_payload(payload: Payload, status: int = 200) -> HTTPResponse:
    stream = payload_to_stream(payload)
    if stream is not None:
        return HTTPResponse(
            status=status,
            headers={"content-type": _SSE_CONTENT_TYPE},
            body=stream.raw.encode("utf-8"),
            stream=stream,  # reuse the recorded chunk framing; do not re-derive it
        )
    body = json.dumps(payload.inline, separators=(",", ":")).encode("utf-8")
    return HTTPResponse(status=status, headers={"content-type": _JSON_CONTENT_TYPE}, body=body)
