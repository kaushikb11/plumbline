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
import logging
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
from plumbline.core.trace import BlobKind, JSONValue, Payload, SeamEvent, canonicalize
from plumbline.proxy.normalizers import DEFAULT_NORMALIZERS, NormalizedRequest, Normalizer
from plumbline.proxy.recording import ProxyDivergence, Redactor, ReplayingProxy
from plumbline.proxy.streaming import (
    CapturedStream,
    assemble_openai,
    payload_to_stream,
    stream_to_payload,
)
from plumbline.proxy.tick import _TICK_OVERRIDE_KEY, TickPolicy

_log = logging.getLogger("plumbline.proxy")
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
        tick_policy: TickPolicy | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._transport = transport
        self._recorder = recorder
        self._store = store
        self._normalizers = normalizers
        self._episode_metadata: Mapping[str, JSONValue] = episode_metadata or {}
        self._opened: set[str] = set()
        self._seq: dict[str, int] = {}
        self._tick_policy = tick_policy
        self._redactor = redactor
        # Faithful replay delegates to a per-episode ReplayingProxy, built ONCE and
        # cached (its __post_init__ indexes the episode by request_digest with a
        # per-digest cursor). This replaces the old per-request store.load_episode()
        # scan, which reparsed the whole trace on every call (O(n^2)) and — worse —
        # returned the FIRST event for a repeated request every time (a false-green:
        # distinct sampled responses collapsed to one). See §4.2.
        self._replayers: dict[str, ReplayingProxy] = {}

    async def record(self, request: HTTPRequest, ctx: Context) -> HTTPResponse:
        started = time.perf_counter()
        response = await self._transport.send(request)  # zero-touch forward
        latency_ms = (time.perf_counter() - started) * 1000.0

        # ZERO-TOUCH (CLAUDE.md invariant 4): the runtime must receive the upstream
        # response it earned even if RECORDING fails (disk full, blob write error, a
        # read-only episode open, a malformed body). EVERY recording step —
        # opening the episode, normalizing, and _capture — lives inside this guard so
        # a fault is logged and dropped, never allowed to turn a good 200 into a 500
        # for the runtime. _ensure_open is here (not before the forward) so a failed
        # episode-open does not throw away an already-earned upstream response.
        try:
            self._ensure_open(ctx.episode_id)
            normalizer = self._select(request.url)
            normalized_request = normalizer.normalize_request(_decode_json(request.body))
            self._capture(request, response, normalizer, normalized_request, ctx, latency_ms)
        except Exception:  # noqa: BLE001 - recording must never break the forward path
            _log.exception("plumbline: recording failed for %s (response forwarded)", request.url)
        return response  # the runtime receives the upstream response unaltered

    def _capture(
        self,
        request: "HTTPRequest",
        response: "HTTPResponse",
        normalizer: "Normalizer",
        normalized_request: "NormalizedRequest",
        ctx: Context,
        latency_ms: float,
    ) -> None:
        if response.stream is not None:
            response_payload = stream_to_payload(response.stream, assemble_openai(response.stream))
            response_blobs: Mapping[str, bytes] = {}
        else:
            normalized_response = normalizer.normalize_response(_decode_json(response.body))
            response_payload = normalized_response.payload
            response_blobs = normalized_response.blobs

        for data in {**normalized_request.blobs, **response_blobs}.values():
            self._store.put_blob(data, BlobKind.BIN)  # content-addressed (§5.3)

        # An explicit tick override (from the x-plumbline-tick header) wins; else the
        # tick policy (if set) derives the tick from the seam sequence; else 0.
        override = ctx.params.get(_TICK_OVERRIDE_KEY)
        override_int = (
            override if isinstance(override, int) and not isinstance(override, bool) else None
        )
        if self._tick_policy is not None:
            logical_tick = self._tick_policy.next_tick(normalized_request.seam, override_int)
        else:
            logical_tick = override_int if override_int is not None else ctx.logical_tick

        # Merge (not either/or): a caller's ctx.params are preserved and the
        # normalizer's extracted params (temperature, etc.) take precedence. The
        # control key is stripped so it never lands in the trace.
        params: dict[str, JSONValue] = {
            k: v for k, v in ctx.params.items() if k != _TICK_OVERRIDE_KEY
        }
        params.update(normalized_request.params)
        params[_HTTP_STATUS_KEY] = response.status

        # Redaction (§5.1) runs INSIDE the zero-touch guard (_capture is called from
        # the guarded block in record()), so a redactor bug cannot break forwarding.
        # It rewrites only what is STORED; the runtime already holds the untouched
        # upstream response. The digest is recomputed from the redacted request so it
        # still equals canonicalize(stored request).digest (the recorder validates).
        request_payload = normalized_request.payload
        request_digest = normalized_request.digest_key
        if self._redactor is not None:
            request_payload = self._redactor(request_payload)
            response_payload = self._redactor(response_payload)
            request_digest = canonicalize(request_payload).digest

        seq = self._seq[ctx.episode_id]
        self._seq[ctx.episode_id] = seq + 1
        self._recorder.record(
            SeamEvent(
                episode_id=ctx.episode_id,
                seq=seq,  # monotonic call order
                seam=normalized_request.seam,
                logical_tick=logical_tick,  # from the tick policy / header override (§6)
                wall_ts=time.time(),
                request=request_payload,
                response=response_payload,
                model_id=normalized_request.model_id or ctx.model_id,
                params=params,
                request_digest=request_digest,
                latency_ms=latency_ms,
            )
        )

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

        if mode is None:
            # Delegate to the tested ReplayingProxy (built once, cached): it advances
            # a per-digest cursor, so a REPEATED request is served in record order
            # (not first-wins), an over-consumed request raises ReplayMiss (a KeyError
            # subclass -> the server's existing 404), and there is no per-request
            # reparse of the trace. normalized.payload canonicalizes to
            # normalized.digest_key, which is exactly the stored request_digest, so
            # the delegate's default_digest keying matches. See §4.2.
            replayer = self._replayer_for(ctx.episode_id)
            event = replayer.faithful_event(normalized.payload, ctx)
            return _response_from_payload(event.response, _status_of(event))

        # NOTE (bounded): this counterfactual path matches against the FIRST recorded
        # event at the seam — it has no positional cursor, so for a seam that recurs
        # across ticks use `Replayer.counterfactual` / `ReplayingProxy` (which advance
        # a per-seam cursor and can go live). The deployed replay server uses only the
        # faithful path above; this branch is for in-process/counterfactual callers.
        episode = self._store.load_episode(ctx.episode_id)
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

    def _replayer_for(self, episode_id: str) -> ReplayingProxy:
        """The per-episode faithful replayer, built once and cached. Its __post_init__
        indexes the episode by request_digest (a single reparse), so serving is O(1)
        per request with a per-digest cursor for record-order repeats (§4.2)."""
        replayer = self._replayers.get(episode_id)
        if replayer is None:
            replayer = ReplayingProxy(store=self._store, episode_id=episode_id)
            self._replayers[episode_id] = replayer
        return replayer

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
