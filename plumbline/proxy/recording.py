"""Recording / replaying proxy core (engineering spec §4.2).

Transport-agnostic: operates on canonical Payloads. The async HTTP layer
(`plumbline.proxy.http`) converts wire bytes <-> Payload via normalizers and
calls into this core; `RecordingProxy` is also the in-process record entry the
zero-touch invariant test exercises directly.

Record mode (`RecordingProxy`): forward the request to the real upstream, capture
request and response, infer the seam, emit a SeamEvent via the Recorder, and
return the real response *unaltered* (the zero-touch invariant, §4.2).

Replay mode (`ReplayingProxy`): do not forward; match the incoming request to the
trace — by `request_digest` for faithful, by `Matcher` for counterfactual — and
serve the recorded response. On a counterfactual miss, apply the DivergencePolicy
(HALT by default: raise, never serve a stale response past the divergence —
invariant 5, §6).
"""

import logging
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from plumbline.core.interceptor import Context
from plumbline.core.matcher import ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.proxy.normalizers import contains_image

_log = logging.getLogger("plumbline.proxy")

UpstreamFn = Callable[[Payload], Payload]
SeamClassifier = Callable[[Payload, Context], Seam]
DigestFn = Callable[[Payload], str]
# A redaction hook applied to request/response Payloads immediately before they are
# handed to the Recorder (§5.1 security note): trace bodies hold prompts, tool
# outputs, and possibly secrets/PII, so an operator may scrub them at capture time.
# It transforms only what is STORED — the runtime always receives the upstream
# response unaltered (zero-touch, invariant 4).
Redactor = Callable[[Payload], Payload]

_REDACTED = "[REDACTED]"


def redact_json_keys(payload: Payload, keys: Collection[str]) -> Payload:
    """Blank the given JSON keys (at any nesting depth) to `"[REDACTED]"` (§5.1).

    A small reusable `Redactor` building block: wrap it (e.g. via `functools.partial`
    or a lambda) to scrub known-sensitive fields — `authorization`, `api_key`, a
    tool result carrying PII — from a request/response Payload before it is recorded.
    Returns a NEW Payload (frozen-data safe); `blobs` are passed through untouched
    (opaque content-addressed media is not key-addressable here)."""
    key_set = frozenset(keys)

    def walk(node: JSONValue) -> JSONValue:
        if isinstance(node, dict):
            return {k: (_REDACTED if k in key_set else walk(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return Payload(inline=walk(payload.inline), blobs=payload.blobs)


def redactor_for(keys: Collection[str]) -> "Redactor":
    """A ready-to-use `Redactor` that blanks `keys` at any depth — pass straight to
    `RecordingProxy(redactor=...)` / `AsyncHTTPProxy(redactor=...)`::

        RecordingProxy(..., redactor=redactor_for({"api_key", "authorization"}))
    """
    key_set = frozenset(keys)
    return lambda payload: redact_json_keys(payload, key_set)


def default_digest(payload: Payload) -> str:
    """Stable matcher key for a payload (§5.2)."""
    return canonicalize(payload).digest


def classify_seam(request: Payload, ctx: Context) -> Seam:
    """Infer the seam from payload content (§4.2): a vision/image request is a
    sensor->caption call; a plain chat/decision request is fuse->decide."""
    if contains_image(request.inline):
        return Seam.SENSOR_TO_CAPTION
    return Seam.FUSE_TO_DECIDE


class ReplayMiss(KeyError):
    """Faithful replay has no (further) recorded response for a request digest (§4.2).

    A `KeyError` subclass so the HTTP layer's `except KeyError -> 404` and existing
    callers keep working, but message-rich: it echoes the offending request digest
    and how many occurrences were recorded versus already served, distinguishing a
    request that was never recorded from one whose recorded occurrences are all
    exhausted (an over-consumption divergence).
    """

    def __init__(self, request_digest: str, recorded: int, consumed: int) -> None:
        self.request_digest = request_digest
        self.recorded = recorded
        self.consumed = consumed
        if recorded == 0:
            detail = f"no recorded response for request {request_digest}"
        else:
            detail = (
                f"request {request_digest} was recorded {recorded} time(s) but has "
                f"already been replayed {consumed} time(s); this extra occurrence has "
                "no recorded response (the runtime issued more calls than recorded)"
            )
        super().__init__(detail)

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class ProxyDivergence(Exception):
    """Raised on a halted counterfactual replay (§6.4): the live request no
    longer matches the recording, so no recorded response is served."""

    def __init__(
        self, seam: Seam, distance: float, request_digest: str, detail: str | None = None
    ) -> None:
        super().__init__(
            detail or f"counterfactual divergence at {seam.value} (distance {distance:.4f})"
        )
        self.seam = seam
        self.distance = distance
        self.request_digest = request_digest


@dataclass
class RecordingProxy:
    """Record-mode proxy: forward, capture, record, return unaltered."""

    upstream: UpstreamFn
    recorder: Recorder
    classifier: SeamClassifier = classify_seam
    digest: DigestFn = default_digest
    episode_metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    # Optional capture-time scrubber (§5.1); None = store bodies verbatim (default,
    # no behavior change). Applied to BOTH request and response before recording.
    redactor: Redactor | None = None
    _opened: set[str] = field(default_factory=set, init=False, repr=False)
    _seq: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def forward(self, request: Payload, ctx: Context) -> Payload:
        started = time.perf_counter()
        response = self.upstream(request)  # zero-touch: forward and return unaltered
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Zero-touch (invariant 4): ALL recording — opening the episode, redaction,
        # and the record() call — happens inside this guard, so a store/open fault
        # (read-only fs, full disk) or a redactor bug is logged and dropped, never
        # allowed to turn a healthy upstream response into an exception for the
        # runtime. _ensure_open lives here (not before the forward) for exactly this:
        # a failed open must not throw a good response away.
        try:
            self._ensure_open(ctx.episode_id)
            seq = self._seq[ctx.episode_id]
            self._seq[ctx.episode_id] = seq + 1
            record_request = self.redactor(request) if self.redactor is not None else request
            record_response = self.redactor(response) if self.redactor is not None else response
            self.recorder.record(
                SeamEvent(
                    episode_id=ctx.episode_id,
                    seq=seq,  # monotonic call order
                    seam=self.classifier(request, ctx),
                    logical_tick=ctx.logical_tick,  # loop-iteration index (§6)
                    wall_ts=time.time(),
                    request=record_request,
                    response=record_response,
                    model_id=ctx.model_id,
                    params=ctx.params,
                    # Digest of the (possibly redacted) stored request, so it matches
                    # what is persisted and passes the recorder's digest validation.
                    request_digest=self.digest(record_request),
                    latency_ms=latency_ms,
                )
            )
        except Exception:  # noqa: BLE001 - recording must never break the forward path
            _log.exception("plumbline: recording failed (response forwarded)")
        return response

    def close(self, episode_id: str) -> None:
        if episode_id in self._opened:
            self.recorder.close_episode(episode_id)
            self._opened.discard(episode_id)

    def _ensure_open(self, episode_id: str) -> None:
        if episode_id not in self._opened:
            self.recorder.open_episode(episode_id, self.episode_metadata)
            self._opened.add(episode_id)
            self._seq[episode_id] = 0


@dataclass
class ReplayingProxy:
    """Replay-mode proxy: serve recorded responses, never forward by default."""

    store: TraceStore
    episode_id: str
    matchers: Mapping[Seam, Matcher] = field(default_factory=dict)
    on_divergence: DivergencePolicy = DivergencePolicy.HALT
    overrides: Mapping[Seam, UpstreamFn] = field(default_factory=dict)
    upstream: UpstreamFn | None = None  # for GO_LIVE / RECORD_NEW
    classifier: SeamClassifier = classify_seam
    digest: DigestFn = default_digest
    # Per-digest LIST (not first-wins): a runtime that re-issues the SAME request
    # (e.g. a static scene at temperature > 0) recorded distinct sampled responses;
    # they must be served in record order, not collapsed to the first — matching the
    # seq-order Replayer path. A digest cursor tracks the next occurrence.
    _by_digest: dict[str, list[SeamEvent]] = field(default_factory=dict, init=False, repr=False)
    _digest_cursor: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _by_seam: dict[Seam, list[SeamEvent]] = field(default_factory=dict, init=False, repr=False)
    _cursor: dict[Seam, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for event in self.store.load_episode(self.episode_id).events:
            self._by_digest.setdefault(event.request_digest, []).append(event)
            self._by_seam.setdefault(event.seam, []).append(event)

    def faithful(self, request: Payload, ctx: Context) -> Payload:
        """Serve the recorded response for the next occurrence of this request digest
        (§4.2). Repeated identical requests are served in record order; an extra
        occurrence beyond what was recorded is a divergence, surfaced as a ReplayMiss
        (a KeyError subclass, so the HTTP layer's 404 mapping is unchanged)."""
        return self.faithful_event(request, ctx).response

    def faithful_event(self, request: Payload, ctx: Context) -> SeamEvent:
        """Like `faithful`, but return the whole SeamEvent for the next occurrence of
        this request digest, not just its response. The HTTP replay path needs the
        event to reconstruct the recorded status/framing (params carry the HTTP
        status), while advancing the SAME per-digest cursor so repeated requests are
        served in record order and over-consumption raises ReplayMiss."""
        digest = self.digest(request)
        events = self._by_digest.get(digest, [])
        index = self._digest_cursor.get(digest, 0)
        if index >= len(events):
            raise ReplayMiss(digest, recorded=len(events), consumed=index)
        self._digest_cursor[digest] = index + 1
        return events[index]

    def counterfactual(self, request: Payload, ctx: Context) -> Payload:
        """Match the live request to the next recorded call at its seam; serve the
        recorded response if it still applies, else apply the DivergencePolicy."""
        seam = self.classifier(request, ctx)
        index = self._cursor.get(seam, 0)
        candidates = self._by_seam.get(seam, [])
        if index >= len(candidates):
            raise KeyError(f"no further recorded calls at {seam.value}")
        recorded = candidates[index]
        self._cursor[seam] = index + 1

        matcher: Matcher = self.matchers.get(seam, ExactMatcher())
        verdict = matcher.matches(request, recorded.request)
        if verdict.is_match:
            return recorded.response

        if self.on_divergence is DivergencePolicy.HALT:
            raise ProxyDivergence(seam, verdict.distance, recorded.request_digest)

        # GO_LIVE / RECORD_NEW: re-execute live where a live function is available.
        live = self.overrides.get(seam, self.upstream)
        if live is None:
            raise ProxyDivergence(seam, verdict.distance, recorded.request_digest)
        return live(request)

    def unconsumed(self) -> tuple[SeamEvent, ...]:
        """Recorded events never served during faithful replay — the UNDER-consumption
        signal. `faithful` flags OVER-consumption loudly (an extra call raises), but a
        runtime that issues FEWER calls than recorded (skips a seam, exits a tick
        early) would otherwise report success while the action sequence silently
        diverged. Call this at end of replay; a non-empty result is a divergence.

        Faithful serving advances a per-digest cursor, so leftovers are the events at
        or past each digest's cursor. (Counterfactual serving uses the per-seam
        cursor; both are reported.)"""
        leftover: list[SeamEvent] = []
        for digest, events in self._by_digest.items():
            leftover.extend(events[self._digest_cursor.get(digest, 0) :])
        return tuple(sorted(leftover, key=lambda e: e.seq))

    def verify_fully_consumed(self) -> None:
        """Raise if any recorded event went unserved (under-consumption divergence)."""
        leftover = self.unconsumed()
        if leftover:
            seams = ", ".join(sorted({e.seam.value for e in leftover}))
            raise ProxyDivergence(
                leftover[0].seam,
                float(len(leftover)),
                leftover[0].request_digest,
                detail=(
                    f"{len(leftover)} recorded event(s) never replayed (seams: {seams}) — "
                    "the runtime issued fewer calls than recorded"
                ),
            )
