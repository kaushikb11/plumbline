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

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from plumbline.core.interceptor import Context
from plumbline.core.matcher import ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.proxy.normalizers import contains_image

UpstreamFn = Callable[[Payload], Payload]
SeamClassifier = Callable[[Payload, Context], Seam]
DigestFn = Callable[[Payload], str]


def default_digest(payload: Payload) -> str:
    """Stable matcher key for a payload (§5.2)."""
    return canonicalize(payload).digest


def classify_seam(request: Payload, ctx: Context) -> Seam:
    """Infer the seam from payload content (§4.2): a vision/image request is a
    sensor->caption call; a plain chat/decision request is fuse->decide."""
    if contains_image(request.inline):
        return Seam.SENSOR_TO_CAPTION
    return Seam.FUSE_TO_DECIDE


class ProxyDivergence(Exception):
    """Raised on a halted counterfactual replay (§6.4): the live request no
    longer matches the recording, so no recorded response is served."""

    def __init__(self, seam: Seam, distance: float, request_digest: str) -> None:
        super().__init__(f"counterfactual divergence at {seam.value} (distance {distance:.4f})")
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
    _opened: set[str] = field(default_factory=set, init=False, repr=False)
    _seq: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def forward(self, request: Payload, ctx: Context) -> Payload:
        self._ensure_open(ctx.episode_id)
        started = time.perf_counter()
        response = self.upstream(request)  # zero-touch: forward and return unaltered
        latency_ms = (time.perf_counter() - started) * 1000.0

        seq = self._seq[ctx.episode_id]
        self._seq[ctx.episode_id] = seq + 1
        event = SeamEvent(
            episode_id=ctx.episode_id,
            seq=seq,  # monotonic call order
            seam=self.classifier(request, ctx),
            logical_tick=ctx.logical_tick,  # loop-iteration index from the loop driver (§6)
            wall_ts=time.time(),
            request=request,
            response=response,
            model_id=ctx.model_id,
            params=ctx.params,
            request_digest=self.digest(request),
            latency_ms=latency_ms,
        )
        self.recorder.record(event)
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
        occurrence beyond what was recorded is a divergence, surfaced as a KeyError."""
        digest = self.digest(request)
        events = self._by_digest.get(digest, [])
        index = self._digest_cursor.get(digest, 0)
        if index >= len(events):
            raise KeyError(f"no recorded response for request {digest}")
        self._digest_cursor[digest] = index + 1
        return events[index].response

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
