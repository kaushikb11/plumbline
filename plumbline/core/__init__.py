"""Frozen substrate (engineering spec §3): the record/replay contract.

Curated re-exports of the interfaces workstreams build against — the seams, the
trace types, the recorder/replayer/store, the clock, and the matchers. Importing
from here (or from the top-level `plumbline`) is the supported surface; the
submodules remain importable but this is the stable, discoverable API.
"""

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context, Interceptor
from plumbline.core.matcher import (
    EmbeddingMatcher,
    ExactMatcher,
    Matcher,
    MatchVerdict,
    NumericToleranceMatcher,
    active_embedder,
    active_embedder_name,
    set_embedder,
    using_embedder,
)
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy, Replayer, ReplayResult
from plumbline.core.seam import Seam
from plumbline.core.store import (
    EpisodeExists,
    EpisodeNotFound,
    EpisodeNotOpen,
    TraceStore,
    UnsafeTraceRef,
)
from plumbline.core.trace import (
    BlobKind,
    BlobRef,
    CanonicalPayload,
    DigestMismatch,
    Episode,
    EpisodeManifest,
    JSONValue,
    Payload,
    SeamEvent,
    Trace,
    canonical_dumps,
    canonicalize,
    make_seam_event,
)

__all__ = [
    "BlobKind",
    "BlobRef",
    "CanonicalPayload",
    "Context",
    "DigestMismatch",
    "DivergencePolicy",
    "EmbeddingMatcher",
    "Episode",
    "EpisodeExists",
    "EpisodeManifest",
    "EpisodeNotFound",
    "EpisodeNotOpen",
    "UnsafeTraceRef",
    "ExactMatcher",
    "Interceptor",
    "JSONValue",
    "MatchVerdict",
    "Matcher",
    "NumericToleranceMatcher",
    "Payload",
    "Recorder",
    "Replayer",
    "ReplayResult",
    "Seam",
    "SeamEvent",
    "Trace",
    "TraceStore",
    "VirtualClock",
    "active_embedder",
    "active_embedder_name",
    "canonical_dumps",
    "canonicalize",
    "make_seam_event",
    "set_embedder",
    "using_embedder",
]
