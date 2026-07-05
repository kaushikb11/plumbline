"""Plumbline — make a language-bus robot runtime reproducible, regression-testable,
and fidelity-measurable by record-replaying the model calls at the four seams of the
perception-to-action loop.

The essentials are re-exported here for a flat, discoverable import surface::

    from plumbline import Seam, SeamEvent, make_seam_event, Recorder, Replayer, TraceStore

Layer-specific APIs live under their submodules: `plumbline.proxy` (recording/replaying
proxy), `plumbline.fidelity` (the §7 metrics), `plumbline.regression` (the CI gate),
`plumbline.adapters` (runtime adapters). The full frozen substrate is `plumbline.core`.
"""

from importlib.metadata import PackageNotFoundError, version

from plumbline.core import (
    Context,
    DigestMismatch,
    DivergencePolicy,
    EmbeddingMatcher,
    EpisodeNotFound,
    EpisodeNotOpen,
    ExactMatcher,
    Matcher,
    NumericToleranceMatcher,
    Payload,
    Recorder,
    Replayer,
    Seam,
    SeamEvent,
    TraceStore,
    VirtualClock,
    canonicalize,
    make_seam_event,
)
from plumbline.session import RecordingSession

try:
    __version__ = version("plumbline")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    "Context",
    "DigestMismatch",
    "DivergencePolicy",
    "EmbeddingMatcher",
    "EpisodeNotFound",
    "EpisodeNotOpen",
    "ExactMatcher",
    "Matcher",
    "NumericToleranceMatcher",
    "Payload",
    "Recorder",
    "RecordingSession",
    "Replayer",
    "Seam",
    "SeamEvent",
    "TraceStore",
    "VirtualClock",
    "__version__",
    "canonicalize",
    "make_seam_event",
]
