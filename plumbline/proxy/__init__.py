"""Recording / replaying HTTP proxy (engineering spec §4.2, §5.4, §5.5).

Public surface: the transport-agnostic record/replay core (`RecordingProxy`,
`ReplayingProxy`), the async HTTP proxy (`AsyncHTTPProxy`) and its transport
types, the provider normalizers, the OTel-GenAI span mapping, and SSE streaming
helpers.
"""

from plumbline.proxy.http import (
    AsyncHTTPProxy,
    AsyncTransport,
    HTTPRequest,
    HTTPResponse,
)
from plumbline.proxy.normalizers import (
    DEFAULT_NORMALIZERS,
    AnthropicMessagesNormalizer,
    GeminiNormalizer,
    NormalizedRequest,
    NormalizedResponse,
    Normalizer,
    OpenAIChatNormalizer,
    contains_image,
    extract_data_urls,
)
from plumbline.proxy.otel import OTelSpan, seam_event_attributes, to_span
from plumbline.proxy.recording import (
    DigestFn,
    ProxyDivergence,
    RecordingProxy,
    ReplayingProxy,
    SeamClassifier,
    UpstreamFn,
    classify_seam,
    default_digest,
)
from plumbline.proxy.streaming import (
    CapturedStream,
    assemble_openai,
    payload_to_stream,
    split_sse,
    stream_to_payload,
)
from plumbline.proxy.tick import BoundaryTickPolicy, TickPolicy
from plumbline.proxy.ws import AsyncWSProxy, WsConnection, WsFrame, WsTransport

__all__ = [
    "DEFAULT_NORMALIZERS",
    "AnthropicMessagesNormalizer",
    "AsyncWSProxy",
    "BoundaryTickPolicy",
    "TickPolicy",
    "WsConnection",
    "WsFrame",
    "WsTransport",
    "AsyncHTTPProxy",
    "AsyncTransport",
    "CapturedStream",
    "DigestFn",
    "GeminiNormalizer",
    "HTTPRequest",
    "HTTPResponse",
    "NormalizedRequest",
    "NormalizedResponse",
    "Normalizer",
    "OTelSpan",
    "OpenAIChatNormalizer",
    "ProxyDivergence",
    "RecordingProxy",
    "ReplayingProxy",
    "SeamClassifier",
    "UpstreamFn",
    "assemble_openai",
    "classify_seam",
    "contains_image",
    "default_digest",
    "extract_data_urls",
    "payload_to_stream",
    "seam_event_attributes",
    "split_sse",
    "stream_to_payload",
    "to_span",
]
