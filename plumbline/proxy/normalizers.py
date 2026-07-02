"""Provider normalizers (engineering spec §5.5).

Each normalizer maps a provider's wire format to a canonical request/response
Payload and tags the seam. This is the only provider-specific code in the
substrate. Normalizers exist for the OpenAI chat/vision schema (which xAI,
DeepSeek, and Ollama largely mirror), the Gemini schema, and the Anthropic
Messages schema.

Large binary content (base64 image data URLs) is extracted into content-
addressed blobs and replaced inline with a `blob:<sha256>` marker, never inlined
(§5.3). The returned `blobs` map (sha256 -> bytes) is handed to the store by the
HTTP proxy. The digest key is the canonical serialization of the blob-extracted
request, so it is stable across runs and across the image bytes' representation.

NOTE: bare-base64 image fields that are not data URLs (Anthropic
`source.data`, Gemini `inlineData.data`) are detected for seam classification but
left inline for now; data-URL extraction is the common path (OpenAI vision) and
field-level extraction for the other two is a flagged refinement.
"""

import base64
import binascii
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from plumbline.core.seam import Seam
from plumbline.core.trace import BlobKind, BlobRef, JSONValue, Payload, canonicalize


@dataclass(frozen=True)
class NormalizedRequest:
    seam: Seam
    payload: Payload
    blobs: Mapping[str, bytes]
    digest_key: str
    model_id: str | None
    params: Mapping[str, JSONValue]


@dataclass(frozen=True)
class NormalizedResponse:
    payload: Payload
    blobs: Mapping[str, bytes]


class Normalizer(Protocol):
    @property
    def system(self) -> str:  # gen_ai.system value, e.g. "openai"
        ...

    def handles(self, url: str) -> bool: ...

    def normalize_request(self, body: JSONValue) -> NormalizedRequest: ...

    def normalize_response(self, body: JSONValue) -> NormalizedResponse: ...


# --- Shared helpers ---------------------------------------------------------


def contains_image(value: JSONValue) -> bool:
    """True if the structure carries image content (any provider's encoding)."""
    if isinstance(value, str):
        return value.startswith("data:image")
    if isinstance(value, list):
        return any(contains_image(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") in ("image_url", "image"):
            return True
        if "inlineData" in value or "inline_data" in value:
            return True
        return any(contains_image(item) for item in value.values())
    return False


def extract_data_urls(value: JSONValue) -> tuple[JSONValue, dict[str, bytes]]:
    """Replace `data:<mime>;base64,...` strings with `blob:<sha256>` markers,
    returning the rewritten structure and a {sha256: bytes} map for the store."""
    blobs: dict[str, bytes] = {}

    def walk(node: JSONValue) -> JSONValue:
        if isinstance(node, str) and node.startswith("data:") and ";base64," in node:
            header, b64 = node.split(";base64,", 1)
            try:
                raw = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError):
                # Malformed base64: keep the string inline rather than crashing
                # record() on one bad/adversarial image field.
                return node
            sha = hashlib.sha256(raw).hexdigest()
            blobs[sha] = raw
            return f"blob:{sha}"
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            return {key: walk(item) for key, item in node.items()}
        return node

    return walk(value), blobs


def _blob_refs(blobs: Mapping[str, bytes]) -> tuple[BlobRef, ...]:
    return tuple(BlobRef(sha256=sha, kind=BlobKind.BIN) for sha in sorted(blobs))


def _get(body: JSONValue, key: str) -> JSONValue:
    return body[key] if isinstance(body, dict) and key in body else None


def _get_str(body: JSONValue, key: str) -> str | None:
    value = _get(body, key)
    return value if isinstance(value, str) else None


def _collect(body: JSONValue, keys: tuple[str, ...]) -> dict[str, JSONValue]:
    out: dict[str, JSONValue] = {}
    if isinstance(body, dict):
        for key in keys:
            if key in body:
                out[key] = body[key]
    return out


def _request(
    body: JSONValue, *, model_id: str | None, params: Mapping[str, JSONValue]
) -> NormalizedRequest:
    inline, blobs = extract_data_urls(body)
    payload = Payload(inline=inline, blobs=_blob_refs(blobs))
    return NormalizedRequest(
        seam=Seam.SENSOR_TO_CAPTION if contains_image(body) else Seam.FUSE_TO_DECIDE,
        payload=payload,
        blobs=blobs,
        # The single request-identity convention across all record paths: the
        # canonical content hash (§5.2), covering inline JSON + blob references.
        digest_key=canonicalize(payload).digest,
        model_id=model_id,
        params=params,
    )


def _response(body: JSONValue) -> NormalizedResponse:
    inline, blobs = extract_data_urls(body)
    return NormalizedResponse(payload=Payload(inline=inline, blobs=_blob_refs(blobs)), blobs=blobs)


# --- OpenAI chat/vision (also xAI, DeepSeek, Ollama OpenAI-compatible) -------


@dataclass(frozen=True)
class OpenAIChatNormalizer:
    system: str = "openai"

    def handles(self, url: str) -> bool:
        lowered = url.lower()
        return "chat/completions" in lowered or lowered.endswith("/completions")

    def normalize_request(self, body: JSONValue) -> NormalizedRequest:
        model = _get_str(body, "model")
        return _request(
            body,
            model_id=f"{self.system}/{model}" if model else None,
            params=_collect(body, ("temperature", "top_p", "max_tokens", "seed", "n", "stop")),
        )

    def normalize_response(self, body: JSONValue) -> NormalizedResponse:
        return _response(body)


# --- Gemini (generateContent) -----------------------------------------------


@dataclass(frozen=True)
class GeminiNormalizer:
    system: str = "gemini"

    def handles(self, url: str) -> bool:
        lowered = url.lower()
        return "generatecontent" in lowered or "generativelanguage" in lowered

    def normalize_request(self, body: JSONValue) -> NormalizedRequest:
        model = _get_str(body, "model")
        generation_config = _get(body, "generationConfig")
        params = _collect(generation_config, ("temperature", "topP", "topK", "maxOutputTokens"))
        return _request(body, model_id=f"{self.system}/{model}" if model else None, params=params)

    def normalize_response(self, body: JSONValue) -> NormalizedResponse:
        return _response(body)


# --- Anthropic Messages -----------------------------------------------------


@dataclass(frozen=True)
class AnthropicMessagesNormalizer:
    system: str = "anthropic"

    def handles(self, url: str) -> bool:
        lowered = url.lower()
        return lowered.endswith("/messages") or "anthropic" in lowered

    def normalize_request(self, body: JSONValue) -> NormalizedRequest:
        model = _get_str(body, "model")
        return _request(
            body,
            model_id=f"{self.system}/{model}" if model else None,
            params=_collect(
                body, ("temperature", "top_p", "top_k", "max_tokens", "stop_sequences")
            ),
        )

    def normalize_response(self, body: JSONValue) -> NormalizedResponse:
        return _response(body)


DEFAULT_NORMALIZERS: tuple[Normalizer, ...] = (
    OpenAIChatNormalizer(),
    GeminiNormalizer(),
    AnthropicMessagesNormalizer(),
)
