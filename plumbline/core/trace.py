"""Trace schema — the Layer 1 data contract (engineering spec §3.2 and §5).

FROZEN (CLAUDE.md invariant 1): these dataclasses are the contract between
recorder, replayer, store, and metrics. Do not change a field, type, or name to
make a local problem easier.

Serialization rules (invariant 3, §5.1): JSON for metadata, safetensors for
tensors, *never* pickle. Large binary payloads are content-addressed and
referenced by hash, never inlined (§5.3).

These types are interface sketches per the spec ("types are illustrative",
§3); a handful of shapes the spec describes in prose but does not fully type
are flagged inline with NOTE and should be treated as the first place to revisit
if the contract needs refinement.
"""

import enum
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from plumbline.core.seam import Seam

# A JSON value: the canonical inline-content type for small structured payloads
# and metadata (§5.1, §5.2). PEP 695 recursive type alias (Python 3.12+).
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]


# --- Blobs (content-addressed large binary content, §5.3) -------------------


class BlobKind(enum.Enum):
    SAFETENSORS = "safetensors"  # tensors (image arrays, embeddings)
    BIN = "bin"  # opaque encoded media (image/audio), typed by manifest


@dataclass(frozen=True)
class BlobRef:
    """Reference to a content-addressed blob (`blobs/<sha256>.<kind>`, §5.3)."""

    sha256: str
    kind: BlobKind
    # NOTE: §5.3 says BIN media is "typed by manifest". Carried here for a
    # self-describing reference; whether the manifest is the sole source of
    # truth for the media type is an open refinement.
    media_type: str | None = None


# --- Payload (§3.2) ---------------------------------------------------------


@dataclass(frozen=True)
class Payload:
    """A captured request or response (§3.2).

    Small structured content is inline JSON; large binary content (images,
    audio, tensors) is content-addressed and referenced via `blobs`, never
    inlined (§5.3).

    NOTE: the spec states this separation in prose but does not type Payload's
    internals; this two-field shape (inline + blobs) is the minimal faithful
    encoding of that separation.
    """

    inline: JSONValue
    blobs: tuple[BlobRef, ...] = ()


# --- SeamEvent / Episode / Trace (§3.2) -------------------------------------


@dataclass(frozen=True)
class SeamEvent:
    """One captured interaction at a seam (§3.2).

    Serialized one-per-line into `events.jsonl` (§5.3), with blob content stored
    out-of-line and referenced by hash through its Payloads.
    """

    episode_id: str
    seq: int  # monotonic per-episode ordering index
    seam: Seam
    logical_tick: int  # virtual-clock tick (§3.4)
    wall_ts: float  # original wall-clock time (recorded, never drives replay)
    request: Payload  # canonicalized request (§5.2)
    response: Payload  # canonicalized response
    model_id: str | None  # e.g. "openai/gpt-4o-2024-08-06"
    # NOTE: §3.2 types this `dict`; Mapping is used for immutability under the
    # frozen-data invariant. Holds temperature, top_p, max_tokens, seed if any.
    params: Mapping[str, JSONValue]
    request_digest: str  # content hash of the canonical request (matcher key)
    latency_ms: float


@dataclass(frozen=True)
class Episode:
    """An ordered sequence of events sharing one episode id — one robot run (§3.2)."""

    episode_id: str
    events: tuple[SeamEvent, ...]
    metadata: Mapping[str, JSONValue]


@dataclass(frozen=True)
class Trace:
    """A collection of episodes (§3.2)."""

    episodes: tuple[Episode, ...]


# --- On-disk schema: manifest.json and config/<config_hash>.json (§5.3) -----


@dataclass(frozen=True)
class SeamIndexEntry:
    """One entry of the manifest's seam index into `events.jsonl` (§5.3).

    NOTE: §5.3 names a "seam index" but does not type it; this per-event index
    (seq -> seam/tick/digest) is the minimal useful interpretation.
    """

    seq: int
    seam: Seam
    logical_tick: int
    request_digest: str


@dataclass(frozen=True)
class ConfigSnapshot:
    """`config/<config_hash>.json`: full runtime config + model versions (§5.3)."""

    config_hash: str
    runtime_config: JSONValue
    model_versions: Mapping[str, str]


@dataclass(frozen=True)
class EpisodeManifest:
    """`manifest.json`: episode metadata, config snapshot, seam index (§5.3).

    The config snapshot is referenced by hash (`config_hash` -> ConfigSnapshot
    stored under `config/`), keeping a recurring config stored once.
    """

    episode_id: str
    metadata: Mapping[str, JSONValue]
    config_hash: str
    seam_index: tuple[SeamIndexEntry, ...]


# --- Canonicalization (§5.2, signature only) --------------------------------


@dataclass(frozen=True)
class CanonicalPayload:
    """Result of canonicalizing a payload (§5.2).

    The digest covers only the semantically meaningful canonical content;
    provider-specific noise (request ids, timestamps, server-assigned ids) is
    stripped into the non-digested `meta` block.

    NOTE: §5.2 specifies the *process* (sorted keys, normalized whitespace,
    fixed float formatting, stripped noise) but not a return type; this shape is
    interpreted to carry exactly those pieces.
    """

    canonical_json: bytes  # canonical serialization that the digest covers
    digest: str  # content hash of canonical_json (matcher / faithful-replay key)
    meta: Mapping[str, JSONValue]  # stripped provider noise, NOT digested


def canonical_dumps(value: JSONValue) -> str:
    """Deterministic JSON serialization (§5.2): sorted keys, no whitespace noise.

    Stable across machines for a given interpreter (shortest-round-trip float
    repr), so digests and stored lines are reproducible. Used by the store,
    recorder, and config hashing.

    `allow_nan=False`: NaN/Infinity are not valid JSON, so a non-finite float
    raises here rather than emitting a token (`NaN`) that a strict or
    cross-language parser would reject — keeping the trace portable.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonicalize(payload: Payload) -> CanonicalPayload:
    """Canonicalize a payload per §5.2.

    The digest covers the semantically meaningful content — the inline JSON plus
    the (sorted) content-addressed blob references — not provider noise, which a
    normalizer is expected to have already split out (here `meta` is empty; the
    HTTP proxy normalizers populate it, §5.4/§5.5).
    """
    digested: JSONValue = {
        "inline": payload.inline,
        "blobs": [
            {"sha256": b.sha256, "kind": b.kind.value, "media_type": b.media_type}
            for b in sorted(payload.blobs, key=lambda b: (b.sha256, b.kind.value))
        ],
    }
    canonical = canonical_dumps(digested).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return CanonicalPayload(canonical_json=canonical, digest=digest, meta={})
