"""The trace store — filesystem implementation (engineering spec §5.3).

FROZEN (CLAUDE.md invariant 1): the method signatures are the contract; the
bodies are WS1 implementation. On-disk layout per §5.3:

    <root>/
      episodes/<episode_id>/manifest.json   # metadata, config_hash, seam index
      episodes/<episode_id>/events.jsonl     # one canonical SeamEvent per line
      blobs/<sha256>.safetensors|.bin        # content-addressed large content
      config/<config_hash>.json              # runtime config + model versions

Serialization is JSON for metadata/events and raw bytes (safetensors for
tensors, .bin for opaque media) for blobs. NO pickle anywhere (invariant 3,
§5.1): nothing here imports pickle/dill/torch.save, and every read parses JSON
or returns raw bytes only.

NOTE: §3 references `TraceStore` but never types its methods; this interface is
derived from the §5.3 layout. The no-arg-capable `__init__(root=...)` is added so
the store can be constructed standalone (the frozen interface declared no
constructor); it does not change any declared method signature.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plumbline.core.seam import Seam
from plumbline.core.trace import (
    BlobKind,
    BlobRef,
    ConfigSnapshot,
    Episode,
    EpisodeManifest,
    JSONValue,
    Payload,
    SeamEvent,
    SeamIndexEntry,
    canonical_dumps,
)

_LOG = logging.getLogger("plumbline.store")

# A hex SHA-256 digest: the ONLY shape a content-addressed reference (blob sha256,
# config_hash) may take. Anything else in a filesystem join is an untrusted-input
# path-traversal attempt (a hostile events.jsonl setting "sha256": "../../etc/..").
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EpisodeNotOpen(KeyError):
    """An episode was written to (or closed) before `open_episode()` registered it.

    A `KeyError` subclass so existing callers that catch `KeyError` keep working,
    but message-rich instead of the cryptic bare `KeyError: '<id>'`.
    """

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        super().__init__(f"episode {episode_id!r} not open; call open_episode() first")

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class EpisodeNotFound(FileNotFoundError):
    """No episode with this id exists in the trace store.

    A `FileNotFoundError` subclass so callers that catch the OS error keep working,
    but it names both the episode id and the store root instead of surfacing a raw
    path from deep inside the read.
    """

    def __init__(self, episode_id: str, root: Path) -> None:
        self.episode_id = episode_id
        self.store_root = root
        super().__init__(f"episode {episode_id!r} not found in trace store at {root}")


class EpisodeExists(FileExistsError):
    """`open_episode()` was called for an episode id that already has recorded events.

    A `FileExistsError` subclass so callers that catch the OS error keep working.
    Opening unconditionally used to `write_text("")` events.jsonl, so restarting a
    crashed recorder with the SAME episode id WIPED the prior recording to zero.
    Refusing (rather than silently truncating) is the crash-safe behavior: reach for
    a fresh episode id or a fresh store root. This is an ADDITIVE guard — no existing
    method signature changes (no `overwrite=` parameter was added).
    """

    def __init__(self, episode_id: str, root: Path) -> None:
        self.episode_id = episode_id
        self.store_root = root
        super().__init__(
            f"episode {episode_id!r} already has recorded events in the trace store at "
            f"{root}; refusing to truncate it. Use a fresh episode id or a fresh store."
        )


class UnsafeTraceRef(ValueError):
    """A trace reference (episode id, blob sha256, config hash) would escape the store.

    "Download someone's trace and replay it" is the product's core flow, so episode
    ids and the sha256/config_hash pulled out of a (possibly hostile) events.jsonl or
    manifest are untrusted input. Any component containing a path separator, `..`, or
    an absolute path — and any sha256/config_hash that is not exactly 64 lowercase hex
    — is rejected BEFORE it reaches a filesystem join, so a read can never resolve to
    a path outside the store root.
    """

    def __init__(self, kind: str, value: object) -> None:
        self.kind = kind
        self.value = value
        super().__init__(
            f"unsafe {kind} {value!r}: refusing to resolve a trace reference that could "
            "escape the store root"
        )


def _safe_component(value: str, *, kind: str) -> str:
    """Validate a single untrusted path component (e.g. an episode id).

    Rejects anything that could traverse outside the store root: a non-str, an empty
    string, `.`/`..`, an embedded `..`, a `/` or `\\` separator, or an absolute path.
    """
    if not isinstance(value, str) or value in ("", ".", ".."):
        raise UnsafeTraceRef(kind, value)
    if "/" in value or "\\" in value or ".." in value or Path(value).is_absolute():
        raise UnsafeTraceRef(kind, value)
    return value


def _safe_hex64(value: str, *, kind: str) -> str:
    """Validate an untrusted content-addressed reference (sha256 / config_hash)."""
    if not isinstance(value, str) or _SHA256_RE.match(value) is None:
        raise UnsafeTraceRef(kind, value)
    return value


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Durably write `data` to `path`: temp file in the same dir, fsync, atomic rename.

    A crash mid-write can then never leave a half-written manifest/config/blob: the
    reader sees either the old bytes or the fully-written new bytes, never a torn file.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


@dataclass
class _OpenEpisode:
    metadata: Mapping[str, JSONValue]
    config_hash: str
    seam_index: list[SeamIndexEntry] = field(default_factory=list)


class TraceStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self._root = (
            Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="plumbline-trace-"))
        )
        (self._root / "episodes").mkdir(parents=True, exist_ok=True)
        (self._root / "blobs").mkdir(parents=True, exist_ok=True)
        (self._root / "config").mkdir(parents=True, exist_ok=True)
        self._open: dict[str, _OpenEpisode] = {}

    @property
    def root(self) -> Path:
        return self._root

    # --- Episode writes (append-only per episode, §5.1/§5.3) ---
    def open_episode(self, manifest: EpisodeManifest) -> None:
        ep_dir = self._episode_dir(manifest.episode_id)  # validates the episode id
        events_path = ep_dir / "events.jsonl"
        # Crash-safety: refuse to silently truncate a recording. Re-opening the SAME
        # episode id after a crash used to wipe the events to zero. An existing
        # events.jsonl with content means real recorded data — raise, never overwrite.
        # A fresh/absent episode (or one opened but never appended to) opens normally.
        if events_path.exists() and events_path.stat().st_size > 0:
            raise EpisodeExists(manifest.episode_id, self._root)
        ep_dir.mkdir(parents=True, exist_ok=True)
        events_path.write_text("", encoding="utf-8")  # empty, append-only
        self._open[manifest.episode_id] = _OpenEpisode(
            metadata=manifest.metadata, config_hash=manifest.config_hash
        )
        # Persist an initial manifest so the episode is readable while still open
        # (e.g. the recording proxy records a call and the trace is inspected
        # before the episode is closed). close_episode rewrites it with the final
        # seam index.
        self._persist_manifest(manifest.episode_id)

    def append_event(self, episode_id: str, event: SeamEvent) -> None:
        if episode_id not in self._open:
            raise EpisodeNotOpen(episode_id)
        open_ep = self._open[episode_id]
        line = canonical_dumps(_event_to_json(event))
        with (self._episode_dir(episode_id) / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # durability: the event survives a crash right after
        open_ep.seam_index.append(
            SeamIndexEntry(
                seq=event.seq,
                seam=event.seam,
                logical_tick=event.logical_tick,
                request_digest=event.request_digest,
            )
        )
        # NOT re-persisted per event: _persist_manifest re-serializes the whole
        # (growing) seam_index, so doing it every append is O(n^2) disk I/O on the
        # robot's record loop. events.jsonl is append-only and authoritative;
        # open_episode wrote an initial manifest (metadata readable mid-record) and
        # close_episode writes the final seam index.

    def close_episode(self, episode_id: str) -> None:
        if episode_id not in self._open:
            raise EpisodeNotOpen(episode_id)
        self._persist_manifest(episode_id)
        del self._open[episode_id]

    def _persist_manifest(self, episode_id: str) -> None:
        open_ep = self._open[episode_id]
        manifest = EpisodeManifest(
            episode_id=episode_id,
            metadata=open_ep.metadata,
            config_hash=open_ep.config_hash,
            seam_index=tuple(open_ep.seam_index),
        )
        _atomic_write_text(
            self._episode_dir(episode_id) / "manifest.json",
            canonical_dumps(_manifest_to_json(manifest)),
        )

    # --- Episode reads ---
    def load_episode(self, episode_id: str) -> Episode:
        manifest = self.load_manifest(episode_id)
        events_path = self._episode_dir(episode_id) / "events.jsonl"
        events: list[SeamEvent] = []
        lines = events_path.read_text(encoding="utf-8").splitlines()
        # A robot killed mid-append leaves a torn/incomplete FINAL line. Recover every
        # good event and drop only that trailing partial line. But a bad line FOLLOWED
        # by more lines is real corruption (not a crash) and must still fail loudly.
        # The distinction is positional: the last non-empty line may be torn; an
        # interior one may not.
        last_content_idx = max((i for i, line in enumerate(lines) if line), default=-1)
        for lineno, line in enumerate(lines, start=1):
            if not line:
                continue
            try:
                events.append(_event_from_json(_loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                if lineno - 1 == last_content_idx:
                    # Torn/incomplete TRAILING line: a crash mid-write. Recover the good
                    # events, drop the partial final line. Losing the whole mission trace
                    # to one interrupted append is the far worse failure.
                    _LOG.warning(
                        "episode %r: dropped torn trailing line %d of events.jsonl "
                        "(crash mid-append); recovered %d event(s)",
                        episode_id,
                        lineno,
                        len(events),
                    )
                    break
                # An INTERIOR corrupt line is a trace-integrity failure, NOT a thing to
                # silently skip (dropping a recorded event is fabrication-adjacent).
                # Fail loudly and locate it, rather than emit a raw mid-iteration error.
                raise ValueError(
                    f"episode {episode_id!r}: corrupt event at events.jsonl line {lineno}: {exc}"
                ) from exc
        events.sort(key=lambda e: e.seq)
        return Episode(episode_id=episode_id, events=tuple(events), metadata=manifest.metadata)

    def load_manifest(self, episode_id: str) -> EpisodeManifest:
        manifest_path = self._episode_dir(episode_id) / "manifest.json"
        if not manifest_path.exists():
            raise EpisodeNotFound(episode_id, self._root)
        raw = _loads(manifest_path.read_text(encoding="utf-8"))
        return _manifest_from_json(raw)

    def list_episodes(self) -> tuple[str, ...]:
        episodes_dir = self._root / "episodes"
        return tuple(sorted(p.name for p in episodes_dir.iterdir() if p.is_dir()))

    # --- Content-addressed blobs (§5.3) ---
    def put_blob(self, data: bytes, kind: BlobKind, media_type: str | None = None) -> BlobRef:
        sha256 = hashlib.sha256(data).hexdigest()  # trusted: locally computed hex
        path = self._root / "blobs" / f"{sha256}.{kind.value}"
        if not path.exists():  # content-addressed: identical bytes stored once
            _atomic_write_bytes(path, data)  # durable: crash can't leave a torn blob
        return BlobRef(sha256=sha256, kind=kind, media_type=media_type)

    def get_blob(self, ref: BlobRef) -> bytes:
        # Untrusted read: a hostile events.jsonl can name any sha256 (e.g. "../../etc").
        sha256 = _safe_hex64(ref.sha256, kind="blob sha256")
        return (self._root / "blobs" / f"{sha256}.{ref.kind.value}").read_bytes()

    # --- Config snapshots (config/<config_hash>.json, §5.3) ---
    def put_config(self, snapshot: ConfigSnapshot) -> str:
        body: JSONValue = {
            "runtime_config": snapshot.runtime_config,
            "model_versions": dict(snapshot.model_versions),
        }
        config_hash = hashlib.sha256(canonical_dumps(body).encode("utf-8")).hexdigest()
        full: JSONValue = {"config_hash": config_hash, **_as_dict(body)}
        _atomic_write_text(self._root / "config" / f"{config_hash}.json", canonical_dumps(full))
        return config_hash

    def get_config(self, config_hash: str) -> ConfigSnapshot:
        # Untrusted read: config_hash may come from a downloaded manifest.
        safe_hash = _safe_hex64(config_hash, kind="config_hash")
        raw = _loads((self._root / "config" / f"{safe_hash}.json").read_text(encoding="utf-8"))
        return ConfigSnapshot(
            config_hash=raw["config_hash"],
            runtime_config=raw["runtime_config"],
            model_versions=raw["model_versions"],
        )

    def _episode_dir(self, episode_id: str) -> Path:
        # Untrusted on both paths: an episode id can arrive from a downloaded trace
        # (read) or an external producer (write). Validate before joining the root.
        return self._root / "episodes" / _safe_component(episode_id, kind="episode_id")


# --- JSON (de)serialization helpers; all encoders return JSONValue ----------


def _loads(text: str) -> Any:
    return json.loads(text)


def _as_dict(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def _blob_to_json(blob: BlobRef) -> JSONValue:
    return {"sha256": blob.sha256, "kind": blob.kind.value, "media_type": blob.media_type}


def _payload_to_json(payload: Payload) -> JSONValue:
    return {
        "inline": payload.inline,
        "blobs": [_blob_to_json(b) for b in payload.blobs],
    }


def _event_to_json(event: SeamEvent) -> JSONValue:
    return {
        "episode_id": event.episode_id,
        "seq": event.seq,
        "seam": event.seam.value,
        "logical_tick": event.logical_tick,
        "wall_ts": event.wall_ts,
        "request": _payload_to_json(event.request),
        "response": _payload_to_json(event.response),
        "model_id": event.model_id,
        "params": dict(event.params),
        "request_digest": event.request_digest,
        "latency_ms": event.latency_ms,
    }


def _seam_index_to_json(entry: SeamIndexEntry) -> JSONValue:
    return {
        "seq": entry.seq,
        "seam": entry.seam.value,
        "logical_tick": entry.logical_tick,
        "request_digest": entry.request_digest,
    }


def _manifest_to_json(manifest: EpisodeManifest) -> JSONValue:
    return {
        "episode_id": manifest.episode_id,
        "metadata": dict(manifest.metadata),
        "config_hash": manifest.config_hash,
        "seam_index": [_seam_index_to_json(s) for s in manifest.seam_index],
    }


def _blob_from_json(raw: Any) -> BlobRef:
    # Parsed from a (possibly hostile) events.jsonl/manifest: reject a traversal sha256
    # at parse time, so load_episode/load_manifest surface it before any blob read.
    return BlobRef(
        sha256=_safe_hex64(raw["sha256"], kind="blob sha256"),
        kind=BlobKind(raw["kind"]),
        media_type=raw["media_type"],
    )


def _payload_from_json(raw: Any) -> Payload:
    return Payload(
        inline=raw["inline"],
        blobs=tuple(_blob_from_json(b) for b in raw["blobs"]),
    )


def _event_from_json(raw: Any) -> SeamEvent:
    return SeamEvent(
        episode_id=raw["episode_id"],
        seq=raw["seq"],
        seam=Seam(raw["seam"]),
        logical_tick=raw["logical_tick"],
        wall_ts=raw["wall_ts"],
        request=_payload_from_json(raw["request"]),
        response=_payload_from_json(raw["response"]),
        model_id=raw["model_id"],
        params=raw["params"],
        request_digest=raw["request_digest"],
        latency_ms=raw["latency_ms"],
    )


def _seam_index_from_json(raw: Any) -> SeamIndexEntry:
    return SeamIndexEntry(
        seq=raw["seq"],
        seam=Seam(raw["seam"]),
        logical_tick=raw["logical_tick"],
        request_digest=raw["request_digest"],
    )


def _manifest_from_json(raw: Any) -> EpisodeManifest:
    return EpisodeManifest(
        episode_id=raw["episode_id"],
        metadata=raw["metadata"],
        config_hash=raw["config_hash"],
        seam_index=tuple(_seam_index_from_json(s) for s in raw["seam_index"]),
    )
