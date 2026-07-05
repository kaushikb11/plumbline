"""Crash-recovery + path-traversal hardening for the TraceStore (framework review).

Two production blockers:

  1. A robot killed mid-write must not lose the whole mission trace. A torn/incomplete
     TRAILING line in events.jsonl is a crash artifact and must be recovered from (drop
     only the partial line); an INTERIOR bad line is real corruption and must still
     raise. And re-opening the SAME episode id after a crash must NOT wipe the recording.

  2. "Download someone's trace and replay it" is the core flow, so episode ids and the
     sha256/config_hash inside a (possibly hostile) events.jsonl/manifest are UNTRUSTED.
     A `"sha256": "../../../etc/passwd"` must be rejected before any filesystem join.

All fixes are additive (new validation + new typed exceptions); no frozen signature
changed (CLAUDE.md invariant 1).
"""

import pytest
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import (
    EpisodeExists,
    TraceStore,
    UnsafeTraceRef,
)
from plumbline.core.trace import (
    BlobKind,
    BlobRef,
    EpisodeManifest,
    Payload,
    SeamEvent,
    canonicalize,
)


def _payload(x: object) -> Payload:
    return Payload(inline={"v": x})  # type: ignore[dict-item]


def _manifest(episode_id: str) -> EpisodeManifest:
    return EpisodeManifest(episode_id=episode_id, metadata={}, config_hash="", seam_index=())


def _record_n(store: TraceStore, episode_id: str, n: int) -> None:
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode(episode_id, {"robot": "go2"})
    for seq in range(n):
        req = _payload(seq)
        recorder.record(
            SeamEvent(
                episode_id,
                seq,
                Seam.FUSE_TO_DECIDE,
                seq,
                0.0,
                req,
                req,
                None,
                {},
                canonicalize(req).digest,
                0.0,
            )
        )
    recorder.close_episode(episode_id)


# --- Blocker 1a: torn trailing line is recoverable; interior is not ----------


def test_torn_trailing_line_recovers_all_good_events() -> None:
    store = TraceStore()
    _record_n(store, "ep", 3)  # three good events on disk
    # A crash mid-append: an interrupted, unparseable FINAL line (no closing brace).
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    events_path.write_text(
        events_path.read_text() + '{"episode_id": "ep", "seq": 3, "seam":',
        encoding="utf-8",
    )
    episode = store.load_episode("ep")  # must NOT raise
    assert len(episode.events) == 3  # every good event recovered
    assert [e.seq for e in episode.events] == [0, 1, 2]  # only the partial line dropped


def test_interior_corrupt_line_still_raises() -> None:
    store = TraceStore()
    _record_n(store, "ep", 3)
    # Inject a bad line in the MIDDLE (followed by more content) — real corruption,
    # not a crash. Splice a garbage line between the recorded ones.
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    good = events_path.read_text().splitlines()
    corrupted = good[:1] + ['{"seq": 1, "seam":'] + good[1:]
    events_path.write_text("\n".join(corrupted) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt event at events.jsonl line 2"):
        store.load_episode("ep")


def test_torn_first_and_only_line_recovers_zero_events() -> None:
    # Crash before the first append completed: the sole line is torn. Recover the
    # (empty) set of good events rather than losing the openable episode.
    store = TraceStore()
    store.open_episode(_manifest("ep"))
    store.close_episode("ep")
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    events_path.write_text('{"episode_id": "ep", "seq": 0', encoding="utf-8")
    episode = store.load_episode("ep")
    assert episode.events == ()


# --- Blocker 1b: re-opening a non-empty episode must not wipe it --------------


def test_reopening_nonempty_episode_raises_and_does_not_wipe() -> None:
    store = TraceStore()
    _record_n(store, "ep", 2)
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    before = events_path.read_text()
    # A restarted recorder points at the same root + id after a crash.
    with pytest.raises(EpisodeExists):
        store.open_episode(_manifest("ep"))
    assert events_path.read_text() == before  # NOT truncated
    assert len(store.load_episode("ep").events) == 2  # recording intact


def test_fresh_and_absent_episode_open_normally() -> None:
    store = TraceStore()
    # Absent id: opens fine.
    store.open_episode(_manifest("fresh"))
    store.close_episode("fresh")
    # An episode opened but never appended to (empty events.jsonl) is re-openable: an
    # empty file is not "recorded data" to protect.
    events_path = store.root / "episodes" / "fresh" / "events.jsonl"
    assert events_path.read_text() == ""
    store.open_episode(_manifest("fresh"))


# --- Blocker 2: path-traversal on the read AND write paths --------------------


def test_traversal_episode_id_rejected_on_write() -> None:
    store = TraceStore()
    with pytest.raises(UnsafeTraceRef):
        store.open_episode(_manifest("../../evil"))
    with pytest.raises(UnsafeTraceRef):
        store.load_episode("/etc/passwd")


def test_traversal_blob_ref_reads_nothing_outside_root(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A hostile blob ref must not read a file outside the store root.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    store = TraceStore(root=tmp_path / "store")
    evil = BlobRef(sha256="../../secret", kind=BlobKind.BIN, media_type=None)
    with pytest.raises(UnsafeTraceRef):
        store.get_blob(evil)
    # A bare absolute path is also refused.
    with pytest.raises(UnsafeTraceRef):
        store.get_blob(BlobRef(sha256=str(secret), kind=BlobKind.BIN, media_type=None))


def test_traversal_config_hash_rejected() -> None:
    store = TraceStore()
    with pytest.raises(UnsafeTraceRef):
        store.get_config("../../../etc/passwd")
    with pytest.raises(UnsafeTraceRef):
        store.get_config("not-64-hex")


def test_valid_refs_still_work() -> None:
    # A valid 64-hex sha256 and a normal episode id round-trip unchanged.
    store = TraceStore()
    ref = store.put_blob(b"raw bytes", BlobKind.BIN)
    assert len(ref.sha256) == 64 and store.get_blob(ref) == b"raw bytes"
    _record_n(store, "ep-1", 1)
    assert len(store.load_episode("ep-1").events) == 1
