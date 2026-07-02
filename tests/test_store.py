"""TraceStore blob + config round-trip (engineering spec §5.3).

The JSON-metadata + content-addressed-bytes serialization boundary that CLAUDE.md
invariant 3 protects (no pickle, ever). Previously untested.
"""

from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.core.trace import BlobKind, ConfigSnapshot


def test_put_get_blob_round_trips_and_dedups() -> None:
    store = TraceStore()
    data = b"\x00\x01\x02 raw image bytes \xff\xfe"
    ref_a = store.put_blob(data, BlobKind.BIN)
    ref_b = store.put_blob(data, BlobKind.BIN)  # identical bytes
    assert store.get_blob(ref_a) == data  # round-trips byte-identical
    assert ref_a.sha256 == ref_b.sha256  # content-addressed: one copy


def test_config_snapshot_round_trips_with_content_hash() -> None:
    store = TraceStore()
    snapshot = ConfigSnapshot(
        config_hash="",  # ignored; put_config recomputes from content
        runtime_config={"rule": "avoid_obstacles"},
        model_versions={"vlm": "llava:v1", "cortex": "llama3.2"},
    )
    hash_a = store.put_config(snapshot)
    hash_b = store.put_config(
        ConfigSnapshot(
            config_hash="different-but-ignored",
            runtime_config={"rule": "avoid_obstacles"},
            model_versions={"vlm": "llava:v1", "cortex": "llama3.2"},
        )
    )
    assert hash_a == hash_b  # hash covers content, not the passed config_hash
    loaded = store.get_config(hash_a)
    assert loaded.runtime_config == {"rule": "avoid_obstacles"}
    assert loaded.model_versions == {"vlm": "llava:v1", "cortex": "llama3.2"}


def test_list_episodes() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("ep-1", {})
    recorder.close_episode("ep-1")
    assert "ep-1" in store.list_episodes()
