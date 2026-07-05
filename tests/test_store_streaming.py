"""Streaming event reads for the TraceStore (production MAJOR: trace-store RAM spike).

`load_episode` used to slurp events.jsonl whole (`read_text().splitlines()` -> a giant
string + a giant list of lines + the events list, a ~3x transient peak that OOM'd CI on a
100k-event mission). The read is now a line-by-line stream, and an additive
`iter_events()` yields events with bounded memory regardless of episode length.

These tests pin: (1) iter_events yields the same events as load_episode().events;
(2) the torn-trailing / interior-corrupt crash semantics are preserved when streaming;
(3) a large synthetic episode is iterable without materializing all lines.

All additive: no frozen signature changed (CLAUDE.md invariant 1).
"""

import json
import tracemalloc

import pytest
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import EpisodeNotFound, TraceStore, UnsafeTraceRef
from plumbline.core.trace import (
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


# --- iter_events matches load_episode ---------------------------------------


def test_iter_events_yields_same_events_as_load_episode() -> None:
    store = TraceStore()
    _record_n(store, "ep", 12)
    loaded = store.load_episode("ep").events
    streamed = tuple(store.iter_events("ep"))
    assert streamed == loaded  # same events, same order (append order == seq order)
    assert [e.seq for e in streamed] == list(range(12))


def test_iter_events_on_empty_episode_yields_nothing() -> None:
    store = TraceStore()
    store.open_episode(_manifest("ep"))
    store.close_episode("ep")
    assert tuple(store.iter_events("ep")) == ()
    assert store.load_episode("ep").events == ()


def test_iter_events_missing_episode_raises_eagerly() -> None:
    # A generator that deferred validation to first next() would surprise callers; the
    # not-found error fires at the call site, not mid-iteration.
    store = TraceStore()
    with pytest.raises(EpisodeNotFound):
        store.iter_events("never-recorded")


def test_iter_events_rejects_unsafe_episode_id_eagerly() -> None:
    store = TraceStore()
    with pytest.raises(UnsafeTraceRef):
        store.iter_events("../../etc/passwd")


# --- crash semantics preserved when streaming -------------------------------


def test_iter_events_recovers_torn_trailing_line() -> None:
    store = TraceStore()
    _record_n(store, "ep", 3)  # three good events on disk
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    events_path.write_text(
        events_path.read_text() + '{"episode_id": "ep", "seq": 3, "seam":',
        encoding="utf-8",
    )
    # Same recovery as load_episode: drop the partial final line, keep the good events.
    streamed = tuple(store.iter_events("ep"))
    assert [e.seq for e in streamed] == [0, 1, 2]
    assert streamed == store.load_episode("ep").events


def test_iter_events_interior_corrupt_line_raises_located() -> None:
    store = TraceStore()
    _record_n(store, "ep", 3)
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    good = events_path.read_text().splitlines()
    corrupted = good[:1] + ['{"seq": 1, "seam":'] + good[1:]
    events_path.write_text("\n".join(corrupted) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt event at events.jsonl line 2"):
        list(store.iter_events("ep"))  # must drain to hit the interior line
    with pytest.raises(ValueError, match="corrupt event at events.jsonl line 2"):
        store.load_episode("ep")  # load_episode agrees


def test_iter_events_torn_line_followed_by_blank_lines_still_recovers() -> None:
    # Trailing blank lines after the torn line are not "content follows": the torn line
    # is still the last NON-empty line and must be recovered, not raised.
    store = TraceStore()
    _record_n(store, "ep", 2)
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    events_path.write_text(
        events_path.read_text() + '{"episode_id": "ep", "seq": 2, "seam":\n\n\n',
        encoding="utf-8",
    )
    assert [e.seq for e in store.iter_events("ep")] == [0, 1]


# --- bounded memory regardless of episode length ----------------------------


def test_iter_events_streams_large_episode_without_materializing() -> None:
    store = TraceStore()
    n = 5000
    _record_n(store, "big", n)
    events_path = store.root / "episodes" / "big" / "events.jsonl"
    file_bytes = events_path.stat().st_size
    assert file_bytes > 1_000_000  # a genuinely large trace (>1 MB on disk)

    # Consume one event at a time, discarding each. Peak allocation must stay a small
    # fraction of the file size — proof we never build the whole-file string / line list
    # / event list. load_episode (which returns the full tuple) legitimately would.
    tracemalloc.start()
    count = 0
    for _ev in store.iter_events("big"):
        count += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert count == n
    assert peak < file_bytes // 4  # bounded well under the on-disk size


def test_iter_events_peak_far_below_load_episode_peak() -> None:
    store = TraceStore()
    _record_n(store, "big", 4000)

    tracemalloc.start()
    for _ev in store.iter_events("big"):
        pass
    _, iter_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _ = store.load_episode("big").events
    _, load_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Streaming holds ~one event; loading holds the whole tuple. Order-of-magnitude gap.
    assert iter_peak * 10 < load_peak


def test_iter_events_line_by_line_matches_manual_jsonl_parse() -> None:
    # Independent oracle: parse the raw jsonl ourselves and compare seqs.
    store = TraceStore()
    _record_n(store, "ep", 20)
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    expected_seqs = [
        json.loads(line)["seq"] for line in events_path.read_text().splitlines() if line
    ]
    assert [e.seq for e in store.iter_events("ep")] == expected_seqs
