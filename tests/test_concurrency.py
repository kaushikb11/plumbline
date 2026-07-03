"""Real-thread concurrency for RecordingSession — the race the locks exist for.

The whole reason RecordingSession holds a lock is the tap-thread-vs-loop race: the
Zenoh tap fires `record_bus_sample` on a native thread while the loop calls
`set_tick`/`record`. Every other test drives the tap handler synchronously on the
test thread, so this is the one that actually contends real threads (framework
review, test-quality finding).
"""

import threading

from plumbline.core.store import TraceStore
from plumbline.session import RecordingSession
from plumbline.transport.bus import BusSample

_N_THREADS = 8
_PER_THREAD = 40


def test_bus_samples_and_set_tick_are_coherent_under_real_thread_contention() -> None:
    store = TraceStore()
    session = RecordingSession(store, episode_id="ep", metadata={})
    session.open()

    start = threading.Barrier(_N_THREADS + 1)

    def worker() -> None:
        start.wait()  # maximize contention: all threads unleash together
        for _ in range(_PER_THREAD):
            session.record_bus_sample(BusSample(key_expr="cmd_vel", payload=None, wall_ts=0.0))

    threads = [threading.Thread(target=worker) for _ in range(_N_THREADS)]
    for t in threads:
        t.start()
    start.wait()
    # The loop thread advances the tick monotonically while the taps record.
    for tick in range(1, 200):
        session.set_tick(tick)
    for t in threads:
        t.join()
    session.close()

    events = store.load_episode("ep").events
    # No lost updates / no collisions: exactly N*PER events, seq gap-free 0..M-1.
    assert len(events) == _N_THREADS * _PER_THREAD
    assert [e.seq for e in events] == list(range(len(events)))
    # Tick is stamped WITH seq under one lock acquisition, and set_tick only ever
    # increases the tick, so ticks are non-decreasing in seq order — never a stale
    # tick landing after a newer one (the exact race the lock prevents).
    ticks = [e.logical_tick for e in events]
    assert ticks == sorted(ticks)


def test_bus_sample_dropped_not_raised_when_recorded_after_close() -> None:
    # The tap fires on its own thread; a sample arriving after close() must be
    # dropped (a crashing callback can wedge the subscriber), not raise.
    store = TraceStore()
    session = RecordingSession(store, episode_id="ep", metadata={})
    session.open()
    session.close()
    session.record_bus_sample(BusSample(key_expr="cmd_vel", payload=None, wall_ts=0.0))  # no raise
    assert store.load_episode("ep").events == ()
