"""CI gate zero: the determinism property test (eng spec §15, CLAUDE.md invariant 2).

Record a toy two-model loop, store it, faithful-load it, and assert the model I/O
survives the store round-trip byte-identically. The stronger property — that
re-DRIVING the loop while serving each recorded response by request_digest
reproduces the same decisions (and a fresh seed diverges) — is proven in
`test_reexecution.py`. Both must stay green on every commit (invariant 2).
"""

import random

from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import Replayer
from plumbline.core.store import TraceStore

from tests.toyloop import (
    StubCaptioner,
    StubDecider,
    default_matchers,
    make_frames,
    model_io_bytes,
    run_loop,
)


def test_faithful_replay_reproduces_model_io_byte_identically() -> None:
    frames = make_frames()

    # Precondition: the stub models are genuinely nondeterministic at temp > 0,
    # so reproducing a recorded run is a real guarantee, not a trivial echo.
    run_a = run_loop(
        frames,
        StubCaptioner(random.Random(1), 0.8),
        StubDecider(random.Random(1), 0.8),
        episode_id="sanity",
    )
    run_b = run_loop(
        frames,
        StubCaptioner(random.Random(2), 0.8),
        StubDecider(random.Random(2), 0.8),
        episode_id="sanity",
    )
    assert model_io_bytes(run_a) != model_io_bytes(run_b)
    # Same seed -> identical: the fixtures themselves are deterministic.
    run_a2 = run_loop(
        frames,
        StubCaptioner(random.Random(1), 0.8),
        StubDecider(random.Random(1), 0.8),
        episode_id="sanity",
    )
    assert model_io_bytes(run_a) == model_io_bytes(run_a2)

    # Record one nondeterministic run.
    recorded = run_loop(
        frames,
        StubCaptioner(random.Random(7), 0.8),
        StubDecider(random.Random(7), 0.8),
        episode_id="ep-det",
    )
    store = TraceStore()
    clock = VirtualClock()
    recorder = Recorder(store, clock)
    recorder.open_episode("ep-det", {"task": "obstacle_avoidance"})
    for event in recorded:
        recorder.record(event)
    recorder.close_episode("ep-det")

    # Faithful replay must reproduce the recorded model I/O byte-for-byte.
    replayer = Replayer(store, clock, default_matchers())
    result = replayer.faithful("ep-det")
    assert result.episode_id == "ep-det"
    assert result.diverged is False
    assert model_io_bytes(result.events) == model_io_bytes(recorded)
