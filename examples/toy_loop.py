"""The zero-setup Plumbline demo: record a toy two-model perception→action loop and
faithful-replay it, proving the model I/O survives byte-identically.

No network, no extras, no GPU — runs on a bare `pip install -e .`::

    python examples/toy_loop.py

This is the substrate guarantee (CLAUDE.md invariant 2) in ~40 lines, using the
flat public API (`from plumbline import …`) and the `make_seam_event` helper that
computes the request digest for you.
"""

from plumbline import (
    Payload,
    Recorder,
    Replayer,
    Seam,
    TraceStore,
    VirtualClock,
    canonicalize,
    make_seam_event,
)

EPISODE = "toy-loop-001"


def captioner(frame: str) -> str:
    """Toy VLM: turn a sensor frame into a caption."""
    return "obstacle ahead" if "rock" in frame else "path clear"


def decider(caption: str) -> str:
    """Toy LLM: turn a caption into an action."""
    return "turn_left" if "obstacle" in caption else "advance"


def main() -> None:
    frames = ["open field", "rock at 0.4m", "open field"]

    # Record the loop: two model calls per tick (SENSOR_TO_CAPTION, FUSE_TO_DECIDE).
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode(EPISODE, {"task": "toy_obstacle_avoidance"})
    seq = 0
    for tick, frame in enumerate(frames):
        caption = captioner(frame)
        action = decider(caption)
        for seam, req, resp in (
            (Seam.SENSOR_TO_CAPTION, frame, caption),
            (Seam.FUSE_TO_DECIDE, caption, action),
        ):
            recorder.record(
                make_seam_event(
                    episode_id=EPISODE,
                    seq=seq,
                    seam=seam,
                    logical_tick=tick,
                    request=Payload(inline={"input": req}),
                    response=Payload(inline={"output": resp}),
                )
            )
            seq += 1
    recorder.close_episode(EPISODE)

    # Faithful replay: reload the trace and prove the model I/O is byte-identical.
    result = Replayer(store, VirtualClock(), {}).faithful(EPISODE)
    recorded = store.load_episode(EPISODE).events

    def io_bytes(events: object) -> bytes:
        return b"".join(
            canonicalize(e.request).digest.encode() + canonicalize(e.response).digest.encode()
            for e in events  # type: ignore[attr-defined]
        )

    assert result.diverged is False
    assert io_bytes(result.events) == io_bytes(recorded)
    print(
        f"✓ recorded {len(recorded)} model calls across {len(frames)} ticks in episode {EPISODE!r}"
    )
    print("✓ faithful replay reproduced every request/response byte-identically")
    print("\nNext: try the CI gate on a real recorded episode —")
    print("    plumbline gate bench/om1_gazebo_gate.py")


if __name__ == "__main__":
    main()
