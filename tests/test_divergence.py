"""Divergence-detection test (eng spec §15, §6; CLAUDE.md invariant 5).

Record a baseline, counterfactual-replay with a captioner swap whose output is
structurally different, and assert the replayer HALTS at the first downstream
seam with a distance above threshold — and serves no stale recorded response
past the divergence. Divergence is a result, never silently swallowed.
"""

import random
from collections.abc import Callable

from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy, Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import Payload, SeamEvent, canonicalize

from tests.toyloop import StubCaptioner, StubDecider, make_frames, run_loop


def _swapped_captioner(recorded_request: Payload) -> Payload:
    """A new captioner whose output is structurally different from the recording:
    it drops the metric distance and rephrases entirely, so the live caption is
    embedding-far from the recorded one and the fuse seam can no longer match."""
    return Payload(inline={"caption": "something is moving close and the way looks blocked"})


def test_counterfactual_halts_at_divergence_without_serving_stale() -> None:
    frames = make_frames()
    recorded = run_loop(
        frames,
        StubCaptioner(random.Random(3), 0.0),
        StubDecider(random.Random(3), 0.0),
        episode_id="ep-div",
    )

    store = TraceStore()
    clock = VirtualClock()
    recorder = Recorder(store, clock)
    recorder.open_episode("ep-div", {"task": "obstacle_avoidance"})
    for event in recorded:
        recorder.record(event)
    recorder.close_episode("ep-div")

    matchers: dict[Seam, Matcher] = {
        Seam.CAPTION_TO_FUSE: EmbeddingMatcher(threshold=0.2),
        Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=0.2),
        Seam.DECIDE_TO_ACT: ExactMatcher(),
    }
    replayer = Replayer(store, clock, matchers)

    # Isolated frontier: only the captioner runs live; everything downstream must
    # match the trace or the policy fires.
    overrides: dict[Seam, Callable[[Payload], Payload]] = {
        Seam.SENSOR_TO_CAPTION: _swapped_captioner,
    }
    result = replayer.counterfactual(
        "ep-div",
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides=overrides,
        on_divergence=DivergencePolicy.HALT,
    )

    # Divergence is reported, not raised away.
    assert result.diverged is True
    # First downstream seam after the swap is where the structural change shows.
    assert result.divergence_seam is Seam.CAPTION_TO_FUSE
    assert result.divergence_distance is not None
    assert result.divergence_distance > 0.2  # above the matcher threshold
    # Halt-on-divergence: no fabricated decision was served past the divergence.
    assert all(event.seam is not Seam.FUSE_TO_DECIDE for event in result.events)
    assert all(event.seam is not Seam.DECIDE_TO_ACT for event in result.events)


def _record(events: list[SeamEvent], episode_id: str) -> tuple[TraceStore, VirtualClock]:
    store = TraceStore()
    clock = VirtualClock()
    recorder = Recorder(store, clock)
    recorder.open_episode(episode_id, {})
    for event in events:
        recorder.record(event)
    recorder.close_episode(episode_id)
    return store, clock


def test_counterfactual_go_live_reports_divergence_and_continues() -> None:
    # GO_LIVE continues past a divergence (bounded), but the first divergence is
    # STILL reported — a non-HALT run is never reported clean (invariant 5).
    recorded = run_loop(
        make_frames(),
        StubCaptioner(random.Random(3), 0.0),
        StubDecider(random.Random(3), 0.0),
        episode_id="ep-golive",
    )
    store, clock = _record(list(recorded), "ep-golive")
    matchers: dict[Seam, Matcher] = {
        Seam.CAPTION_TO_FUSE: EmbeddingMatcher(threshold=0.2),
        Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=0.2),
        Seam.DECIDE_TO_ACT: ExactMatcher(),
    }
    result = Replayer(store, clock, matchers).counterfactual(
        "ep-golive",
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _swapped_captioner},
        on_divergence=DivergencePolicy.GO_LIVE,
    )
    assert result.diverged is True  # reported, not swallowed
    assert result.divergence_seam is Seam.CAPTION_TO_FUSE
    # It did NOT halt: it continued and served downstream seams FROM THE TRACE
    # (bounded GO_LIVE), not a fabricated response.
    served_fuse = [e for e in result.events if e.seam is Seam.FUSE_TO_DECIDE]
    recorded_fuse = [
        e for e in store.load_episode("ep-golive").events if e.seam is Seam.FUSE_TO_DECIDE
    ]
    assert served_fuse and recorded_fuse
    assert served_fuse[0].response == recorded_fuse[0].response


def _caption_event(seq: int, caption: str) -> SeamEvent:
    request = Payload(inline={"frame": seq})
    return SeamEvent(
        episode_id="ep-multi",
        seq=seq,
        seam=Seam.SENSOR_TO_CAPTION,
        logical_tick=0,  # both in the same tick
        wall_ts=0.0,
        request=request,
        response=Payload(inline={"caption": caption}),
        model_id=None,
        params={},
        request_digest=canonicalize(request).digest,
        latency_ms=0.0,
    )


def test_counterfactual_preserves_multiple_events_per_tick() -> None:
    # Two calls at the same (tick, seam) — e.g. multi-camera — must both survive,
    # not be collapsed to one by tick grouping (§6.1).
    store, clock = _record(
        [_caption_event(0, "left camera"), _caption_event(1, "right camera")], "ep-multi"
    )
    result = Replayer(store, clock, {}).counterfactual(
        "ep-multi", live_frontier=set(), overrides={}, on_divergence=DivergencePolicy.HALT
    )
    captions = [event for event in result.events if event.seam is Seam.SENSOR_TO_CAPTION]
    assert len(captions) == 2


def test_counterfactual_multi_event_change_not_masked_by_unchanged_sibling() -> None:
    # Two captions in one tick, both live: the FIRST changes, the second is unchanged.
    # The unchanged sibling must NOT erase the first's divergence (invariant 5) —
    # this fails against the last-writer-wins bug.
    fuse_request = Payload(inline={"fused": "x"})
    events = [
        _caption_event(0, "cam0"),
        _caption_event(1, "cam1"),
        SeamEvent(
            episode_id="ep-multi",
            seq=2,
            seam=Seam.CAPTION_TO_FUSE,
            logical_tick=0,
            wall_ts=0.0,
            request=fuse_request,
            response=Payload(inline={"ok": True}),
            model_id=None,
            params={},
            request_digest=canonicalize(fuse_request).digest,
            latency_ms=0.0,
        ),
    ]
    store, clock = _record(events, "ep-multi")

    def override(request: Payload) -> Payload:
        # change camera 0; leave camera 1 exactly as recorded
        frame = request.inline["frame"] if isinstance(request.inline, dict) else None
        return Payload({"caption": "CHANGED"}) if frame == 0 else Payload({"caption": "cam1"})

    result = Replayer(store, clock, {Seam.CAPTION_TO_FUSE: ExactMatcher()}).counterfactual(
        "ep-multi",
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: override},
        on_divergence=DivergencePolicy.HALT,
    )
    assert result.diverged is True
    assert result.divergence_seam is Seam.CAPTION_TO_FUSE
