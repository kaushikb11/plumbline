"""Trace-diff viewer (engineering spec §11).

Two runs of the same episode are aligned by (logical_tick, seam): an unchanged run
diffs clean, a swapped captioner shows CHANGED at that seam, and a counterfactual
that halted before the action seam leaves the downstream seams as ONLY_A.
"""

from plumbline.core.seam import Seam
from plumbline.core.trace import Episode, JSONValue, Payload, SeamEvent, canonicalize
from plumbline.observability import StepStatus, diff_episodes, diff_traces


def _event(seq: int, seam: Seam, tick: int, request: JSONValue, response: JSONValue) -> SeamEvent:
    req = Payload(inline=request)
    return SeamEvent(
        episode_id="ep",
        seq=seq,
        seam=seam,
        logical_tick=tick,
        wall_ts=float(tick),
        request=req,
        response=Payload(inline=response),
        model_id=None,
        params={},
        request_digest=canonicalize(req).digest,
        latency_ms=0.0,
    )


def _golden() -> list[SeamEvent]:
    return [
        _event(0, Seam.SENSOR_TO_CAPTION, 0, {"scene": "obstacle"}, {"caption": "obstacle ahead"}),
        _event(1, Seam.CAPTION_TO_FUSE, 0, {"captions": ["obstacle ahead"]}, {"fused": "..."}),
        _event(2, Seam.FUSE_TO_DECIDE, 0, {"prompt": "..."}, {"action_plan": {"action": "avoid"}}),
        _event(3, Seam.DECIDE_TO_ACT, 0, {"commands": [{"type": "move", "x": -0.3}]}, {"ok": True}),
    ]


def test_identical_runs_diff_clean() -> None:
    diff = diff_traces(_golden(), _golden())
    assert diff.identical is True
    assert diff.first_divergence is None
    assert all(step.status is StepStatus.SAME for step in diff.steps)


def test_swapped_caption_shows_changed_at_that_seam() -> None:
    candidate = _golden()
    candidate[0] = _event(
        0, Seam.SENSOR_TO_CAPTION, 0, {"scene": "obstacle"}, {"caption": "the path is clear"}
    )
    diff = diff_traces(_golden(), candidate)

    assert diff.identical is False
    first = diff.first_divergence
    assert first is not None
    assert first.seam is Seam.SENSOR_TO_CAPTION
    assert first.status is StepStatus.CHANGED
    assert first.response_changed is True
    assert first.request_changed is False


def test_halted_counterfactual_leaves_downstream_seams_only_in_golden() -> None:
    # A counterfactual that halted at CAPTION_TO_FUSE: only the live SENSOR seam.
    halted = [
        _event(0, Seam.SENSOR_TO_CAPTION, 0, {"scene": "obstacle"}, {"caption": "clear path"})
    ]
    diff = diff_traces(_golden(), halted)

    statuses = {step.seam: step.status for step in diff.steps}
    assert statuses[Seam.SENSOR_TO_CAPTION] is StepStatus.CHANGED
    assert statuses[Seam.CAPTION_TO_FUSE] is StepStatus.ONLY_A
    assert statuses[Seam.FUSE_TO_DECIDE] is StepStatus.ONLY_A
    assert statuses[Seam.DECIDE_TO_ACT] is StepStatus.ONLY_A
    # The rendered view names the seam that introduced the divergence.
    assert "sensor_to_caption" in diff.as_text()
    assert "first divergence" in diff.as_text()


def test_diff_episodes_wraps_event_sequences() -> None:
    episode = Episode(episode_id="ep", events=tuple(_golden()), metadata={})
    assert diff_episodes(episode, episode).identical is True
