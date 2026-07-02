"""Experiment B — silent-regression detection against baselines (spec §4, §12).

Two recorded Go2 runs: golden (a good captioner sees the obstacle and the robot
moves away) and candidate (a swapped captioner confidently reports a clear path,
so the robot moves toward the obstacle — the LiDAR-dog inversion). The two runs
have equal latency and equally well-formed LLM calls; only the physical action
differs. The harness must show the latency stack and the generic tracer stay
green while Plumbline goes red.
"""

from collections.abc import Sequence

from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.observability import compare_against_baselines


def _event(
    seam: Seam, request: JSONValue, response: JSONValue, *, model_id: str | None, latency_ms: float
) -> SeamEvent:
    req = Payload(inline=request)
    return SeamEvent(
        episode_id="ep",
        seq=0,
        seam=seam,
        logical_tick=0,
        wall_ts=0.0,
        request=req,
        response=Payload(inline=response),
        model_id=model_id,
        params={},
        request_digest=canonicalize(req).digest,
        latency_ms=latency_ms,
    )


def _run(caption: str, cortex_action: str, move_x: float) -> list[SeamEvent]:
    """One recorded loop iteration: caption -> decide -> act."""
    return [
        _event(
            Seam.SENSOR_TO_CAPTION,
            {"scene": "obstacle"},
            {"caption": caption},
            model_id="openai/vlm",
            latency_ms=40.0,
        ),
        _event(
            Seam.FUSE_TO_DECIDE,
            {"prompt": caption},
            {"choices": [{"message": {"content": cortex_action}}]},
            model_id="openai/cortex",
            latency_ms=120.0,
        ),
        _event(
            Seam.DECIDE_TO_ACT,
            {"commands": [{"type": "move", "x": move_x}]},
            {"executed": True},
            model_id=None,
            latency_ms=0.0,
        ),
    ]


# Golden: obstacle seen -> move away (x = -0.3). Candidate: "clear path" -> move
# toward the obstacle (x = +0.3). Both captions are well-formed and equally fast.
_GOLDEN = _run("an obstacle is 0.3 m directly ahead", "avoid", move_x=-0.3)
_CANDIDATE = _run("the path directly ahead is clear", "advance", move_x=0.3)


def test_experiment_b_baselines_stay_green_while_plumbline_goes_red() -> None:
    comparison = compare_against_baselines(_GOLDEN, _CANDIDATE)

    verdicts = {verdict.name: verdict for verdict in comparison.verdicts}
    # The observability a team would rely on sees nothing wrong.
    assert verdicts["om1-latency"].healthy is True
    assert verdicts["otel-genai-tracer"].healthy is True
    # Plumbline scores the physical decision and catches the inversion.
    assert verdicts["plumbline-behavior"].healthy is False

    assert comparison.caught_by == ("plumbline-behavior",)
    assert comparison.missed_by == ("om1-latency", "otel-genai-tracer")


def test_no_regression_keeps_every_monitor_green() -> None:
    comparison = compare_against_baselines(_GOLDEN, _GOLDEN)
    assert comparison.caught_by == ()
    assert all(verdict.healthy for verdict in comparison.verdicts)


def test_latency_regression_is_caught_by_the_latency_monitor() -> None:
    slow: Sequence[SeamEvent] = [
        _event(
            Seam.FUSE_TO_DECIDE,
            {"prompt": "x"},
            {"choices": [{"message": {"content": "avoid"}}]},
            model_id="openai/cortex",
            latency_ms=600.0,  # 5x slower
        ),
        _GOLDEN[2],
    ]
    comparison = compare_against_baselines(_GOLDEN, slow)
    assert "om1-latency" in comparison.caught_by
