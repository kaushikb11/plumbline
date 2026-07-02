"""Dashboard-feed rollups (engineering spec §11)."""

from typing import Any

from plumbline.core.seam import Seam
from plumbline.core.trace import Episode, JSONValue, Payload, SeamEvent, canonicalize
from plumbline.observability.baselines import BaselineComparison, MonitorVerdict
from plumbline.observability.feed import baseline_feed, episode_telemetry, gate_feed
from plumbline.regression.gate import EpisodeDrift, FailurePolicy, GateResult


def _event(seq: int, seam: Seam, latency_ms: float, response: JSONValue | None = None) -> SeamEvent:
    request = Payload(inline={"m": seq})
    return SeamEvent(
        episode_id="ep",
        seq=seq,
        seam=seam,
        logical_tick=seq // 2,
        wall_ts=float(seq),
        request=request,
        response=Payload(inline=response or {"ok": True}),
        model_id=None,
        params={},
        request_digest=canonicalize(request).digest,
        latency_ms=latency_ms,
    )


def test_episode_telemetry_rollups() -> None:
    episode = Episode(
        "ep",
        (
            _event(0, Seam.SENSOR_TO_CAPTION, 10.0),
            _event(1, Seam.FUSE_TO_DECIDE, 30.0),
            _event(2, Seam.SENSOR_TO_CAPTION, 20.0),
        ),
        {},
    )
    feed: Any = episode_telemetry(episode)
    assert feed["episode_id"] == "ep"
    seams = {row["seam"]: row for row in feed["seams"]}
    assert seams[Seam.SENSOR_TO_CAPTION.value]["count"] == 2
    assert seams[Seam.SENSOR_TO_CAPTION.value]["latency_mean_ms"] == 15.0
    ticks = {row["logical_tick"]: row["seam_count"] for row in feed["ticks"]}
    assert ticks == {0: 2, 1: 1}


def test_gate_feed_mirrors_result() -> None:
    result = GateResult(
        passed=False,
        threshold=0.1,
        policy=FailurePolicy.ANY,
        per_episode=(
            EpisodeDrift(
                "e1",
                drift=0.5,
                diverged=True,
                divergence_seam=Seam.CAPTION_TO_FUSE,
                divergence_distance=0.3,
            ),
            EpisodeDrift(
                "e2", drift=0.0, diverged=False, divergence_seam=None, divergence_distance=None
            ),
        ),
    )
    feed: Any = gate_feed(result)
    assert feed["passed"] is False
    assert feed["max_drift"] == 0.5
    assert feed["diverged_fraction"] == 0.5
    assert feed["episodes"][0]["divergence_seam"] == Seam.CAPTION_TO_FUSE.value
    assert feed["episodes"][1]["divergence_seam"] is None


def test_baseline_feed_status_and_partition() -> None:
    comparison = BaselineComparison(
        verdicts=(
            MonitorVerdict("om1-latency", True, "ok"),
            MonitorVerdict("plumbline-behavior", False, "drift 1.00"),
        )
    )
    feed: Any = baseline_feed(comparison)
    statuses = {row["name"]: row["status"] for row in feed["verdicts"]}
    assert statuses["om1-latency"] == "green"
    assert statuses["plumbline-behavior"] == "red"
    assert feed["caught_by"] == ["plumbline-behavior"]
    assert feed["missed_by"] == ["om1-latency"]
