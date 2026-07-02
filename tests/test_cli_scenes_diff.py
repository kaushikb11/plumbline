"""The `plumbline scenes` and `plumbline diff` subcommands (engineering spec §4, §11).

`scenes` authors an Experiment-C scenes.json from images + labels; `diff` renders
where two recorded episodes diverged.
"""

import json
from pathlib import Path

import pytest
from plumbline.bench.leaderboard import load_scenes
from plumbline.cli import main, run_diff
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize


def _event(episode_id: str, seam: Seam, tick: int, response: JSONValue) -> SeamEvent:
    request = Payload(inline={"tick": tick})
    return SeamEvent(
        episode_id=episode_id,
        seq=0,
        seam=seam,
        logical_tick=tick,
        wall_ts=0.0,
        request=request,
        response=Payload(inline=response),
        model_id=None,
        params={},
        request_digest=canonicalize(request).digest,
        latency_ms=0.0,
    )


def _record(store: TraceStore, episode_id: str, caption: str) -> None:
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode(episode_id, {})
    recorder.record(_event(episode_id, Seam.SENSOR_TO_CAPTION, 0, {"caption": caption}))
    recorder.close_episode(episode_id)


def test_scenes_subcommand_builds_loadable_scenes(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "hall-01.png").write_bytes(b"\x89PNG\r\n\x1a\n fake image bytes")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"hall-01.png": "obstacle dead ahead"}), encoding="utf-8")
    out = tmp_path / "scenes.json"

    assert main(["scenes", str(images), str(labels), "-o", str(out)]) == 0
    scenes = load_scenes(out)
    assert len(scenes) == 1
    assert scenes[0].scene_id == "hall-01"
    assert scenes[0].image.startswith("data:image/png;base64,")
    assert scenes[0].render_g == "obstacle dead ahead"


def test_diff_subcommand_reports_divergence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_root = tmp_path / "traces"
    store = TraceStore(root=store_root)
    _record(store, "golden", "obstacle ahead")
    _record(store, "candidate", "path is clear")

    assert main(["diff", "golden", "candidate", "--store", str(store_root)]) == 0
    out = capsys.readouterr().out
    assert "first divergence at sensor_to_caption" in out


def test_diff_identical_episodes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store_root = tmp_path / "traces"
    store = TraceStore(root=store_root)
    _record(store, "a", "same caption")
    _record(store, "b", "same caption")

    assert run_diff("a", "b", str(store_root)) == 0
    assert "identical" in capsys.readouterr().out
