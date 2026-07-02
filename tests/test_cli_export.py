"""CLI `export` subcommand — episode -> OTLP/JSON or telemetry feed (§11)."""

import json
from pathlib import Path
from typing import Any

from plumbline.cli import main
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import Payload, SeamEvent, canonicalize


def _record(root: str) -> None:
    store = TraceStore(root=root)
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("ep", {})
    request = Payload(inline={"m": 1})
    recorder.record(
        SeamEvent(
            "ep",
            0,
            Seam.FUSE_TO_DECIDE,
            0,
            0.0,
            request,
            Payload(inline={"choices": [{"message": {"content": "x"}}]}),
            "openai/gpt-4o",
            {},
            canonicalize(request).digest,
            5.0,
        )
    )
    recorder.close_episode("ep")


def test_cli_export_otlp(tmp_path: Path) -> None:
    _record(str(tmp_path / "traces"))
    out = tmp_path / "spans.json"
    code = main(
        ["export", "ep", "--store", str(tmp_path / "traces"), "-o", str(out), "--format", "otlp"]
    )
    assert code == 0
    document: Any = json.loads(out.read_text(encoding="utf-8"))
    assert document["resourceSpans"][0]["scopeSpans"][0]["spans"]


def test_cli_export_telemetry(tmp_path: Path) -> None:
    _record(str(tmp_path / "traces"))
    out = tmp_path / "feed.json"
    code = main(
        [
            "export",
            "ep",
            "--store",
            str(tmp_path / "traces"),
            "-o",
            str(out),
            "--format",
            "telemetry",
        ]
    )
    assert code == 0
    feed: Any = json.loads(out.read_text(encoding="utf-8"))
    assert feed["episode_id"] == "ep"
