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


def _write_gate_config(config_path: Path, store_root: str) -> None:
    config_path.write_text(
        "from plumbline.core.store import TraceStore\n"
        "from plumbline.regression import Config, GateSpec, GoldenSet\n\n\n"
        "def build():\n"
        f"    store = TraceStore(root={store_root!r})\n"
        "    golden = GoldenSet(store)\n"
        "    golden.add('ep')\n"
        "    cfg = Config(live_frontier=set(), overrides={}, matchers={})\n"
        "    return GateSpec(store=store, golden=golden, config=cfg, drift_threshold=0.1)\n",
        encoding="utf-8",
    )


def test_cli_gate_emit_feed(tmp_path: Path) -> None:
    _record(str(tmp_path / "traces"))
    config = tmp_path / "gate_config.py"
    _write_gate_config(config, str(tmp_path / "traces"))
    feed = tmp_path / "gate_feed.json"
    assert main(["gate", str(config), "--emit-feed", str(feed)]) == 0
    data: Any = json.loads(feed.read_text(encoding="utf-8"))
    assert "passed" in data


def test_cli_gate_emit_feed_write_failure_preserves_verdict(tmp_path: Path) -> None:
    _record(str(tmp_path / "traces"))
    config = tmp_path / "gate_config.py"
    _write_gate_config(config, str(tmp_path / "traces"))
    bad = tmp_path / "missing_dir" / "feed.json"  # parent dir does not exist
    # A feed-write failure must NOT flip the gate PASS into a nonzero exit.
    assert main(["gate", str(config), "--emit-feed", str(bad)]) == 0
