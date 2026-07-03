"""Plumbline CLI (engineering spec §11).

Subcommands:
  record  — run the recording proxy server: point the runtime's base URL at it; it
            forwards to the real provider and captures each call (§4.2).
  replay  — run the replaying proxy server: serves recorded responses by digest,
            never hitting upstream, so the runtime re-drives deterministically (§4.2).
  gate    — CI for robot behavior: counterfactual-replay golden episodes under a
            candidate config and fail on behavioral drift (§8.4).
  diff    — trace-diff two recorded episodes: show where and at which seam they
            diverged (§11).
  scenes  — author a scenes.json for the Experiment-C leaderboard from a folder of
            images + labels (§4, §7.6).

    plumbline record --upstream https://api.openai.com --store ./traces --episode ep1
    plumbline replay --store ./traces --episode ep1
    plumbline gate   path/to/gate_config.py
    plumbline diff   EPISODE_A EPISODE_B --store ./traces
    plumbline scenes ./images labels.json -o scenes.json

`record` and `replay` run an HTTP server and need uvicorn (pip install
"plumbline[proxy]").

The gate config is a Python file exposing `build() -> GateSpec`. It is Python, not
data, because a candidate config change (a swapped captioner, an edited prompt) is
inherently a seam override — code that re-runs a seam. `build()` assembles the
golden set and the candidate config; the CLI runs the gate and exits non-zero if
behavior drifts, so a GitHub Action (or any CI) can gate a merge on it.
"""

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plumbline.recording import ReconstructingAdapter

from plumbline.bench.scenes import build_scenes, write_scenes_json
from plumbline.core.store import TraceStore
from plumbline.observability import diff_episodes
from plumbline.regression import GateResult, GateSpec, gate


def run_gate(spec: GateSpec) -> GateResult:
    return gate(
        spec.store,
        spec.golden,
        spec.config,
        spec.drift_threshold,
        behavior_matcher=spec.behavior_matcher,
        policy=spec.policy,
        quantile=spec.quantile,
    )


def format_report(result: GateResult) -> str:
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        f"Robot-behavior gate: {verdict}",
        f"  policy={result.policy.value} threshold={result.threshold} "
        f"max_drift={result.max_drift:.3f} diverged={result.diverged_fraction:.0%}",
    ]
    for episode in result.per_episode:
        marker = "ok " if episode.drift <= result.threshold else "DRIFT"
        detail = ""
        if episode.diverged:
            seam = episode.divergence_seam.value if episode.divergence_seam else "?"
            distance = episode.divergence_distance
            suffix = f" dist={distance:.3f}" if distance is not None else ""
            detail = f"  diverged@{seam}{suffix}"
        lines.append(f"  [{marker}] {episode.episode_id} drift={episode.drift:.3f}{detail}")
    return "\n".join(lines)


def load_gate_spec(path: str) -> GateSpec:
    # SECURITY: this executes the config module (a seam override is inherently
    # code). The path must be trusted — a CI-committed file, not arbitrary input.
    if not os.path.isfile(path):
        raise ValueError(f"gate config not found: {path}")
    module_spec = importlib.util.spec_from_file_location("plumbline_gate_config", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot load gate config from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    build = getattr(module, "build", None)
    if build is None:
        raise ValueError(f"{path} must define build() -> GateSpec")
    spec = build()
    if not isinstance(spec, GateSpec):
        raise TypeError(f"{path} build() must return a GateSpec, got {type(spec).__name__}")
    return spec


def run_diff(episode_a: str, episode_b: str, store_root: str) -> int:
    store = TraceStore(root=store_root)
    diff = diff_episodes(store.load_episode(episode_a), store.load_episode(episode_b))
    print(diff.as_text())
    first = diff.first_divergence
    print("identical" if first is None else f"first divergence at {first.seam.value}")
    return 0


def run_scenes(image_dir: str, labels_path: str, out_path: str) -> int:
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        raise ValueError("labels file must be a JSON object of {filename: render_g}")
    scenes = build_scenes(image_dir, labels)
    write_scenes_json(scenes, out_path)
    print(f"wrote {len(scenes)} scenes to {out_path}")
    return 0


def run_export(store_root: str, episode: str, out_path: str, fmt: str) -> int:
    from plumbline.observability.feed import episode_telemetry, write_feed
    from plumbline.observability.otlp import write_otlp

    store = TraceStore(root=store_root)
    loaded = store.load_episode(episode)
    if fmt == "otlp":
        write_otlp(loaded, out_path)
    else:  # telemetry
        write_feed(episode_telemetry(loaded), out_path)
    print(f"wrote {fmt} for episode {episode!r} to {out_path}")
    return 0


def _serve(app: object, host: str, port: int) -> None:  # pragma: no cover - thin uvicorn wrapper
    import importlib

    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        raise SystemExit("record/replay need uvicorn — pip install 'plumbline[proxy]'") from exc
    uvicorn.run(app, host=host, port=port)


def _build_adapter(name: str) -> "ReconstructingAdapter":
    from plumbline.adapters.generic import GenericAgentAdapter
    from plumbline.adapters.om1 import OM1Adapter
    from plumbline.recording import ReconstructingAdapter

    # Adapters with BOTH reconstruct_* hooks (G1's action schema is a placeholder and
    # has no reconstruct_decide_to_act yet, so it can't back the coordinator).
    builders: dict[str, ReconstructingAdapter] = {
        "om1": OM1Adapter(proxy_base_url=""),
        "generic": GenericAgentAdapter(proxy_base_url=""),
    }
    if name not in builders:
        raise ValueError(f"unknown adapter {name!r}; choose om1 or generic")
    return builders[name]


def run_record(
    upstream: str, store_root: str, episode: str, host: str, port: int, adapter: str | None = None
) -> int:
    import httpx

    from plumbline.core.clock import VirtualClock
    from plumbline.core.recorder import Recorder
    from plumbline.proxy.http import AsyncHTTPProxy
    from plumbline.proxy.server import HttpxTransport, make_asgi_app
    from plumbline.proxy.tick import BoundaryTickPolicy

    store = TraceStore(root=store_root)
    # With --adapter, the RecordingCoordinator reconstructs CAPTION_TO_FUSE +
    # DECIDE_TO_ACT into a full four-seam episode; otherwise the plain recorder
    # captures the model seams (now correctly auto-ticked). Both close gap #2.
    recorder: Recorder
    if adapter is None:
        recorder = Recorder(store, VirtualClock())
    else:
        from plumbline.recording import RecordingCoordinator

        recorder = RecordingCoordinator(store, episode_id=episode, adapter=_build_adapter(adapter))
    proxy = AsyncHTTPProxy(
        transport=HttpxTransport(httpx.AsyncClient()),
        recorder=recorder,
        store=store,
        tick_policy=BoundaryTickPolicy(),
    )
    app = make_asgi_app(proxy, upstream=upstream, episode_id=episode)
    print(
        f"recording {upstream} -> episode {episode!r} at {store.root} (listening on {host}:{port})"
    )
    _serve(app, host, port)
    return 0


def run_replay(store_root: str, episode: str, host: str, port: int) -> int:
    from plumbline.proxy.server import make_replay_asgi_app

    store = TraceStore(root=store_root)
    app = make_replay_asgi_app(store, episode_id=episode)
    print(f"replaying episode {episode!r} from {store.root} (listening on {host}:{port})")
    _serve(app, host, port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plumbline", description="Plumbline CLI (§11)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Run the recording proxy server (§4.2)")
    record_parser.add_argument("--upstream", required=True, help="Real provider base URL")
    record_parser.add_argument("--store", required=True, help="TraceStore root directory")
    record_parser.add_argument("--episode", required=True, help="Episode id to record into")
    record_parser.add_argument("--host", default="127.0.0.1")
    record_parser.add_argument("--port", type=int, default=8900)
    record_parser.add_argument(
        "--adapter",
        choices=("om1", "generic"),
        default=None,
        help="Reconstruct a full four-seam episode via this adapter (else model seams only)",
    )

    replay_parser = subparsers.add_parser("replay", help="Run the replaying proxy server (§4.2)")
    replay_parser.add_argument("--store", required=True, help="TraceStore root directory")
    replay_parser.add_argument("--episode", required=True, help="Episode id to serve")
    replay_parser.add_argument("--host", default="127.0.0.1")
    replay_parser.add_argument("--port", type=int, default=8900)

    gate_parser = subparsers.add_parser(
        "gate", help="Run the regression gate over golden episodes (§8)"
    )
    gate_parser.add_argument("config", help="Python file defining build() -> GateSpec")
    gate_parser.add_argument("--emit-feed", help="Write the gate feed JSON (regression dashboard)")

    diff_parser = subparsers.add_parser("diff", help="Trace-diff two recorded episodes (§11)")
    diff_parser.add_argument("episode_a")
    diff_parser.add_argument("episode_b")
    diff_parser.add_argument("--store", required=True, help="TraceStore root directory")

    export_parser = subparsers.add_parser(
        "export", help="Export an episode as OTLP/JSON spans or a telemetry feed (§11)"
    )
    export_parser.add_argument("episode")
    export_parser.add_argument("--store", required=True, help="TraceStore root directory")
    export_parser.add_argument("-o", "--out", required=True, help="Output JSON path")
    export_parser.add_argument("--format", choices=("otlp", "telemetry"), default="otlp")

    scenes_parser = subparsers.add_parser(
        "scenes", help="Author scenes.json for the Experiment-C leaderboard (§4)"
    )
    scenes_parser.add_argument("image_dir", help="Directory of image files")
    scenes_parser.add_argument("labels", help="JSON object {filename: render_g}")
    scenes_parser.add_argument("-o", "--out", default="scenes.json", help="Output scenes.json path")

    args = parser.parse_args(argv)

    try:
        if args.command == "record":
            return run_record(
                args.upstream, args.store, args.episode, args.host, args.port, args.adapter
            )
        if args.command == "replay":
            return run_replay(args.store, args.episode, args.host, args.port)
        if args.command == "gate":
            result = run_gate(load_gate_spec(args.config))
            print(format_report(result))
            if args.emit_feed:
                from plumbline.observability.feed import gate_feed, write_feed

                # A feed-write failure must NOT flip the gate verdict (the CI signal)
                # into a false pass/fail — warn and preserve the real exit code.
                try:
                    write_feed(gate_feed(result), args.emit_feed)
                except OSError as exc:
                    print(f"plumbline gate: could not write feed: {exc}", file=sys.stderr)
            return 0 if result.passed else 1
        if args.command == "diff":
            return run_diff(args.episode_a, args.episode_b, args.store)
        if args.command == "scenes":
            return run_scenes(args.image_dir, args.labels, args.out)
        if args.command == "export":
            return run_export(args.store, args.episode, args.out, args.format)
    except (ValueError, FileNotFoundError, TypeError, KeyError) as exc:
        # Bad user input (missing config/store/episode/labels) — a clean message,
        # not a raw traceback. (SystemExit from _serve propagates unchanged.)
        print(f"plumbline {args.command}: {exc}", file=sys.stderr)
        return 1
    return 2  # unreachable: argparse requires a known subcommand


if __name__ == "__main__":
    raise SystemExit(main())
