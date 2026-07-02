"""Plumbline CLI (engineering spec §11).

Subcommands:
  gate    — CI for robot behavior: counterfactual-replay golden episodes under a
            candidate config and fail on behavioral drift (§8.4).
  diff    — trace-diff two recorded episodes: show where and at which seam they
            diverged (§11).
  scenes  — author a scenes.json for the Experiment-C leaderboard from a folder of
            images + labels (§4, §7.6).

    plumbline gate   path/to/gate_config.py
    plumbline diff   EPISODE_A EPISODE_B --store ./traces
    plumbline scenes ./images labels.json -o scenes.json

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
from collections.abc import Sequence
from pathlib import Path

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plumbline", description="Plumbline CLI (§11)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate_parser = subparsers.add_parser(
        "gate", help="Run the regression gate over golden episodes (§8)"
    )
    gate_parser.add_argument("config", help="Python file defining build() -> GateSpec")

    diff_parser = subparsers.add_parser("diff", help="Trace-diff two recorded episodes (§11)")
    diff_parser.add_argument("episode_a")
    diff_parser.add_argument("episode_b")
    diff_parser.add_argument("--store", required=True, help="TraceStore root directory")

    scenes_parser = subparsers.add_parser(
        "scenes", help="Author scenes.json for the Experiment-C leaderboard (§4)"
    )
    scenes_parser.add_argument("image_dir", help="Directory of image files")
    scenes_parser.add_argument("labels", help="JSON object {filename: render_g}")
    scenes_parser.add_argument("-o", "--out", default="scenes.json", help="Output scenes.json path")

    args = parser.parse_args(argv)

    if args.command == "gate":
        result = run_gate(load_gate_spec(args.config))
        print(format_report(result))
        return 0 if result.passed else 1
    if args.command == "diff":
        return run_diff(args.episode_a, args.episode_b, args.store)
    if args.command == "scenes":
        return run_scenes(args.image_dir, args.labels, args.out)
    return 2  # unreachable: argparse requires a known subcommand


if __name__ == "__main__":
    raise SystemExit(main())
