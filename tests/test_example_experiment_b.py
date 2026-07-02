"""Offline dry-run of the Experiment-B example (no network).

Exercises the pure `build_run` assembly and asserts the headline property — latency
+ tracer green, only Plumbline red — mirroring the deterministic guarantee in
test_baselines.py. Never touches Ollama.
"""

import pytest

pytest.importorskip("httpx")
pytest.importorskip("PIL")

import examples.experiment_b as exp_b  # noqa: E402
from plumbline.observability.baselines import compare_against_baselines  # noqa: E402


def test_build_run_reproduces_baseline_inversion() -> None:
    golden = exp_b.build_run(
        "golden",
        caption="an obstacle is directly ahead on the floor",
        decision={"action": "back_up"},
        cap_latency_ms=100.0,
        dec_latency_ms=50.0,
    )
    candidate = exp_b.build_run(
        "candidate",
        caption="the path ahead is clear",
        decision={"action": "move_forward"},
        cap_latency_ms=100.0,
        dec_latency_ms=50.0,
    )
    comparison = compare_against_baselines(golden, candidate)
    assert comparison.caught_by == ("plumbline-behavior",)  # only Plumbline sees it
    assert comparison.missed_by == ("om1-latency", "otel-genai-tracer")


def test_main_is_callable() -> None:
    assert callable(exp_b.main)
