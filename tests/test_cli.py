"""The `plumbline gate` CLI (engineering spec §8.4, §11).

The bundled example config passes; an injected regression (an incompatible
captioner) fails with a non-zero exit and a report attributing the divergence.
"""

from plumbline.bench.example_gate import (
    INCOMPATIBLE_CAPTION,
    captioner,
    demo_matchers,
    record_demo_episode,
)
from plumbline.cli import format_report, load_gate_spec, main, run_gate
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.regression import Config, GateSpec, GoldenSet

_EXAMPLE = "plumbline/bench/example_gate.py"


def _failing_spec() -> GateSpec:
    store = TraceStore()
    episode_id = record_demo_episode(store)
    golden = GoldenSet(store)
    golden.add(episode_id)
    config = Config(
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: captioner(INCOMPATIBLE_CAPTION)},
        matchers=demo_matchers(),
    )
    return GateSpec(store=store, golden=golden, config=config, drift_threshold=0.1)


def test_cli_gate_passes_on_the_bundled_example() -> None:
    assert main(["gate", _EXAMPLE]) == 0


def test_loaded_example_spec_passes() -> None:
    result = run_gate(load_gate_spec(_EXAMPLE))
    assert result.passed is True
    assert result.max_drift == 0.0
    assert "PASS" in format_report(result)


def test_cli_gate_fails_on_an_injected_regression() -> None:
    result = run_gate(_failing_spec())
    assert result.passed is False
    assert result.diverged_fraction == 1.0

    report = format_report(result)
    assert "FAIL" in report
    assert "diverged@caption_to_fuse" in report
