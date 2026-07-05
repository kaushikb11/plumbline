"""The pytest-plumbline plugin: record→replay via `recorded_proxy` under the record
modes, and the gate-as-assertion. The record/replay flow is exercised through a real
sub-pytest run (`pytester`) so the fixtures and CLI options are wired end-to-end.
"""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

# A test body reused across record and replay runs. In `none` mode it asks for no
# upstream at all, so a served response can ONLY have come from the recorded trace.
_LOOP_TEST = """
from plumbline import Context, Payload
from plumbline.record_mode import RecordMode

def test_loop(recorded_proxy, plumbline_record_mode):
    ep = "demo-ep"
    if plumbline_record_mode is RecordMode.NONE:
        proxy = recorded_proxy(ep)                      # replay: no upstream needed
    else:
        proxy = recorded_proxy(ep, upstream=lambda r: Payload(inline={"echo": r.inline}))
    ctx = Context(episode_id=ep, model_id="m", params={}, logical_tick=0)
    out = proxy.forward(Payload(inline={"q": 1}), ctx)
    assert out.inline == {"echo": {"q": 1}}
"""


def _run(pytester: pytest.Pytester, store: Path, mode: str) -> pytest.RunResult:
    # The plugin auto-loads via its pytest11 entry point; no explicit -p needed
    # (passing it would double-register).
    return pytester.runpytest(
        f"--plumbline-record={mode}",
        f"--plumbline-store={store}",
    )


def test_record_once_then_replay_none(pytester: pytest.Pytester, tmp_path: Path) -> None:
    store = tmp_path / "traces"
    pytester.makepyfile(_LOOP_TEST)

    # First run records the loop live (once).
    _run(pytester, store, "once").assert_outcomes(passed=1)
    assert (store / "episodes" / "demo-ep").exists()

    # Second run replays from the trace with NO upstream — proving no live call.
    _run(pytester, store, "none").assert_outcomes(passed=1)


def test_none_mode_without_a_trace_fails_with_a_hint(
    pytester: pytest.Pytester, tmp_path: Path
) -> None:
    pytester.makepyfile(_LOOP_TEST)
    result = _run(pytester, tmp_path / "empty", "none")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*no recorded trace for episode*"])


def test_gate_as_assertion_passes_on_a_clean_config() -> None:
    # Drives the real behavior gate on the committed golden episode via the fixture's
    # helper (no drift -> no failure raised).
    from plumbline.pytest_plugin import _GateAsserter

    config = Path(__file__).resolve().parent.parent / "bench" / "om1_gazebo_gate.py"
    result = _GateAsserter(None).assert_no_drift(str(config))
    assert result.passed is True
