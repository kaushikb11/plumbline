"""The REAL Gazebo golden episode in CI (§8.4, §15): the committed trace of
`om1-gazebo-maze-003` (bench/golden) must replay byte-identically, gate green on
an unchanged config, and gate red with seam attribution on an injected decision
flip. This puts the determinism property test on a real robot episode, not just
the toy loop."""

import importlib.util
import sys
from pathlib import Path

from plumbline.cli import run_gate
from plumbline.core.interceptor import Context
from plumbline.core.seam import Seam
from plumbline.core.trace import canonicalize
from plumbline.proxy.recording import ReplayingProxy

_BENCH = Path(__file__).resolve().parent.parent / "bench"

# Content hash of the accepted behavior (§8.1). If this changes, the committed
# golden trace was modified — that must be a deliberate re-baselining, never an
# accident.
GOLDEN_VERSION = "abb676a6359eac92fecaa058f9ff8bda7f76387d996733998ee0438c9111e2f1"


def _gate_module() -> object:
    spec = importlib.util.spec_from_file_location("om1_gazebo_gate", _BENCH / "om1_gazebo_gate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["om1_gazebo_gate"] = module
    spec.loader.exec_module(module)
    return module


def test_golden_version_pins_the_committed_trace() -> None:
    module = _gate_module()
    assert module.build().golden.version() == GOLDEN_VERSION  # type: ignore[attr-defined]


def test_real_episode_replays_byte_identical() -> None:
    # The determinism property (§15, CI gate zero) on a REAL recorded episode:
    # every one of the 4,095 events serves back byte-identically by digest.
    module = _gate_module()
    spec = module.build()  # type: ignore[attr-defined]
    episode_id = module.EPISODE_ID  # type: ignore[attr-defined]
    episode = spec.store.load_episode(episode_id)
    assert len(episode.events) == 4095
    replay = ReplayingProxy(spec.store, episode_id)
    ctx = Context(episode_id=episode_id, model_id=None, params={}, logical_tick=0)
    assert all(
        canonicalize(replay.faithful(event.request, ctx)).digest
        == canonicalize(event.response).digest
        for event in episode.events
    )


def test_gate_green_on_unchanged_config() -> None:
    module = _gate_module()
    result = run_gate(module.build())  # type: ignore[attr-defined]
    assert result.passed


def test_gate_red_with_attribution_on_injected_decision_flip() -> None:
    module = _gate_module()
    result = run_gate(module.build_regressed())  # type: ignore[attr-defined]
    assert not result.passed
    episode = result.per_episode[0]
    assert episode.diverged and episode.divergence_seam is Seam.DECIDE_TO_ACT
    assert episode.drift > 0.9  # a full behavior inversion, not noise
