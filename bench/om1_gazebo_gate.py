"""CI gate over the REAL Gazebo golden episode (engineering spec §8.4, §10).

The golden set is `om1-gazebo-maze-003` — the showcase episode recorded from the
unmodified OM1 runtime driving a physics-simulated Go2 through a maze with real
lidar-conditioned decisions (docs/results-om1-gazebo.md), committed under
`bench/golden/` (4.2 MB; content addressing folds 3,789 Twist frames into 9 blobs).

    plumbline gate bench/om1_gazebo_gate.py

`build()` returns the UNCHANGED candidate config: the gate must PASS — green CI
means the accepted behavior of a real robot episode is reproduced. Its
counterpart `build_regressed()` (used by tests/test_golden_gazebo.py and
runnable the same way) injects an OFFLINE decision flip — every recorded
'move forwards' tool call becomes 'move back' — and the gate must FAIL with the
divergence attributed at DECIDE_TO_ACT. The offline flip keeps CI hermetic (no
model endpoint); the live-model version of the same regression is
examples/experiment_b_om1.py.
"""

from collections.abc import Callable
from pathlib import Path

from plumbline.adapters.action_matcher import recommended_behavior_matcher
from plumbline.adapters.om1 import OM1ActionSchema
from plumbline.core.replayer import DivergencePolicy
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload
from plumbline.regression import Config, FailurePolicy, GateSpec, GoldenSet

EPISODE_ID = "om1-gazebo-maze-003"
GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"


def _store() -> TraceStore:
    return TraceStore(root=GOLDEN_ROOT)


def _spec(store: TraceStore, config: Config) -> GateSpec:
    golden = GoldenSet(store)
    golden.add(EPISODE_ID, success=True)
    return GateSpec(
        store=store,
        golden=golden,
        config=config,
        drift_threshold=0.0,
        policy=FailurePolicy.ANY,
        behavior_matcher=recommended_behavior_matcher(OM1ActionSchema()),
    )


def build() -> GateSpec:
    """The unchanged candidate config — CI must stay green."""
    return _spec(
        _store(),
        Config(live_frontier={Seam.FUSE_TO_DECIDE}, overrides={}, matchers={}),
    )


def _flip(value: JSONValue) -> JSONValue:
    """Recursively flip 'move forwards' -> 'move back' inside tool-call arguments."""
    if isinstance(value, str):
        return value.replace("move forwards", "move back")
    if isinstance(value, list):
        return [_flip(item) for item in value]
    if isinstance(value, dict):
        return {key: _flip(item) for key, item in value.items()}
    return value


def _flipping_decider(store: TraceStore) -> Callable[[Payload], Payload]:
    """An OFFLINE injected regression: serve the recorded decision with every
    'move forwards' flipped to 'move back' — a behavior inversion with no model
    endpoint, keyed by the recorded request digest."""
    episode = store.load_episode(EPISODE_ID)
    flipped: dict[str, list[Payload]] = {}
    for event in episode.events:
        if event.seam is Seam.FUSE_TO_DECIDE:
            flipped.setdefault(event.request_digest, []).append(
                Payload(inline=_flip(event.response.inline), blobs=event.response.blobs)
            )
    cursor: dict[str, int] = {}

    def decide(request: Payload) -> Payload:
        from plumbline.core.trace import canonicalize

        digest = canonicalize(request).digest
        responses = flipped[digest]
        index = min(cursor.get(digest, 0), len(responses) - 1)
        cursor[digest] = index + 1
        return responses[index]

    return decide


def build_regressed() -> GateSpec:
    """The injected regression — the gate must FAIL with seam attribution."""
    store = _store()
    return _spec(
        store,
        Config(
            live_frontier={Seam.FUSE_TO_DECIDE},
            overrides={Seam.FUSE_TO_DECIDE: _flipping_decider(store)},
            matchers={},
            on_divergence=DivergencePolicy.HALT,
        ),
    )
