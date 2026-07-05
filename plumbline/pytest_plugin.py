"""``pytest-plumbline`` — record/replay and the robot-behavior gate as native pytest.

Auto-loaded via the ``pytest11`` entry point. Meet robotics tests where they live:

    # a mode-aware record/replay proxy — records live once, then replays in CI
    def test_my_loop(recorded_proxy):
        proxy = recorded_proxy("my-episode", upstream=my_model)
        run_my_runtime(proxy)          # proxy.forward(request, ctx) records or replays

    # the CI gate as an assertion (fails the build on drift, with seam attribution)
    @pytest.mark.plumbline_gate("bench/om1_gazebo_gate.py")
    def test_captioner_swap_does_not_drift(plumbline_gate):
        plumbline_gate.assert_no_drift()

Record mode is chosen with ``--plumbline-record={none,once,all}`` (or ``PLUMBLINE_RECORD``);
the default is ``none`` — a green CI run proves nothing hit a live model.
"""

import os
from collections.abc import Callable, Iterator, Mapping

import pytest

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload
from plumbline.proxy.recording import RecordingProxy, ReplayingProxy
from plumbline.record_mode import RecordMode, missing_trace_is_error, should_record
from plumbline.regression import GateResult

RecordedFactory = Callable[..., "RecordedProxy"]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("plumbline")
    group.addoption(
        "--plumbline-record",
        action="store",
        default=os.environ.get("PLUMBLINE_RECORD", "none"),
        choices=[m.value for m in RecordMode],
        help="record mode: none (replay only, CI default) | once (record if absent) | all",
    )
    group.addoption(
        "--plumbline-store",
        action="store",
        default=os.environ.get("PLUMBLINE_STORE", ".plumbline"),
        help="trace store directory for recorded/replayed episodes (default: .plumbline)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "plumbline_gate(config_path): run the Plumbline behavior gate on this config "
        "and fail the test on drift.",
    )


@pytest.fixture(scope="session")
def plumbline_record_mode(pytestconfig: pytest.Config) -> RecordMode:
    """The active record mode for the run (from --plumbline-record / PLUMBLINE_RECORD)."""
    return RecordMode.parse(str(pytestconfig.getoption("--plumbline-record")))


@pytest.fixture(scope="session")
def plumbline_store(pytestconfig: pytest.Config) -> TraceStore:
    """A TraceStore rooted at --plumbline-store (persists across the session)."""
    return TraceStore(root=str(pytestconfig.getoption("--plumbline-store")))


class RecordedProxy:
    """A mode-aware proxy: ``forward(request, ctx)`` records live or replays a stored
    trace depending on the active record mode — the test body is identical either way."""

    def __init__(self, proxy: RecordingProxy | ReplayingProxy, episode_id: str, recording: bool):
        self._proxy = proxy
        self.episode_id = episode_id
        self.recording = recording

    def forward(self, request: Payload, ctx: Context) -> Payload:
        if isinstance(self._proxy, RecordingProxy):
            return self._proxy.forward(request, ctx)
        return self._proxy.faithful(request, ctx)

    def _finish(self) -> None:
        if isinstance(self._proxy, RecordingProxy):
            self._proxy.close(self.episode_id)


@pytest.fixture
def recorded_proxy(
    plumbline_store: TraceStore, plumbline_record_mode: RecordMode
) -> "Iterator[Callable[..., RecordedProxy]]":
    """Factory: ``recorded_proxy(episode_id, upstream, *, metadata=None) -> RecordedProxy``.

    Records a live run or replays the stored trace per the active mode. In ``none``
    mode with no trace, the test fails with a hint to re-record — never a silent live
    call. Recording episodes are closed at teardown."""
    created: list[RecordedProxy] = []

    def make(
        episode_id: str,
        upstream: Callable[[Payload], Payload] | None = None,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> RecordedProxy:
        exists = episode_id in set(plumbline_store.list_episodes())
        if missing_trace_is_error(plumbline_record_mode, exists):
            pytest.fail(
                f"--plumbline-record=none but no recorded trace for episode {episode_id!r}; "
                "re-run with --plumbline-record=once to record it first"
            )
        if should_record(plumbline_record_mode, exists):
            if upstream is None:
                pytest.fail(
                    f"recording episode {episode_id!r} needs an `upstream` callable "
                    "(record mode is not 'none')"
                )
            proxy: RecordingProxy | ReplayingProxy = RecordingProxy(
                upstream=upstream,
                recorder=Recorder(plumbline_store, VirtualClock()),
                episode_metadata=dict(metadata or {}),
            )
            wrapper = RecordedProxy(proxy, episode_id, recording=True)
        else:
            wrapper = RecordedProxy(
                ReplayingProxy(store=plumbline_store, episode_id=episode_id),
                episode_id,
                recording=False,
            )
        created.append(wrapper)
        return wrapper

    yield make
    for wrapper in created:
        wrapper._finish()


class _GateAsserter:
    def __init__(self, marker_path: str | None):
        self._marker_path = marker_path

    def assert_no_drift(self, config_path: str | None = None) -> "GateResult":
        """Load a gate config (`build() -> GateSpec`) and fail the test on drift."""
        from plumbline.cli import format_report, load_gate_spec, run_gate

        path = config_path or self._marker_path
        if path is None:
            pytest.fail(
                "plumbline_gate.assert_no_drift needs a config path — pass one, or mark "
                "the test with @pytest.mark.plumbline_gate('path/to/gate.py')"
            )
        result = run_gate(load_gate_spec(str(path)))
        if not result.passed:
            pytest.fail(format_report(result), pytrace=False)
        return result


@pytest.fixture
def plumbline_gate(request: pytest.FixtureRequest) -> _GateAsserter:
    """Run the behavior gate as an assertion. Path comes from
    ``@pytest.mark.plumbline_gate('config.py')`` or ``assert_no_drift(path)``."""
    marker = request.node.get_closest_marker("plumbline_gate")
    marker_path = str(marker.args[0]) if marker and marker.args else None
    return _GateAsserter(marker_path)
