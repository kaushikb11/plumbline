# pytest-plumbline & record modes

Plumbline ships a pytest plugin (auto-loaded — no config) so record/replay and the
behavior gate are native pytest, the way `pytest-vcr` and `deepeval` are.

## Record modes

Every record/replay harness needs one decision per episode: record a live run, or
replay a stored trace? Plumbline names the policy (the VCR.py ergonomic):

| Mode | Behavior |
|---|---|
| `none` (default) | **Replay only.** A missing trace or an unrecorded live call is an error. A green CI run *proves* nothing hit a live model. |
| `once` | Replay if the episode's trace exists, else record it live once. |
| `all` | Always (re-)record live, overwriting the stored trace. |

Select it with `--plumbline-record=<mode>` (or the `PLUMBLINE_RECORD` env var). The
trace store is `--plumbline-store=<dir>` (or `PLUMBLINE_STORE`, default `.plumbline`).
The policy is also usable directly: `from plumbline.record_mode import RecordMode,
should_record, missing_trace_is_error`.

## `recorded_proxy` — a mode-aware record/replay proxy

```python
def test_my_loop(recorded_proxy):
    # records live under --plumbline-record=once, replays under =none (the CI default)
    proxy = recorded_proxy("my-episode", upstream=my_model)
    run_my_runtime(proxy)               # proxy.forward(request, ctx) records or replays
```

The same test body records and replays — the fixture picks based on the mode and
whether the episode's trace exists. In `none` mode with no trace, the test fails with
a re-record hint rather than silently going live. Recording episodes are closed at
teardown. Typical flow:

```bash
pytest --plumbline-record=once     # first run: records the loop
pytest                             # CI: replays it, no live calls (mode defaults to none)
```

## `plumbline_gate` — the behavior gate as an assertion

```python
import pytest

@pytest.mark.plumbline_gate("bench/om1_gazebo_gate.py")
def test_captioner_swap_does_not_drift(plumbline_gate):
    plumbline_gate.assert_no_drift()     # fails the build on drift, with seam attribution
```

`assert_no_drift(path)` also takes the config path directly. The config is a Python
file exposing `build() -> GateSpec` (same format as `plumbline gate <file>`); on drift
the test fails with the gate report (episode, drift, diverging seam).

## Loading a third-party adapter

The CLI resolves adapters three ways — built-in name, installed plugin, or a
`module:Class` path — and conformance-checks the result at load (a wrong adapter fails
here with a clear message, not later as a replay divergence):

```bash
plumbline record --adapter om1 …                          # built-in
plumbline record --adapter my_robot …                     # a plugin registered under
                                                          #   the 'plumbline.adapters' group
plumbline record --adapter mypkg.adapters:MyAdapter …     # a module:Class path
```

Register an adapter as a plugin in your `pyproject.toml`:

```toml
[project.entry-points."plumbline.adapters"]
my_robot = "mypkg.adapters:MyRobotAdapter"
```

The adapter must satisfy the unified 7-method contract — see
[writing-an-adapter.md](writing-an-adapter.md) and `plumbline.adapters.assert_conforms`.
