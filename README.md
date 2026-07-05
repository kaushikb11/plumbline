# Plumbline

**Record-replay and fusion-fidelity evaluation for language-bus robot runtimes.**

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue) ![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green) ![Typed: mypy strict](https://img.shields.io/badge/mypy-strict-blue) ![Status: alpha](https://img.shields.io/badge/status-alpha-orange)

Robot runtimes like OpenMind's [OM1](https://github.com/OpenMind/OM1) turn sensor streams into text captions, fuse them into a single prompt, and hand it to a cloud LLM that decides what the robot does. Every model in that loop is nondeterministic — so you **can't reproduce a run, can't regression-test a prompt or model change, and can't measure how much task-relevant information survives the language bottleneck.**

Plumbline fixes all three by recording and replaying the model calls at the four seams of the perception-to-action loop:

- **Reproducibility** — capture the nondeterministic model calls and replay them, so a runtime re-drives its exact decision/action sequence offline.
- **Regression testing** — a CI gate that counterfactually replays a golden run under a candidate change and fails on *behavioral* drift, not sampling noise.
- **Fidelity measurement** — metrics that score how much decision-relevant information survives the caption/fuse bottleneck, corrected for the model's own noise.

> **The pitch in one line:** you append one innocuous governance rule to a real OM1 episode, and every decision flips from *move forward* to *move back*. OM1's latency dashboard and a generic tracer stay green. **Plumbline goes red — and attributes it to the `DECIDE_TO_ACT` seam.** [See the result →](docs/results-experiment-b-om1.md)

> *A plumbline is a fixed reference you hang a structure against to detect drift, and the instrument you drop to sound the depth of something you can't see into. Both meanings are load-bearing here.*

## Quickstart

```bash
pip install -e .              # core: dependency-free — loads/replays traces, runs the gate
pip install -e '.[proxy]'     # + the record/replay HTTP server (httpx, uvicorn)
```

Plumbline is not on PyPI yet — install from a clone. Then run something real in one command:

```bash
plumbline gate bench/om1_gazebo_gate.py     # gate a real OM1+Gazebo golden (4,095 events) — must replay byte-identically
python examples/toy_loop.py                 # record → faithful-replay a toy loop in 40 lines, zero deps
```

The full record → replay → gate → measure walkthrough is in **[docs/quickstart.md](docs/quickstart.md)**; the mental model is **[docs/concepts.md](docs/concepts.md)**.

## The four seams

Plumbline taps the loop where it is already natural language — the architecture's narrow waist:

```
sensors ─▶ caption (VLM/ASR) ─▶ fuse (captions+rules+RAG ─▶ one prompt)
        ─▶ decide (Cortex LLM ─▶ action plan) ─▶ act (orchestrator ─▶ HAL) ─▶ sensors …
```

| Seam | Request → Response | Interception |
|------|--------------------|--------------|
| `SENSOR_TO_CAPTION` | frame / audio / state → caption text | recording HTTP proxy |
| `CAPTION_TO_FUSE` | captions + rules + RAG → fused prompt | derived / bus tap |
| `FUSE_TO_DECIDE` | fused prompt → action plan | recording HTTP proxy |
| `DECIDE_TO_ACT` | action plan → HAL commands | Zenoh / ROS2 tap |

Three of four seams are HTTP calls to model endpoints, so a recording proxy captures them with **zero source changes** — you point the runtime's base URL at Plumbline. One is a robotics bus, caught by a tap.

## How you use it

Point your runtime's model base URL at the proxy (no code changes), then:

```bash
plumbline record --upstream https://api.openai.com --store ./traces --episode go2-001  # capture a golden run
plumbline replay --store ./traces --episode go2-001                                    # re-drive it, offline, deterministic
plumbline gate   my_gate.py                                                             # fail CI on behavioral drift
plumbline diff   go2-001 go2-002 --store ./traces                                       # where two runs diverged, at which seam
```

The gate config is a small Python file exposing `build() -> GateSpec` (the golden episodes + the change you're testing, as seam overrides). It counterfactually replays each golden under the candidate and **halts on the first divergence**, reporting the seam and distance — so CI catches a silent behavior regression a latency dashboard cannot.

Prefer pytest? The plugin ships `recorded_proxy` (record/replay fixture with VCR-style `--plumbline-record=none/once/all` modes) and `plumbline_gate` (the gate as an assertion) — **[docs/pytest-plugin.md](docs/pytest-plugin.md)**.

Measuring fidelity is the Python API:

```python
from plumbline.fidelity import caption_loss
# how much acting on the caption diverges from acting on ground truth, beyond the decider's own noise
loss = caption_loss(decider, caption="the path looks clear", oracle_context="obstacle at 0.3m", n=16)
```

## Proven on a real robot loop

The showcase episode (`om1-gazebo-maze-003`) is the **unmodified** OM1 runtime driving a physics-simulated Go2 quadruped through a maze on Modal — simulated lidar → fused "safe directions" prompt → live cloud Cortex → gait control → physics, all genuine and headless. **8.37 m walked, 153 decisions tracking 15 lidar-derived perception states, byte-identical replay over 4,095 events (verified cross-machine).** The trajectory's right-turn arc sits exactly where "turn left" dropped out of the recorded prompts.

- [results-om1-gazebo.md](docs/results-om1-gazebo.md) — the showcase, in depth.
- [results-experiment-b-om1.md](docs/results-experiment-b-om1.md) — silent-regression detection: baselines green, Plumbline red.
- [results-experiment-c.md](docs/results-experiment-c.md) — captioner fidelity on real models (a narrow field of view that drops the obstacle scores 2–3× higher `caption_loss` on exactly the scenes where it flips the decision).

## What Plumbline guarantees (and what it does not)

Plumbline is precise about its envelope, because it exists to catch tools that are not:

> On replay, every model call receives the recorded request and returns the recorded response, so the decision/action sequence is reproduced. Plumbline claims **deterministic model-I/O replay — not** deterministic wall-clock scheduling. It does not control the runtime's scheduler unless an adapter exposes a clock hook.

Full statement in [docs/determinism-envelope.md](docs/determinism-envelope.md). No log line, comment, or doc in this project should be read as claiming full scheduler determinism.

## Documentation

| Start here | |
|---|---|
| [concepts.md](docs/concepts.md) | The mental model: the four seams and the record → replay → counterfactual → gate lifecycle. |
| [quickstart.md](docs/quickstart.md) | Point at the proxy → record → replay → measure fidelity, end to end. |
| [api.md](docs/api.md) | The frozen `core/` contract + the public `fidelity` / `proxy` / `regression` / `adapters` surfaces. |
| [writing-an-adapter.md](docs/writing-an-adapter.md) | Teach Plumbline a new runtime — the 7-method `Adapter` contract. |
| [pytest-plugin.md](docs/pytest-plugin.md) | Record/replay and the gate as native pytest, with record modes. |
| [faq.md](docs/faq.md) · [stability.md](docs/stability.md) · [limitations.md](docs/limitations.md) | Troubleshooting · the 0.x API policy · the honest scope audit. |

The two files in [`spec/`](spec) are the source of truth for the *why* and the *how*. The novelty position and its neighbors (this is a specialization-and-integration contribution, not new primitives) are in [docs/related-work.md](docs/related-work.md).

## Status

Plumbline is **alpha, built in vertical slices**, and after a four-dimension adversarial production review it is **ready for early OM1 adopters**: the record/replay server, live recorder, security, and observability are hardened and tested; a real wheel installs clean. The stable contract is `plumbline.core` + the flat `from plumbline import …` re-exports + the trace format (see [stability.md](docs/stability.md)); the fidelity/regression *math* is experimental. Read [limitations.md](docs/limitations.md) before assuming a headline capability — a tool built to detect overclaiming should not overclaim.

## Repository layout

```
plumbline/
  core/          # FROZEN interfaces: seam, trace, clock, recorder, replayer, matcher, store
  proxy/         # recording/replaying proxy, provider normalizers, OTel schema, SSE, ASGI server
  transport/     # zenoh tap
  fidelity/      # decision distributions, noise floor, caption/fusion loss, judge
  regression/    # gate, drift, golden episodes
  adapters/      # adapter contract, OM1 / G1 / generic, recording-session coordinator
  bench/         # captioner leaderboard, OpenAI-compatible client, scene authoring
  observability/ # baseline monitors, trace-diff, Grafana dashboards + OTLP/feed export
  cli.py         # record / replay / gate / list / diff / export / scenes
examples/        # runnable demos + the zero-dep toy_loop
tests/           # determinism, divergence, re-execution, matchers, proxy, fidelity, gate, adapters, …
```

## Contributing & license

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the six hard invariants (frozen core, determinism gate, no pickle, model-I/O-only determinism, halt-on-divergence, vertical slices) that keep the substrate trustworthy. Licensed under **Apache-2.0**.
