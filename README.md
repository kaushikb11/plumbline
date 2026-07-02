# Plumbline

**Record-replay and fusion-fidelity evaluation for language-bus robot runtimes.**

*A plumbline is a fixed reference you hang a structure against to detect drift, and the instrument you use to sound the depth of something you can't see into. Both meanings are load-bearing here.*

---

Robot runtimes like OpenMind's [OM1](https://github.com/OpenMind/OM1) turn multimodal sensor streams into natural-language captions, fuse them into a single prompt at roughly 1 Hz, and hand that prompt to a Cortex LLM that decides what to do. Every model in that loop — the VLM captioner, the ASR, the Cortex LLM — is a nondeterministic external dependency, usually a cloud API sampling at non-zero temperature. The consequence: you cannot reproduce a run, cannot regression-test a model or prompt change, and cannot measure how much task-relevant information survives the language bottleneck.

Plumbline is a standalone, runtime-agnostic library that fixes all three:

1. **Reproducibility** — a deterministic record-replay substrate that captures every nondeterministic model call at the four seams of the perception-to-action loop and replays them, making any language-bus runtime bit-reproducible despite nondeterministic models.
2. **Fidelity measurement** — metrics that quantify information loss across the *caption* and *fuse* boundaries, scored on downstream robot **decision success**, corrected for the decision-maker's own sampling noise.
3. **Regression testing** — a gate that catches silent behavior regressions when a model, prompt, or governance rule changes — drift that latency dashboards and text-level tracers cannot see.

OM1 is the flagship reference integration.

## The four seams

```
sensors ──▶ caption (VLM / ASR)  ──▶ fuse (captions + rules + RAG ──▶ one prompt)
        ──▶ decide (Cortex LLM ──▶ action plan) ──▶ act (orchestrator ──▶ HAL) ──▶ sensors …
```

| Seam | Captured request | Captured response | Interception |
|------|------------------|-------------------|--------------|
| `SENSOR_TO_CAPTION` | raw frame / audio / state | caption text | HTTP proxy |
| `CAPTION_TO_FUSE` | captions + rules + RAG | fused prompt | derived / bus tap |
| `FUSE_TO_DECIDE` | fused prompt | action plan | HTTP proxy |
| `DECIDE_TO_ACT` | action plan | HAL commands | Zenoh / ROS2 tap |

The bus is already text and already the architecture's narrow waist, so three of four seams come from a recording HTTP proxy and one from a bus tap. See the [engineering spec](spec/plumbline-engineering-spec.md) §3–§6 for the full contract.

## Status

Plumbline is built in vertical slices. What is implemented and tested today:

| Workstream | State |
|---|---|
| **WS1 Substrate** (`core/`) — seams, trace, virtual clock, recorder, replayer (faithful + counterfactual), matchers | ✅ implemented, `mypy --strict`, property-tested |
| **WS2 Trace + proxy** (`proxy/`) — recording/replaying proxy, OpenAI/Gemini/Anthropic normalizers, OTel-GenAI schema, content-addressed store, SSE capture, ASGI proxy server | ✅ implemented & tested. The proxy uses an **injected** async transport, with a bundled ASGI server (uvicorn-runnable); TLS termination is left to a front proxy |
| **WS3 Fidelity** (`fidelity/`) — decision distributions, the noise floor, caption/fusion loss, behavioral-equivalence judge | ✅ implemented & tested. The §14.5/§14.6 judgment calls (`render(G)`, `salient`) are **flagged for human review** |
| **Bench** (`bench/`) — captioner-for-decisions leaderboard (Experiment C), caption verbosity/fidelity curve (Experiment A), OpenAI-compatible client, scene authoring | ✅ implemented & tested; **demonstrated on real models** — see [Results](#results) |
| **WS5 adapters** (`adapters/`, `transport/`) — OM1 (proxy config, Zenoh tap, seam classification, counterfactual swap) **plus a generic OpenAI-agent-loop adapter** proving the frozen contract is runtime-agnostic (bus-less, derived action seam) | ✅ implemented & tested against *synthetic* episodes. A real Gazebo recording and sim ground-truth extraction are not yet done |
| **WS4 Gate** (`regression/`) — golden episodes, drift gate, `plumbline gate` CLI, GitHub Action | ✅ implemented & tested: the gate fails on an injected regression and passes on an unchanged config |
| **WS4 Observability** (`observability/`) — baseline-comparison monitors (Experiment B), trace-diff viewer | ✅ monitors + trace-diff implemented & tested; Grafana dashboards not yet |
| **CLI** (spec §11) | ✅ `record`, `replay`, `gate`, `diff`, `scenes` subcommands (record/replay run the proxy server; need uvicorn) |

The whole test suite (124 tests) is green under `mypy --strict`, `ruff` clean, with a dependency-free core. This honesty about what is and isn't built is the point: a tool that detects overclaiming should not overclaim.

## Results

Plumbline's fidelity metric, run **end-to-end on real models** (a real VLM + a real LLM via [Ollama](https://ollama.com), no robot, no simulator): two perception front-ends of the *same* vision model, ranked by downstream **decision** fidelity.

> A **narrow field of view** that can't see the floor drops the obstacle from its caption — and Plumbline charges it **2–3× higher `caption_loss` on exactly the obstacle scenes**, where the missing object flips the robot's decision from *stop* to *move*. The wide field of view wins (decision fidelity **0.814 vs 0.752**). A latency dashboard or text-quality tracer sees nothing wrong; Plumbline sees the decision break.

Full numbers, honest noise caveats, and the reproducible script are in **[docs/results-experiment-c.md](docs/results-experiment-c.md)** (`python examples/experiment_c.py`).

**Experiment A** (`python examples/experiment_a.py`) makes the same point *within* one caption: a surface text-similarity metric (`token_dice`) is **blind to which word carries the decision** — two captions degraded to *identical* surface similarity can have *opposite* decision fidelity. (The size of the gap depends on caption structure and the degradation knob, so it's a demonstration of the blindness, not a universal constant — see [`docs`/the module docstring](plumbline/bench/verbosity.py).)

## Install

```bash
pip install -e ".[dev]"   # Python ≥ 3.12
```

## Quickstart

The operator flow is: **point your runtime's base URLs at the proxy → record → replay → gate.** The full runnable walkthrough is in [docs/quickstart.md](docs/quickstart.md); the essentials:

### 1. Point your runtime at the proxy (zero source changes)

```python
from plumbline.adapters.om1 import OM1Adapter

cfg = OM1Adapter(proxy_base_url="http://localhost:8900").configure_proxy()
cfg.env
# {'OPENAI_BASE_URL': 'http://localhost:8900/v1',
#  'OPENAI_API_BASE': 'http://localhost:8900/v1',
#  'ANTHROPIC_BASE_URL': 'http://localhost:8900',
#  'GEMINI_API_BASE': 'http://localhost:8900',
#  'OLLAMA_HOST': 'http://localhost:8900', ...}
```

Export those env vars (or set the equivalent fields in OM1's `config/*.json5`) and the runtime's provider clients talk to Plumbline instead of the cloud. No OM1 source changes.

### 2. Record

In record mode the proxy forwards each model call to the real endpoint, captures and canonicalizes the request/response, infers the seam, emits a `SeamEvent`, and **returns the upstream response unaltered** (the zero-touch invariant). The action seam is captured by a passive Zenoh tap. Run it as a server:

```bash
plumbline record --upstream https://api.openai.com --store ./traces --episode go2-001
```

### 3. Replay

```python
from plumbline.core.replayer import Replayer, DivergencePolicy
from plumbline.core.seam import Seam

replayer = Replayer(store, clock, matchers)

# Faithful: serve every seam from the trace → bit-identical model I/O.
result = replayer.faithful("go2-gazebo-001")

# Counterfactual: swap the captioner; only that seam runs live, the rest is
# pinned to the trace. Halts and reports the seam + distance if the swap diverges
# enough that the recorded fused prompt no longer applies.
result = replayer.counterfactual(
    "go2-gazebo-001",
    live_frontier={Seam.SENSOR_TO_CAPTION},
    overrides={Seam.SENSOR_TO_CAPTION: new_captioner},
    on_divergence=DivergencePolicy.HALT,   # the default; divergence is a result, not an error
)
```

Or serve faithful replay so the runtime re-drives against recorded responses (no upstream): `plumbline replay --store ./traces --episode go2-001`.

### 4. Measure fidelity

```python
from plumbline.fidelity import caption_loss, fusion_loss, decision_stability

# How much does acting on the caption diverge from acting on ground truth,
# beyond the decision-maker's own noise? (render(G) is supplied by the sim — §14.5)
loss = caption_loss(decider, caption, oracle_context=render_G, n=64)
```

### 5. Gate, diff, author scenes

The regression gate counterfactual-replays each golden episode under a candidate config (a swapped model / edited prompt, expressed as seam overrides), computes behavioral drift from the accepted action sequence, and fails per policy — so CI catches a silent behavior regression a latency dashboard cannot (engineering spec §8):

```bash
plumbline gate path/to/gate_config.py                 # exits non-zero on drift; wrap in CI
plumbline diff EPISODE_A EPISODE_B --store ./traces   # where two runs diverged, and which seam
plumbline scenes ./images labels.json -o scenes.json  # author Experiment-C leaderboard input
```

A ready-to-run gate config lives in [`plumbline/bench/example_gate.py`](plumbline/bench/example_gate.py), and the shipped GitHub Action wraps `plumbline gate` for CI.

## Determinism envelope (read this)

Plumbline is precise about what it guarantees, because the project exists to catch tools that are not.

> **Plumbline guarantees that on replay every model call receives the recorded request and returns the recorded response, so the sequence of decisions and actions is reproduced. It does *not* control the runtime's wall-clock scheduler unless an adapter exposes a clock hook. We claim deterministic model-I/O replay, *not* deterministic wall-clock scheduling.**

The OM1 adapter's `clock_hook()` returns `None` today, so loop *timing* may vary across replays while model *I/O* does not. Nothing in this project — no log line, comment, or doc — should be read as claiming full scheduler or wall-clock determinism. This is engineering spec §3.4 / §14.4 and CLAUDE.md invariant 4; the full statement is in [docs/determinism-envelope.md](docs/determinism-envelope.md).

## Related work and the novel claim

Each capability Plumbline provides exists in an adjacent form for text agents or offline multimodal data. The contribution is the **specialization and integration**, not the invention of the primitives — so the neighbors are named explicitly.

**The one atomic novel claim:**

> Deterministic, seam-by-seam record-replay of the nondeterministic perception / language / decision model calls of an embodied language-bus robot runtime, with counterfactual single-component swap, halt-on-divergence, and downstream-decision scoring corrected by a decision-stability noise floor.

No single prior work crosses the modality boundary into an embodied LLM loop this way. The baselines it differentiates against:

- **ViSIL** (information-loss metric scored by downstream VQA via VLM inference; arXiv 2601.09851) — the closest metric precedent and the lineage Plumbline's fidelity metric sits in. **Differentiation:** closed-loop embodied *decision* success rather than passive VQA; the decision-stability noise floor; in-runtime measurement at the fuse seam. Cited as the primary metric baseline.
- **Trace-Based Assurance Framework** (Message-Action Traces, deterministic replay, fault injection, governance at the language-to-action boundary; arXiv 2603.18096) — the closest record-replay/governance precedent, but text/service-only: no perception, no modality boundary, no empirical study. **Differentiation:** the seams are `sensor→caption→fuse→decide→act` in an embodied runtime.
- **AgentRR** (generalized experience replay for text/GUI agents; arXiv 2505.17716) — explicitly *not* bit-perfect and *not* embodied. Its "check functions" are a cousin of Plumbline's input-consistency matchers. **Differentiation:** Plumbline does the faithful, bit-identical replay AgentRR declines.
- **Digital-twin execution tracing** (arXiv 2508.11406) — classical-robotics determinism (deterministic planners, fixed sim), *not* replay of nondeterministic cloud model calls. Does not touch the caption/fuse/LLM-decide loop.
- **Deterministic Simulation Testing** (FoundationDB, Antithesis) and generic deterministic LLM-agent harnesses — the acknowledged source of the virtual-clock and record-replay primitives. Borrowed and credited; the embodied/multimodal application is the new part.
- **Agent observability** (Langfuse, LangSmith, Phoenix, AgentOps, OpenTelemetry GenAI) — score text-task quality, latency, and cost; none scores embodied decision success or fusion fidelity. This is the corroboration for the headline demo: *"existing observability says fine, Plumbline says broken, the robot was in fact broken."*
- **LLM-regression-testing practice** (eval-gated CI, golden cases, median-of-N against judge noise, fail-on-drift) — established. What is new is the *target* (embodied robot decisions) and the determinism the replay substrate provides: you gate on *reproduced* behavior, not re-rolled samples (engineering spec §8.5).
- **VLA-FEB** (the VLA Fusion Evaluation Benchmark) — an *offline* benchmark that scores monolithic end-to-end VLA models on composite fusion dimensions (fusion efficiency, generalization, real-to-sim transfer, cross-modal alignment), linking architectural design variables to aggregate benchmark performance. **Low overlap, verified (§14.7 below):** different object (monolithic VLA policies, not a language-bus runtime with discrete model-call seams), offline and aggregate (not in-the-loop), scored at the model/representation level against benchmark performance — not per-decision success at the caption/fuse seam, corrected by a noise floor, with deterministic replay attribution.

### §14.7 verification note (VLA-FEB)

The VLA-FEB low-overlap verdict was checked against its primary source, not a summary, per engineering spec §14.7. Findings:

- VLA-FEB is proposed in Muhayyuddin et al., *"Multimodal Fusion with Vision-Language-Action Models for Robotic Manipulation: A Systematic Review"* (Information Fusion; [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253525011248), [project page](https://muhayyuddin.github.io/VLAs/)).
- ⚠️ It appears in the **published version and project page, not in the arXiv v1 preprint** ([2507.10672v1](https://arxiv.org/html/2507.10672v1)), which omits it entirely. Verify against the published article, not the preprint. (Search-engine summaries assert VLA-FEB's metrics confidently; the preprint does not contain them — exactly the failure mode §14.7 guards against.)
- The benchmark's *object* (offline, monolithic VLAs, design-variable-to-performance) is sufficient to establish low overlap. Its formal per-metric definitions of "fusion efficiency" and "cross-modal alignment" were **not fully retrievable** from accessible sources; if those turn out to be computed in-the-loop on a language-bus runtime with decision-success scoring (they are not, per the source's framing), this section would need revisiting before publication.

Citations follow engineering spec §12. Only the VLA-FEB verdict was independently re-checked this pass; **re-verify all arXiv identifiers immediately before any public release** (spec §7 risk note).

## Repository layout

```
plumbline/
  core/          # FROZEN interfaces: seam, trace, clock, recorder, replayer, matcher, store
  proxy/         # recording/replaying proxy, normalizers, OTel schema, SSE, ASGI server
  transport/     # zenoh tap + shim
  fidelity/      # decision distributions, noise floor, caption/fusion loss, judge
  regression/    # gate, drift, golden episodes
  adapters/      # adapter contract, OM1 adapter, recording-session coordinator
  bench/         # captioner leaderboard, OpenAI-compatible client, scene authoring
  observability/ # baseline-comparison monitors, trace-diff viewer   (Grafana: not yet)
  cli.py         # record / replay / gate / diff / scenes subcommands
examples/        # runnable Experiment-C demo (real models via Ollama)
tests/           # determinism, divergence, re-execution, matchers, proxy, fidelity, judge, gate, om1, cli, ...
spec/            # the two specs — source of truth
```

## License

Apache-2.0.

---

Sources for the §14.7 VLA-FEB check: [Systematic review (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1566253525011248) · [Project page](https://muhayyuddin.github.io/VLAs/) · [arXiv 2507.10672v1 preprint](https://arxiv.org/html/2507.10672v1)
