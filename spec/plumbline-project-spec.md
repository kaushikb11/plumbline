# Plumbline

**Record-replay and fusion-fidelity evaluation for language-bus robot runtimes.**

*Working name. A plumbline is a fixed reference you hang a structure against to detect drift, and the instrument you use to sound the depth of something you can't see into. Both meanings are load-bearing here.*

---

## 0. One-paragraph pitch

Robot runtimes like OpenMind's OM1 turn multimodal sensor streams into natural-language captions, fuse them into a single prompt at roughly 1 Hz, and hand that prompt to a Cortex LLM that decides what to do. Every model in that loop (the VLM captioner, the ASR, the Cortex LLM) is a nondeterministic external dependency, usually a cloud API sampling at non-zero temperature. The consequence is that you cannot reproduce a run, cannot regression-test a model or prompt change, and cannot measure how much task-relevant information survives the language bottleneck. Plumbline is a standalone library that fixes all three: a deterministic record-replay substrate that makes any language-bus runtime bit-reproducible despite nondeterministic models, a fidelity layer that quantifies information loss across the caption and fuse boundaries against downstream decision success, and a CI gate that catches silent behavior regressions when a model, prompt, or governance rule changes. It ships with OM1 as the flagship reference integration: record a real episode on a Unitree Go2, replay it identically, swap the captioner, and watch the regression surface in CI.

---

## 1. Why this, why now

### 1.1 The architecture creates the problem

OM1's design is a deliberate compression pipeline:

```
sensors -> caption (VLM/ASR/state) -> fuse (NL snippets + rules + RAG -> one prompt) -> decide (Cortex LLM) -> act (orchestrator -> HAL)
```

The founders' own paper ("A Paragraph Is All It Takes," arXiv 2412.18588) frames the narrow waist as a feature: the data fusion cycle runs at ~1 Hz and the central bus at roughly 40 bits/s, the rate of human language. That is an elegant bet. It is also a fidelity hazard, and the paper documents the hazard directly: a dog-persona robot fed raw LiDAR collision data turned *toward* obstacles instead of avoiding them, because the caption and fuse steps stripped the context the Cortex LLM needed. The fix was richer captioning ("there is a human 32 cm to your left, and the human looks scared of you"). That anecdote is the whole problem statement. Every modality is squeezed through a lossy text bottleneck, and nobody can currently say how lossy, for which tasks, or whether a model swap made it worse.

### 1.2 Three capabilities, and what already exists

The honest framing is not "nobody has done any of this." Each capability exists in an adjacent form for text agents or offline multimodal data. What does not exist is the three of them fused inside a running embodied language-bus runtime and scored on robot decision success. Plumbline's contribution is the specialization and the integration, not the invention of the primitives. Naming the neighbors is deliberate: it is what makes the differentiation credible to a reviewer who already knows them.

1. **Reproducibility.** You cannot replay an OM1 run bit-for-bit. The Cortex LLM samples at temperature; cloud endpoints are not seed-stable; wall-clock timing leaks into the loop. Same scene, different behavior. *Prior art:* deterministic record-replay for text LLM agents is a named, active area, including AgentRR (arXiv 2505.17716) and a trace-based assurance framework with deterministic replay, fault injection at the language-to-action boundary, and runtime governance (arXiv 2603.18096), plus commercial agent tracers (Langfuse, LangSmith, AgentOps) built on OpenTelemetry GenAI conventions. For robots specifically, digital-twin execution tracing replays belief state and sensor histories (arXiv 2508.11406). *The gap:* none of these spans the multimodal perception-to-action loop of an embodied runtime, and none replays the caption and fuse seams.
2. **Regression testing.** Swap the captioner from one VLM to another, or edit the system prompt, and there is no way to know whether perception or behavior silently degraded. OM1 ships Prometheus/Grafana latency dashboards, which tell you the loop got slower, never that the dog started walking into walls. *The gap:* existing agent observability gates on text-task success or latency, not on whether a perception swap inverted a robot's physical behavior.
3. **Fidelity measurement.** There is no in-the-loop metric for how much task-relevant information survives caption and fuse. *Prior art:* ViSIL (arXiv 2601.09851) is an information-theoretic information-loss metric for multimodal summarization scored via VLM inference and correlated to downstream VQA; the VLA-FEB benchmark proposes multimodal fusion-quality and alignment metrics; VLM perception benchmarks exist (PhysBench, EPOS-VLM, HomeSafeBench). *The gap:* all of these score offline artifacts or policies in isolation. None measures fusion fidelity inside a running runtime, scored on downstream robot decision success, with deterministic replay to attribute a behavior change to a specific seam. That last clause is the whole project.

### 1.3 The insight that makes it tractable

You do not make the models deterministic. That is impossible for a cloud LLM at temperature. You make the *harness* deterministic by capturing every external model call and replaying it. This is deterministic simulation testing (the FoundationDB and Antithesis playbook), and record-replay for text agents already uses it; Plumbline carries it across the modality boundary into an embodied multimodal loop, where the captured calls include VLM perception and the replay must hold the caption and fuse seams fixed. The determinism lives in the substrate, not the model. Record once, replay forever, and the only thing that changes between replays is the component you deliberately swapped.

### 1.4 Why it is a strong artifact for OpenMind specifically

OM1's value proposition is upgradability (swap frozen models freely), observability (a human can read the language bus), and durable alignment (rules in plain English). All three are unfalsifiable without exactly this tooling. You cannot claim safe upgradability if you cannot detect that an upgrade broke obstacle avoidance. Plumbline turns OM1's core promises into things you can measure and gate on. It is built from VLM-pipeline, multimodal-fusion, agentic-orchestration, and eval-harness work, which is the actual job, and it deliberately does not pretend OM1 trains models, because it does not.

---

## 2. Scope and non-goals

**In scope.**
- Deterministic record-replay of the full perception-to-action loop for language-bus runtimes.
- A trace format aligned to OpenTelemetry GenAI semantic conventions.
- Fidelity metrics for the caption and fuse boundaries, scored on downstream decision success.
- A regression gate runnable in CI.
- A working OM1 reference integration (record and replay on Go2 in Gazebo, real Go2 if hardware is available).
- Simulation-based episode generation so the benchmark runs with no hardware.

**Explicit non-goals.**
- No training, fine-tuning, distillation, or distributed training infrastructure. OM1's brain is remote inference; training tooling is off-thesis and off-core.
- No new VLA policy or foundation model. Plumbline evaluates and reproduces, it does not learn.
- No high-frequency motor control loop. OM1's semantic loop is ~1 Hz; real-time chunking for 30 to 50 Hz actuation is a different regime and a separate (optional) workstream, not the spine.
- Not a fork of OM1. Standalone library plus a thin adapter. Generality is the contribution; OM1 is the flagship reference.

---

## 3. Architecture

Three layers, clean interfaces between them, each independently useful.

### 3.1 Layer 1 — Deterministic record-replay substrate

The substrate sits at the **interception boundary**: every nondeterministic external call the runtime makes. For a language-bus runtime there are exactly four seams, and they map onto the bus itself, which is the architecture's narrow waist and conveniently already text:

| Seam | Captured request | Captured response |
|------|------------------|-------------------|
| sensor to caption | raw frame / audio / state | caption text |
| caption to fuse | set of captions + rules + RAG | fused prompt |
| fuse to decide | fused prompt | LLM action plan |
| decide to act | action plan | HAL commands issued |

Two modes:

- **RECORD.** The runtime runs normally. Every call at every seam is intercepted and written to the trace store as `(request, response, model_id, params, wall_latency, logical_tick)`. A **virtual clock** records logical ticks so that on replay, timing is reconstructed from the trace rather than from wall-clock, which removes a major source of behavioral nondeterminism (a slow cloud call changing what the next tick sees).
- **REPLAY.** External calls are *not* made by default. Each seam's response is served from the trace, keyed by request identity. Two replay sub-modes:
  - *Faithful replay.* Serve the exact recorded response at every seam. Result is bit-identical behavior. This is the reproducibility guarantee and the regression baseline.
  - *Counterfactual replay.* Swap one component (say the captioner model) and let only that seam re-execute live, feeding its new output downstream while everything upstream of the swap is served from the trace. This isolates the effect of a single change, which is what makes regression attribution possible. The hard part is what happens downstream of the swap, addressed below.

#### 3.1.1 Counterfactual replay: the divergence problem and how it is handled

This is the keystone mechanism and the part most likely to be wrong if it is hand-waved, so it is specified rather than gestured at. Naive counterfactual replay assumes that swapping a component changes only that component's output and that everything downstream can still be served from the trace. That assumption breaks the moment the swapped output changes the *shape* of what follows. A new captioner may emit a different number of captions, different timing, or a caption that causes the Fuser to build a structurally different prompt, at which point "serve the fuse seam from the trace" is meaningless because the recorded fuse response no longer corresponds to the live input.

Plumbline borrows the discipline that text-agent replay frameworks (AgentRR, the assurance-framework line of work) converged on, and makes it explicit for the multimodal loop:

- **Seam classification per run.** Before a counterfactual run, every seam downstream of the swap point is classified as *replayable* (its recorded input still matches the live input within tolerance) or *invalidated* (the live input diverged). Replayable seams serve from trace; invalidated seams must either re-execute live or halt the run, by policy.
- **Input-consistency validation.** At each downstream seam, the live request is compared against the recorded request by a seam-specific matcher (exact match for structured fields, embedding-distance threshold for free-text captions and prompts). A match within tolerance means the trace response is still valid. A mismatch means divergence.
- **Halt-on-divergence by default.** When a seam diverges and the run is not explicitly configured to re-execute it live, the replayer raises rather than silently serving a stale response. Silent continuation is the failure mode that makes a replay tool produce confident garbage; the assurance-framework prior art calls this "failure on exhaustion," and Plumbline adopts it. A diverged run is a *result*, not an error to suppress: "this swap changed behavior so much the trace no longer applies" is itself a regression signal.
- **Cascade control.** A counterfactual run declares its *live frontier*: the set of seams permitted to re-execute. Three useful settings. *Isolated:* only the swapped seam runs live, everything else must match the trace or halt. This measures the swap's direct effect with maximum attribution. *Downstream-live:* the swapped seam and everything after it run live, only upstream is pinned to the trace. This measures end-to-end behavioral effect but attributes less precisely. *Pinned-decision:* re-run perception live but pin the Cortex LLM to its recorded outputs where inputs still match, isolating perception changes from decision noise.
- **Divergence as a reported metric.** Every counterfactual run reports where its live frontier first diverged from the trace and by how much, per seam. This is not bookkeeping; it is the attribution result. "The captioner swap diverged at the fuse seam in 40% of episodes, and in those episodes the decision flipped" is exactly the finding the project exists to produce.

The consequence for scope: counterfactual replay is reliable when the swapped component is downstream-pure (its change does not restructure upstream context), and explicitly bounded when it is not. Plumbline does not pretend to replay across arbitrary structural divergence; it detects divergence, reports it, and refuses to fabricate. That refusal is a feature.

The substrate is implemented as a set of interceptors with a stable interface, plus a recorder/replayer and the virtual clock. It knows nothing about OM1 specifically; runtimes plug in via adapters (Section 3.4).

### 3.2 Layer 2 — Fidelity metrics

This is the novel and hardest layer, and the one that carries the scientific contribution. The wrong move is to score captions on surface similarity (BLEU against a reference caption). A caption can be fluent and accurate and still drop the one fact the decision needed, or be clumsy and still preserve it. The right unit of measurement is **downstream decision success**: does acting on this caption produce the behavior that ground-truth scene state would have produced?

Definitions:

- **Caption fidelity.** Given raw sensor input with known ground-truth scene state (available in sim), does a decision derived from the caption match the decision derived from ground truth? Loss is the gap. This catches the LiDAR-dog failure as a number.
- **Fusion fidelity.** Information present in an individual caption that does not survive into the fused prompt, weighted by whether that information was task-relevant. Fusion loss is distinct from caption loss and the two are separately attributable because the substrate captures both seams.
- **Decision stability.** The decision-maker is an LLM at temperature, so "the decision" is a distribution, not a point, and fidelity is therefore measured through a noisy instrument. Decision stability is the variance of the decision under fixed inputs across repeated samples. It is not itself a fidelity metric; it is the noise floor that every fidelity number must be reported against. A caption-fidelity gap is only real if it exceeds the stability floor for that scene. Concretely, fidelity is scored over decision *distributions* (sample the decision-maker N times on caption-derived input and on ground-truth-derived input, compare the distributions) with the stability floor subtracted, not over single decisions. This is the answer to the inevitable "your metric is just sampling noise" objection, and it belongs in the metric definition, not in a footnote.

Scoring machinery: ground-truth comparison in simulation where scene state is known, plus LLM-as-judge for behavioral equivalence where ground truth is unavailable, with the judge itself recorded and replayable so the eval is reproducible.

A scope honesty note, stated here rather than buried: caption fidelity requires known ground-truth scene state, which means simulation. The headline fidelity-versus-bandwidth result (Experiment A) is therefore a *simulation* result, and the spec says so plainly rather than letting a reviewer discover it. Two things make this acceptable. First, OM1 now supports Isaac Sim for both Go2 and G1 natively (physics-accurate, GPU-accelerated), not just Gazebo, so the sim is a credible stand-in for the target hardware rather than a toy. Second, the regression and leaderboard results (Experiments B and C) only need A-versus-B comparison, not absolute ground truth, so they run on real-robot recordings. The fidelity curve is sim-bound by nature; the behavioral findings are not. Claiming otherwise would be the kind of overreach this project is explicitly built to detect in others.

### 3.3 Layer 3 — Regression gate and observability

- **Golden episodes.** A curated, versioned set of recorded episodes with known-good behavior.
- **Gate.** On a config change (new model, new prompt, edited rule), counterfactual-replay the golden episodes and compute behavior drift and fidelity deltas. Fail if drift exceeds threshold. This is the "CI for robot behavior" deliverable, runnable as a GitHub Action.
- **Observability.** Grafana panels for fidelity over time and per-component attribution, complementing OM1's existing latency stack rather than replacing it. A trace-diff viewer that shows, side by side, where two runs of the same episode diverged and which seam introduced the divergence.

### 3.4 Layer 4 — Adapters (OM1 is the flagship)

An adapter wires a specific runtime's seams into the substrate's interceptor interface. The OM1 adapter is a headline deliverable, not an afterthought, because with the PR track removed it is the thing that proves the library understands OM1's real runtime rather than an abstract notion of one.

OM1 integration has two mechanisms, and the adapter uses both:

- **Zenoh tap (observation).** OM1's recommended middleware is Zenoh and the bus is already on it. Subscribe to the bus topics and passively record everything. Low-invasiveness, gets RECORD for free, no changes to OM1 internals.
- **Provider shim (replay injection).** To replay, the adapter shims OM1's model-provider clients (the OpenAI/Gemini/DeepSeek/Anthropic/Ollama call sites) so that in REPLAY mode the call returns a trace response instead of hitting the network. This is where the determinism guarantee is enforced.

The adapter targets the documented language-bus contract and the provider interface, not fragile internal structs, so it survives OM1's beta churn. As OM1's Go runtime matures its multimodal vision inputs, the adapter records what exists at each point and is structured to absorb the vision seam as it firms up.

---

## 4. Headline experiments

Three results, each requiring the full substrate, each publishable and demo-able.

### Experiment A — The fidelity-versus-bandwidth curve
Vary caption verbosity and fusion frequency. Plot downstream task success against effective bus bandwidth. Find the knee where adding words stops improving decisions. This converts the founders' 40 bits/s design intuition and the LiDAR-dog anecdote into a measured curve, and it is the kind of result that gets cited. The headline figure of the whole project.

### Experiment B — Silent-regression detection, against a real baseline
Record golden episodes with captioner A. Counterfactual-replay with captioner B. The result only lands if it beats what a team would otherwise use, so the experiment is run head-to-head against two baselines that stay green while Plumbline goes red: (1) OM1's own Prometheus/Grafana latency stack, which reports the loop as healthy because latency is unchanged, and (2) a generic agent tracer (Langfuse- or OpenTelemetry-GenAI-style) configured on the same run, which sees well-formed LLM calls and flags nothing because the text-level outputs are plausible. Plumbline catches the behavior inversion (obstacle avoidance flips to obstacle seeking, or a guardrail stops firing) because it scores the physical decision, not the latency or the token stream. "Existing observability says fine, Plumbline says broken, the robot was in fact broken" is the demo that sells the project, and naming the baselines it beats is what makes it a demo rather than an assertion.

### Experiment C — Caption-quality-for-decisions leaderboard
Rank candidate VLMs as OM1 captioners by downstream decision success, not by caption surface quality. Demonstrate that the best caption by NLP metrics is not the best caption for behavior. Practical value to anyone choosing a captioner, and a direct shot at the assumption that perception quality and language quality are the same thing.

---

## 5. Workstreams

Scoped for a team of six to eight engineers over roughly four months. The honest justification for that headcount is not the substrate code, which a small number of strong systems engineers could carry; it is two things that genuinely do not compress. First, the **golden-episode dataset**: curating, annotating, and ground-truth-labeling a benchmark of robot episodes across multiple tasks and multiple embodiments is slow, manual, and the part most often underestimated, and the entire fidelity and regression story rests on its quality. Second, **cross-embodiment breadth**: making the OM1 adapter and the benchmark work across a quadruped (Go2), a humanoid (G1), and both Gazebo and Isaac Sim, in sim and on real hardware, multiplies the integration and validation surface. Those two are where the engineers go. Interfaces between workstreams are the layer boundaries in Section 3, so the code workstreams parallelize after the substrate interface is frozen in week 2.

**WS1 — Deterministic substrate (2 engineers, systems-heavy).**
Interceptor interface, recorder, replayer, virtual clock, faithful and counterfactual replay modes. Owns the reproducibility guarantee. Critical path for everyone else.

**WS2 — Trace format and store (1 to 2 engineers).**
OpenTelemetry GenAI-aligned span schema, episode-as-trace model, content-addressed store, safe serialization. No pickle, ever; the LeRobot async-inference RCE (CVE-2026-25874, CVSS 9.3, unauthenticated pickle deserialization over gRPC) is the cautionary tale and a talking point. Tensors via safetensors, structured data via protobuf or msgpack.

**WS3 — Fidelity metrics and benchmark (2 engineers, ML/eval-heavy).**
Caption fidelity, fusion fidelity, decision stability, the golden-episode dataset, the LLM-as-judge equivalence scorer. The scientific core and the part most aligned with VLM/multimodal/eval strength.

**WS4 — Regression gate and observability (1 engineer).**
CI action, drift thresholds, Grafana panels, trace-diff viewer. Ties into OM1's nightly CI and metrics package.

**WS5 — OM1 adapter and simulation (1 to 2 engineers).**
Zenoh tap, provider shim, episode generation in both Gazebo (Go2) and Isaac Sim (Go2 and G1, both supported in OM1 today), the end-to-end demo on quadruped and humanoid. Owns the flagship integration that carries the application signal and the cross-embodiment breadth that justifies the headcount.

---

## 6. Roadmap

Phased so a working OM1 demo exists early, not in the final week. No maintainer-goodwill phase, since the PR track is cut; week one goes straight to the substrate.

- **Weeks 1 to 2. Substrate interface frozen.** Interceptor interface, virtual clock, trace schema draft. Goal: faithful-replay a recorded toy language-agent episode bit-for-bit.
- **Weeks 3 to 4. OM1 adapter, early.** Zenoh tap recording a real OM1 episode in Gazebo. This validates every downstream design against a real runtime instead of a mock, which is why it comes now and not at the end.
- **Weeks 5 to 8. Replay and provider shim.** Counterfactual replay working through the OM1 adapter. Goal: swap a model and re-run a single seam live while the rest serves from trace.
- **Weeks 9 to 12. Fidelity layer.** Experiments A and C. Publish the bandwidth curve and the captioner leaderboard.
- **Weeks 13 to 16. Regression gate.** Experiment B. CI action catching an injected regression on golden episodes. Observability panels and trace-diff viewer.
- **Weeks 16+. Optional expansions.** Deeper multimodal vision-seam coverage as OM1's Go runtime matures it; edge async-inference adapter against the 1 Hz staleness problem; multi-robot coordination eval (the FABRIC-thesis extension), which the record-replay substrate is a prerequisite for anyway.

---

## 7. Risks and mitigations

- **LLM nondeterminism even at fixed seed.** Do not fight it. Capture-replay sidesteps it entirely for reproducibility; for live counterfactual runs, decision-stability measurement bounds the noise floor so real regressions are distinguishable from sampling, and fidelity is scored over decision distributions with that floor subtracted (Section 3.2).
- **Defining "fidelity" defensibly.** The whole project is vulnerable to "your metric is arbitrary." Anchor every metric to downstream decision success, never to caption surface similarity, and report every fidelity number against the decision-stability floor. Make the metric definition itself a reviewed design doc, and read VLA-FEB and ViSIL closely first so the metric is positioned against existing fusion-quality and information-loss work rather than reinventing it.
- **Prior art and differentiation.** Record-replay and information-loss metrics exist for text agents and offline multimodal data (AgentRR, the assurance-framework line, ViSIL, VLA-FEB). The risk is a reviewer concluding Plumbline is derivative. Mitigation is to state the differentiation on the page, not defend it after the fact: the contribution is the embodied, in-the-loop, decision-success-scored, replay-attributable integration, and the spec cites its neighbors explicitly to make that line credible.
- **OM1 beta churn.** The Go runtime is at beta and parts of the perception path are still maturing. The adapter targets the documented bus contract and provider interface, not internal structs, and the standalone-with-adapter posture means the library has value even if OM1's internals move. Re-verify the repo state immediately before any public release, since OM1 is moving fast enough that capabilities (Isaac Sim support, for instance) have shipped between research passes.
- **Sim ground truth availability.** Caption fidelity needs known scene state, which means sim, so the headline fidelity curve is a sim result and the spec says so. Isaac Sim support for Go2 and G1 in OM1 makes the sim credible; regression and leaderboard results run on real-robot recordings, which need only A-versus-B comparison.
- **Judge reproducibility.** An LLM judge is itself nondeterministic. Record and replay the judge through the same substrate so the eval is as reproducible as the thing it evaluates.

---

## 8. What this signals

A standalone library that is genuinely useful to anyone building a language-bus robot stack (the field contribution), with OM1 as a working, demoed, end-to-end reference integration (the application signal). It is built from exactly the skills the role wants: production VLM pipelines, multimodal fusion, agentic orchestration, and the discipline to build eval and reproducibility infrastructure that a small fast-moving team has not had time to build for itself. It does not depend on anyone's review queue, and a public benchmark plus a working integration is a stronger artifact than a handful of merged fixes would have been.