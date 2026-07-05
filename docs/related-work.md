# Related work and the novel claim

Each capability Plumbline provides exists in an adjacent form for text agents or offline
multimodal data. The contribution is the **specialization and integration**, not the
invention of the primitives — so the neighbors are named explicitly. (The Jan–Jul 2026
pre-release novelty re-check — multi-source, adversarially verified — is in
[related-work-audit-2026-07.md](related-work-audit-2026-07.md).)

## The one atomic novel claim

> Deterministic, seam-by-seam record-replay of the nondeterministic
> perception / language / decision model calls of an embodied language-bus robot
> runtime — **across the perception→language→decision modality boundary** — with
> counterfactual single-component swap, halt-on-divergence, and downstream-decision
> scoring corrected by a decision-stability noise floor.

The individual mechanisms are, by 2026, established for *text* agents (counterfactual
component swap: AgenTracer, CTA; decision record-replay: langchain-replay; replay-based
regression gating: the assurance framework below). Plumbline claims none of them in
isolation. **The novel part is the conjunction carried across the modality boundary** —
bit-identical replay of VLM *perception* + language *fuse* + LLM *decision* in one
embodied loop, with divergence halting and a noise-floor-corrected *physical-decision*
metric. No single prior work crosses that boundary into an embodied LLM loop.

## The baselines it differentiates against

- **ViSIL** (information-loss metric scored by downstream VQA via VLM inference; arXiv
  2601.09851) — the closest metric precedent and the lineage Plumbline's fidelity metric
  sits in. **Differentiation:** closed-loop embodied *decision* success rather than
  passive VQA; the decision-stability noise floor; in-runtime measurement at the fuse
  seam. Cited as the primary metric baseline.
- **Trace-Based Assurance Framework** (Message-Action Traces, deterministic replay, fault
  injection, governance at the language-to-action boundary; arXiv 2603.18096) — the
  closest record-replay/governance precedent, but text/service-only: no perception, no
  modality boundary, no empirical study. **Differentiation:** the seams are
  `sensor→caption→fuse→decide→act` in an embodied runtime.
- **AgentRR** (generalized experience replay for text/GUI agents; arXiv 2505.17716) —
  explicitly *not* bit-perfect and *not* embodied. Its "check functions" are a cousin of
  Plumbline's input-consistency matchers. **Differentiation:** Plumbline does the
  faithful, bit-identical replay AgentRR declines. (**PreAct**, arXiv 2606.17929 — GUI
  agents that compile runs into replayable state-machine programs with screen-state
  checks — is the same accelerate-by-replay family; same differentiation.)
- **DFAH — "Replayable Financial Agents"** (a Determinism-Faithfulness Assurance Harness
  for tool-using LLM agents; arXiv 2601.15322) — the closest *new* determinism-and-
  faithfulness neighbor: it measures trajectory/decision determinism + faithfulness over
  thousands of runs. **Differentiation:** DFAH *measures whether* text/financial agents
  are reproducible; Plumbline's substrate *enforces* reproducibility by record-replaying
  the model calls, and does it in an **embodied** perception→decision loop.
- **Caption Bottleneck Models (CaBM)** (interpretable classification with the information
  bottleneck moved entirely into natural-language captions; arXiv 2607.00578) —
  independent, concurrent validation of the *thesis* that a language bottleneck between
  perception and decision is a real, useful architectural boundary. **Differentiation:**
  CaBM is offline image *classification* (leakage-free interpretability); Plumbline scores
  information loss across that boundary on **embodied decision success**, corrected by a
  noise floor, with deterministic replay attribution — a measurement/reproducibility tool
  over an embodied loop, not a classifier.
- **Embodied runtime governance** (an external policy-checking layer that intercepts
  unauthorized embodied-agent actions; arXiv 2604.07833) — the assurance neighbor closest
  to the *embodiment*. **Differentiation:** governance/interception of live actions vs.
  Plumbline's *reproducibility and regression-gating* of decisions; complementary.
- **AgenTracer** (failure attribution for LLM multi-agent systems; surveyed in arXiv
  2606.04990) — **already does counterfactual replay + fault injection** to attribute
  failures, so counterfactual replay *per se* is not Plumbline's novelty.
  **Differentiation:** software multi-agent orchestration, not an embodied
  perception→decision loop; no modality boundary, no deterministic model-I/O replay, no
  physical-decision fidelity metric.
- **CTA — Counterfactual Trace Auditing** (arXiv 2605.11946) — evaluates a single-
  component swap by running the agent *live twice* and emitting post-hoc "divergence
  records" over aligned trace windows. **Differentiation:** it does *not* control decoder
  sampling (so it is not deterministic replay), and its divergence records are post-hoc
  annotations, not Plumbline's *halt*-on-divergence during pinned replay; text agents only.
- **langchain-replay** (Apache-2.0, [github.com/sixty-north/langchain-replay](https://github.com/sixty-north/langchain-replay))
  — records an LLM agent's decisions to JSONL (no pickle) and replays them while
  re-executing tools live, with a pytest plugin for CI. The closest OSS record-replay
  precedent. **Differentiation:** no counterfactual swap, no divergence detection, no
  fidelity/noise-floor metric, no perception/robot support — desktop LangChain agents.
- **Agent observability** (Langfuse, LangSmith, Phoenix, AgentOps, OpenTelemetry GenAI) —
  score text-task quality, latency, and cost; none scores embodied decision success or
  fusion fidelity. This is the corroboration for the headline demo: *existing
  observability says fine, Plumbline says broken, the robot was in fact broken.*
- **Digital-twin execution tracing** (arXiv 2508.11406) and classical Deterministic
  Simulation Testing (FoundationDB, Antithesis) — classical-robotics / systems
  determinism (deterministic planners, fixed sim), the acknowledged source of the
  virtual-clock and record-replay primitives. Borrowed and credited; the
  embodied/multimodal application of replaying *nondeterministic cloud model calls* is the
  new part.

## §14.7 verification note (VLA-FEB)

**VLA-FEB** (the VLA Fusion Evaluation Benchmark) is an *offline* benchmark that scores
monolithic end-to-end VLA models on composite fusion dimensions — low overlap, verified
against its primary source per engineering spec §14.7:

- Proposed in Muhayyuddin et al., *"Multimodal Fusion with Vision-Language-Action Models
  for Robotic Manipulation: A Systematic Review"* (Information Fusion;
  [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253525011248),
  [project page](https://muhayyuddin.github.io/VLAs/)).
- ⚠️ It appears in the **published version and project page, not in the arXiv v1
  preprint** ([2507.10672v1](https://arxiv.org/html/2507.10672v1)) — verify against the
  published article. (Search summaries assert VLA-FEB's metrics confidently; the preprint
  does not contain them — exactly the failure mode §14.7 guards against.)
- The benchmark's *object* (offline, monolithic VLAs, design-variable-to-performance) is
  sufficient to establish low overlap versus a language-bus runtime with per-decision
  success scoring at the caption/fuse seam.

Citations follow engineering spec §12. **Re-verify all arXiv identifiers immediately
before any public release** (spec §7 risk note).
