# Plumbline: Full Engineering Specification

**Record-replay and fusion-fidelity evaluation for language-bus robot runtimes.**

Version 0.1 (build spec). This is the engineering document one level below the project spec. It defines the abstractions, interfaces, trace format, metric mathematics, adapter contract, repository layout, and test strategy in enough detail to start writing code and to parallelize across a team. It is opinionated on purpose; every open decision is collected in Section 14 rather than left implicit.

A note on what changed from the project spec. The earlier spec described interception as "shim OM1's provider clients." This spec replaces that with a **recording HTTP proxy** as the primary interception mechanism, because it is runtime-agnostic and requires zero source changes to the target runtime. That single decision is what lets Plumbline be a general tool with OM1 as one adapter, rather than an OM1 patch. The reasoning is in Section 4.2.

---

## 1. Problem and design goals

### 1.1 The problem, stated for a builder

A language-bus robot runtime (OM1 is the reference) runs this loop:

```
sensors --> caption (VLM / ASR / state)  --> fuse (captions + rules + RAG --> one prompt)
        --> decide (Cortex LLM --> action plan) --> act (orchestrator --> HAL) --> sensors ...
```

Every box that says "VLM," "ASR," or "LLM" is a call to a nondeterministic model, usually a cloud API sampling at temperature above zero. Three consequences follow, and Plumbline targets all three:

1. **You cannot reproduce a run.** Same scene, different model samples, different behavior. Debugging is archaeology.
2. **You cannot regression-test a change.** Swap the captioner, edit the system prompt, change a governance rule, and nothing tells you whether the robot's physical behavior silently changed.
3. **You cannot measure what the language bottleneck costs.** OM1 compresses rich sensor data to roughly 40 bits per second of natural language. Nobody can say how much task-relevant information survives, for which tasks, or whether a model swap made it worse.

### 1.2 Design goals

- **Runtime-agnostic core, runtime-specific adapters.** The core knows about seams, traces, and model calls. It does not know about OM1. OM1 is the first adapter and the flagship demo.
- **Zero-touch interception where possible.** Recording a run should require configuration, not source edits to the runtime.
- **Honest determinism.** Be precise about what is guaranteed bit-identical (model input/output) and what is not (the runtime's internal scheduler, unless it exposes a clock hook). Never claim more than the mechanism delivers.
- **Metrics anchored to decisions, not surface text.** Fidelity is scored on downstream robot decision success, corrected for the decision-maker's own sampling noise. Never BLEU, never CIDEr.
- **Refuse to fabricate.** When a counterfactual replay diverges past the point where the trace still applies, halt and report. A diverged run is a result, not an error to swallow.
- **The framework must be as reproducible as the thing it measures.** Plumbline's own tests prove record-then-replay yields identical model I/O.

### 1.3 Non-goals

No training or fine-tuning. No new VLA or foundation model. No high-frequency (30 to 50 Hz) motor control loop; the target is the roughly 1 Hz semantic loop. Not a fork of any runtime.

---

## 2. System overview

Four layers. Each has a stable interface to the next and is independently useful.

```
                          +-------------------------------------------------+
   target runtime         |                   PLUMBLINE                     |
   (e.g. OM1, unmodified) |                                                 |
        |                 |  Layer 1: Record-Replay Substrate               |
   model calls (HTTP) ----+--> HTTP proxy ---> Recorder ---> Trace Store    |
        |                 |        ^                              |         |
   bus events (Zenoh) ----+--> Zenoh tap ------^                  |         |
        |                 |                                       v         |
        |                 |  Layer 2: Fidelity Metrics  <---- Trace + Replay|
        |                 |        (caption / fusion / stability)           |
        |                 |                  |                              |
        |                 |  Layer 3: Regression Gate (CI) <----+           |
        |                 |        golden episodes, drift, attribution      |
        |                 |                                                 |
        |                 |  Layer 4: Adapters (OM1 reference) + Bench       |
        |                 +-------------------------------------------------+
```

- **Layer 1** records every nondeterministic external interaction at the four seams and replays them, faithfully (bit-identical) or counterfactually (one component swapped, divergence handled).
- **Layer 2** uses recorded traces plus counterfactual replay to compute fidelity metrics scored on decision success, with a noise floor.
- **Layer 3** turns recorded "golden" episodes into a CI gate that fails when a config change drifts behavior.
- **Layer 4** is the per-runtime glue plus the benchmark dataset.

---

## 3. Core abstractions (Layer 1)

This section defines the types every other layer depends on. Signatures are Python (see Section 14.1 for the language decision). They are interface sketches, not final code; types are illustrative.

### 3.1 Seams

A seam is a boundary in the loop where a nondeterministic interaction crosses between components. There are exactly four, and naming them is itself part of the contribution because no prior work frames a language bus this way.

```python
class Seam(enum.Enum):
    SENSOR_TO_CAPTION = "sensor_to_caption"   # raw frame/audio/state -> caption text
    CAPTION_TO_FUSE   = "caption_to_fuse"     # set of captions + rules + RAG -> fused prompt
    FUSE_TO_DECIDE    = "fuse_to_decide"      # fused prompt -> action plan
    DECIDE_TO_ACT     = "decide_to_act"       # action plan -> HAL commands
```

Mapping to interception mechanism (Section 4):
- `SENSOR_TO_CAPTION`: HTTP proxy (the VLM/ASR call).
- `FUSE_TO_DECIDE`: HTTP proxy (the Cortex LLM call).
- `CAPTION_TO_FUSE`: derived (captions are VLM outputs already captured; the fused prompt is the Cortex input already captured) or observed via the Zenoh tap. No separate model call.
- `DECIDE_TO_ACT`: Zenoh tap / ROS2 (the action plan and resulting HAL commands on the bus).

So three of four seams come from the HTTP proxy and one from the bus tap. The proxy is the spine.

### 3.2 The SeamEvent and the Trace

A `SeamEvent` is one captured interaction. An `Episode` is an ordered sequence of events sharing an episode id (one robot run). A `Trace` is a collection of episodes. The schema is the contract between recorder, replayer, store, and metrics; it is defined fully in Section 5.

```python
@dataclass(frozen=True)
class SeamEvent:
    episode_id: str
    seq: int                 # monotonic per-episode ordering index
    seam: Seam
    logical_tick: int        # virtual-clock tick (Section 3.4)
    wall_ts: float           # original wall-clock time (recorded, never used to drive replay)
    request: Payload         # canonicalized request (Section 5.2)
    response: Payload        # canonicalized response
    model_id: str | None     # e.g. "openai/gpt-4o-2024-08-06"
    params: dict             # temperature, top_p, max_tokens, seed if any
    request_digest: str      # content hash of the canonical request (matcher key)
    latency_ms: float
```

`Payload` separates small structured content (inline JSON) from large binary content (images, audio) which is stored content-addressed and referenced by hash, never inlined (Section 5.3).

### 3.3 The interception interface

Everything that captures events implements one interface, so the proxy, the bus tap, and any future mechanism are interchangeable.

```python
class Interceptor(Protocol):
    def on_request(self, seam: Seam, request: Payload, ctx: Context) -> None: ...
    def on_response(self, seam: Seam, response: Payload, ctx: Context) -> None: ...
    # In replay mode, an interceptor may instead SERVE a response from the trace:
    def maybe_replay(self, seam: Seam, request: Payload, ctx: Context) -> Payload | None: ...
```

`maybe_replay` returning non-None means "do not call the real model, use this." This is the hinge between record and replay.

### 3.4 The virtual clock

OM1's loop is nominally hertz-driven. Wall-clock time leaking into the loop (a slow cloud call changing what the next tick observes) is a nondeterminism source. The virtual clock records a logical tick on every event and, in replay, exposes recorded ticks rather than wall time.

```python
class VirtualClock:
    def now_tick(self) -> int: ...
    def advance(self) -> int: ...
    def bind_replay(self, episode: Episode) -> None: ...   # serve recorded ticks
```

**Determinism envelope (state this honestly, Section 14.4):** Plumbline guarantees that on replay every model call receives the recorded request and returns the recorded response, so the *sequence of decisions and actions* is reproduced. It controls the runtime's internal scheduler only if the adapter exposes a clock hook. Absent that hook, loop timing may vary while model I/O does not. We claim deterministic model I/O replay, not deterministic wall-clock scheduling.

### 3.5 The Recorder

```python
class Recorder:
    def __init__(self, store: TraceStore, clock: VirtualClock): ...
    def record(self, event: SeamEvent) -> None: ...
    def open_episode(self, episode_id: str, metadata: dict) -> None: ...
    def close_episode(self, episode_id: str) -> None: ...
```

The recorder canonicalizes payloads (Section 5.2), computes digests, assigns `seq` and `logical_tick`, and writes to the store. It is append-only per episode.

### 3.6 The Replayer

Two modes. Faithful replay serves every seam from the trace and must reproduce behavior bit-identically. Counterfactual replay re-executes a declared live frontier and serves the rest, handling divergence. This is the keystone; its full semantics are Section 6.

```python
class Replayer:
    def __init__(self, store: TraceStore, clock: VirtualClock,
                 matchers: dict[Seam, Matcher]): ...

    def faithful(self, episode_id: str) -> ReplayResult: ...

    def counterfactual(self, episode_id: str,
                       live_frontier: set[Seam],
                       overrides: dict[Seam, Callable],   # the swapped component(s)
                       on_divergence: DivergencePolicy
                       ) -> ReplayResult: ...
```

### 3.7 The Matcher

For counterfactual replay, when a downstream seam's live request must be compared to the recorded request to decide whether the recorded response still applies.

```python
class Matcher(Protocol):
    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict: ...

@dataclass
class MatchVerdict:
    is_match: bool
    distance: float          # 0.0 = identical; matcher-specific scale
    reason: str
```

Built-in matchers:
- `ExactMatcher`: byte/structural equality for structured fields (action schemas, params).
- `EmbeddingMatcher(threshold)`: cosine distance between embeddings of free text (captions, prompts); match if distance < threshold. The embedding model is itself pinned and recorded so the matcher is reproducible.
- `NumericToleranceMatcher(rtol, atol)`: for pose/coordinate payloads.

---

## 4. Interception mechanisms (Layer 1, concrete)

### 4.1 Why two mechanisms

Three seams are model calls over HTTP; one is bus traffic. So there are two capture mechanisms, both implementing `Interceptor`.

### 4.2 The recording HTTP proxy (primary, the key design choice)

OM1 and most language-bus runtimes reach cloud models over HTTPS to provider endpoints (OpenAI, Gemini, DeepSeek, xAI, Anthropic, or a local Ollama HTTP server). Plumbline runs a local proxy and the runtime is pointed at it by setting the provider base URL (an environment variable or config field in every one of these clients). No source change.

```
OM1 --(OPENAI_BASE_URL=http://localhost:8900)--> Plumbline proxy --(real)--> api.openai.com
```

- **Record mode:** proxy forwards to the real endpoint, captures the request and response, normalizes them per provider (Section 5.4), emits a `SeamEvent` (seam inferred from endpoint and payload: vision request -> `SENSOR_TO_CAPTION`, the Cortex chat completion -> `FUSE_TO_DECIDE`), forwards the real response back to OM1 unchanged.
- **Replay mode:** proxy does not forward. It matches the incoming request against the trace (by `request_digest` for faithful, by `Matcher` for counterfactual) and serves the recorded response. On a counterfactual miss it applies the `DivergencePolicy` (Section 6).

Why this is the right call: it is zero-touch, provider-agnostic, and works for any runtime that talks HTTP to its models, which is essentially all of them. It also captures exactly the seams we care about for free. The cost is that it cannot see purely in-process model calls (a locally embedded model invoked via FFI, not HTTP); for those an in-process shim adapter would be needed, which is a documented fallback, not the main path.

Implementation: an async proxy (mitmproxy as a library, or a small custom asyncio/`httpx` reverse proxy with TLS termination via a locally trusted CA). Streaming responses (SSE token streams) must be captured whole and replayed with the same chunk boundaries, because some runtimes behave differently on streaming granularity; the trace stores the assembled body plus the chunk framing.

### 4.3 The Zenoh tap (secondary, for the action seam)

OM1's data bus and action orchestration ride Zenoh (and ROS2). The tap is a passive Zenoh subscriber on the relevant key expressions. It records the action plan emitted by the Cortex and the resulting HAL commands as `DECIDE_TO_ACT` events. It is observe-only in record mode. In replay it is generally not needed (the action plan is downstream of the decision, which is already reproduced), but it is the ground truth for the *behavioral* comparison the regression gate needs (Section 7), because the action stream is the robot's actual behavior.

---

## 5. The trace format (Layer 1 data contract)

### 5.1 Principles

Language-neutral, append-only, content-addressed for large blobs, aligned to OpenTelemetry GenAI semantic conventions so existing tooling can read it, and **never pickle** (the LeRobot CVE-2026-25874 RCE is the standing lesson: unauthenticated pickle deserialization over the wire is how you get arbitrary code execution; Plumbline uses JSON for metadata and safetensors for tensors).

### 5.2 Canonicalization

Before hashing or storing, every request and response is canonicalized so that semantically identical payloads produce identical digests: sorted JSON keys, normalized whitespace, fixed float formatting, provider-specific noise fields (request ids, timestamps, server-assigned ids) stripped into a separate non-digested `meta` block. The digest covers only the semantically meaningful request so that matchers and faithful replay are stable across runs.

### 5.3 Storage layout

```
trace/
  episodes/
    <episode_id>/
      manifest.json          # episode metadata, config snapshot, seam index
      events.jsonl           # one canonical SeamEvent per line (no blobs inline)
  blobs/
    <sha256>.safetensors     # tensors (image arrays, embeddings)
    <sha256>.bin             # opaque media (encoded image/audio), typed by manifest
  config/
    <config_hash>.json       # full runtime config + model versions for this episode
```

Content addressing means an image that recurs across episodes is stored once. `events.jsonl` references blobs by hash.

### 5.4 OTel GenAI alignment and the span schema

Each `SeamEvent` maps to a span. Attributes follow `gen_ai.*` where defined (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.request.temperature`, `gen_ai.response.id`, `gen_ai.usage.*`) and add a `plumbline.*` namespace for what GenAI conventions do not cover (`plumbline.seam`, `plumbline.logical_tick`, `plumbline.request_digest`, `plumbline.episode_id`, `plumbline.seq`). This means a Plumbline trace is viewable in any OTel-GenAI-aware backend (Langfuse, Phoenix, Grafana Tempo) as well as by Plumbline's own tools, which matters for the "existing observability stays green" demonstration in Section 7.

### 5.5 Provider normalizers

Each provider has a small normalizer that maps its wire format to the canonical request/response and back, and tags the seam. This is the only provider-specific code in the substrate. Normalizers exist for the OpenAI chat/vision schema (which xAI, DeepSeek, and Ollama largely mirror), the Gemini schema, and the Anthropic Messages schema.

---

## 6. Counterfactual replay (the keystone)

This section is the one a reviewer will scrutinize hardest and the one most likely to be wrong if hand-waved. It is specified, not gestured at.

### 6.1 The problem

Naive counterfactual replay assumes swapping a component changes only that component's output and everything downstream still serves from the trace. That breaks the instant the swapped output changes the *shape* of what follows. A new captioner may emit a different number of captions, different wording, different timing, causing the Fuser to build a structurally different prompt, at which point the recorded `FUSE_TO_DECIDE` request no longer corresponds to the live one and serving its recorded response is fabrication.

### 6.2 Live frontier

A counterfactual run declares a `live_frontier`: the set of seams permitted to re-execute live. Everything else is pinned to the trace. Three standard configurations:

- **Isolated:** only the swapped seam runs live. Maximum attribution; every downstream seam must match the trace or the policy fires. Answers "what did this swap change, holding all else fixed."
- **Downstream-live:** the swapped seam and everything after it run live; only upstream is pinned. Answers "what is the end-to-end behavioral effect of this swap." Attributes less precisely.
- **Pinned-decision:** re-run perception live but pin the Cortex LLM to recorded outputs where inputs still match. Isolates perception changes from decision noise.

### 6.3 Per-seam divergence handling

At each seam downstream of the swap, the live request is compared to the recorded request by that seam's `Matcher`:

- **Match (distance < threshold):** the recorded response still applies; serve it (unless the seam is in the live frontier, in which case re-execute).
- **Mismatch (divergence):** apply the `DivergencePolicy`.

```python
class DivergencePolicy(enum.Enum):
    HALT        = "halt"          # default: stop, mark episode diverged, report seam+distance
    GO_LIVE     = "go_live"       # re-execute this seam and everything downstream live
    RECORD_NEW  = "record_new"    # go live AND record a new trace branch (for re-baselining)
```

### 6.4 Halt-on-divergence is the default, and divergence is a result

The default policy is HALT. Silent continuation past divergence is the failure mode that makes a replay tool produce confident garbage; the text-agent assurance-framework literature calls the equivalent "failure on exhaustion." A halted run is not an error to suppress. "This captioner swap diverged at the fuse seam in 40% of episodes, and in those episodes the decision flipped" is exactly the finding the project exists to produce. The `ReplayResult` records, per episode, the first divergence seam and its distance, and the regression gate consumes that as attribution.

### 6.5 Scope honesty

Counterfactual replay is reliable when the swapped component is downstream-pure (its change does not restructure upstream context) and explicitly bounded when it is not. Plumbline detects divergence, reports it, and refuses to fabricate across it. That refusal is a feature, and it is the line that separates a trustworthy tool from a misleading one.

---

## 7. Fidelity metrics (Layer 2)

This is the scientific core and the part most exposed to "your metric is arbitrary." Every definition below is anchored to downstream decision success and corrected for the decision-maker's own sampling noise. Related work to differentiate against is named in Section 12; ViSIL in particular is the primary metric baseline and its lineage (score a summary by downstream task success, not surface similarity) is acknowledged, not claimed.

### 7.1 The decision distribution

Let the decision-maker be the Cortex LLM (or a fixed probe decision function for controlled experiments). For an input context `x`, define `D(x)` as the distribution over decisions (action-plan classes) the decision-maker produces. Because the decision-maker samples at temperature, `D(x)` is estimated by drawing `N` samples.

```python
def decision_distribution(decider, context, n: int) -> Distribution: ...
```

Decisions are compared by a divergence `div(P, Q)` over the action space. Default: total variation distance for discrete typed action plans; Jensen-Shannon for soft distributions; a task-success-rate gap when episodes have a defined success criterion.

### 7.2 The decision-stability noise floor (the differentiating step)

The decision-maker disagrees with itself across samples. That self-disagreement is the noise floor below which no fidelity gap is real. Estimate it by splitting the `N` samples of `D(x)` into two halves and measuring their divergence:

```
sigma(x) = E[ div( D_half1(x), D_half2(x) ) ]
```

Every fidelity number is reported against this floor. A measured gap counts only insofar as it exceeds `sigma`. This is what ViSIL (which stabilizes with a geometric mean over three samples) does not do, and it is a defensible methodological difference, not a renaming.

### 7.3 Caption fidelity

Given raw sensor input `S` with ground-truth scene state `G` (available in simulation), the oracle context is `render(G)`, a faithful structured description of ground truth. The captioner produces caption `C`. Then:

```
caption_loss(C) = max(0,  div( D(C), D(render(G)) )  -  sigma )
```

Read: how much does acting on the caption diverge from acting on ground truth, beyond the decision-maker's own noise. This turns the founders' LiDAR-dog anecdote (a caption that dropped the collision context, so the robot turned toward the obstacle) into a number. `sigma` here is computed on `render(G)` so the floor reflects noise at the oracle input.

### 7.4 Fusion fidelity

Captions `C_1..C_k` enter the Fuser and produce fused prompt `F`. Fusion loss is task-relevant information present in the captions but not recoverable from `F`. Operationalize by counterfactual salience: for each caption `C_i`, compare the decision on `F` against the decision on `F` with `C_i`'s content re-emphasized (`F + salient(C_i)`):

```
fusion_loss = sum_i  weight_i * max(0,  div( D(F), D(F + salient(C_i)) )  -  sigma )
```

If re-adding `C_i` changes the decision beyond the noise floor, the Fuser dropped task-relevant information from `C_i`. `weight_i` reflects the task relevance of `C_i` (uniform by default; learned or rule-based optionally). This separates fusion loss from caption loss because `F` and the `C_i` it was built from are both captured at their seams; the substrate makes the decomposition possible.

### 7.5 The behavioral-equivalence judge

When ground truth is unavailable (real-robot recordings), fidelity and regression comparisons fall back to behavioral equivalence between two runs' action sequences. Two mechanisms:

- **Structural:** typed action plans compared field-wise (an `ExactMatcher`/`NumericToleranceMatcher` over the action schema).
- **Semantic:** an LLM-as-judge given both action/trajectory sequences and asked whether they are behaviorally equivalent. The judge call goes through the same proxy and is recorded, so the eval is as reproducible as the thing it evaluates. The judge's own noise floor is measured the same way as `sigma`.

### 7.6 The three headline experiments, as metric instances

- **Experiment A (fidelity-versus-bandwidth curve):** sweep caption verbosity and fusion frequency, plot `1 - caption_loss` and task success against effective bus bandwidth, find the knee. Sim-bound by construction (needs `G`); Section 9 and Section 14.5 state this honestly. Headline figure.
- **Experiment B (silent-regression detection, against baselines):** record golden episodes with captioner A, counterfactual-replay with captioner B, show Plumbline flags behavior inversion while OM1's latency dashboard and a generic OTel-GenAI tracer both stay green. The baselines are the point; "existing observability says fine, we say broken, the robot was broken" is the demo.
- **Experiment C (caption-quality-for-decisions leaderboard):** rank candidate captioners by `E[1 - caption_loss]` across the episode set; demonstrate that the best caption by NLP surface metrics is not the best caption for behavior.

---

## 8. Regression gate (Layer 3)

### 8.1 Golden episodes

A versioned set of recorded episodes whose behavior has been accepted as good. Stored as full traces plus an accepted-behavior summary (the action sequence and any success labels).

```python
class GoldenSet:
    def add(self, episode_id: str, label: BehaviorLabel) -> None: ...
    def version(self) -> str: ...        # content hash of the set
```

### 8.2 The gate

```python
def gate(candidate_config: Config,
         golden: GoldenSet,
         drift_threshold: float) -> GateResult: ...
```

For each golden episode, counterfactual-replay under `candidate_config` (the swapped model, edited prompt, or changed rule is the override), compute behavioral drift from the golden behavior, and fail if drift exceeds threshold on any episode (configurable: any-episode, aggregate, or quantile). The result includes per-episode drift, per-seam divergence attribution (from Section 6.4), and the diverged-episode fraction.

### 8.3 Drift metric

Behavioral drift between candidate behavior `B_c` and golden behavior `B_g`:

```
drift = div_behavior(B_c, B_g)
```

where `div_behavior` is the structural-or-semantic behavioral equivalence distance from Section 7.5, aggregated over the episode's action sequence (alignment then per-step distance, so insertions/deletions from a different number of actions are penalized).

### 8.4 CI integration

A GitHub Action (and a plain CLI for other CI) that runs the gate on a config change and posts the result. Green/red plus a link to the trace-diff view (Section 11). The action is the deliverable that makes "CI for robot behavior" literal.

### 8.5 Honest positioning

Eval-gated CI, golden cases, median-of-N against judge noise, and fail-on-drift are established LLM-regression-testing practice (Section 12). What is new here is the target (embodied robot decisions) and the determinism the replay substrate provides (you are gating on reproduced behavior, not re-rolled samples). The README says this plainly rather than implying the gate mechanism is invented.

---

## 9. Adapters and the OM1 reference (Layer 4)

### 9.1 The adapter contract

An adapter teaches Plumbline how to attach to a specific runtime. It is small by design.

```python
class Adapter(Protocol):
    def configure_proxy(self) -> ProxyConfig: ...
        # how to point the runtime's model clients at the proxy
        # (env vars / config fields to set, per provider)

    def bus_tap(self) -> BusTap | None: ...
        # how to subscribe to the action bus, if any

    def seam_of(self, request: Payload, endpoint: str) -> Seam: ...
        # classify a captured call into a seam

    def action_schema(self) -> ActionSchema: ...
        # typed action plan, for behavioral comparison

    def clock_hook(self) -> ClockHook | None: ...
        # optional: control the runtime's loop clock for full determinism
```

### 9.2 The OM1 adapter

- `configure_proxy`: set the base URL for each configured provider in OM1's `config/*.json5` (or via env) to the Plumbline proxy. Zero source changes.
- `bus_tap`: subscribe to OM1's Zenoh action key expressions and the natural-language data bus topics.
- `seam_of`: vision/ASR requests -> `SENSOR_TO_CAPTION`; the Cortex chat completion (the fused prompt) -> `FUSE_TO_DECIDE`; bus action plans -> `DECIDE_TO_ACT`. `CAPTION_TO_FUSE` is reconstructed by associating the captions (VLM responses) with the subsequent fused prompt (Cortex request) within an episode tick.
- `action_schema`: OM1's elemental commands (`move(x,y,yaw)`, named skills like "shake paw," speech acts, expressions) typed for structural comparison.
- `clock_hook`: initially None (accept the model-I/O determinism envelope of Section 3.4). If OM1 exposes or accepts a hook on its `hertz` loop, upgrade to full scheduler determinism. Listed as an open item (Section 14.4).

### 9.3 Simulation

Episodes are generated in Gazebo (Go2) and Isaac Sim (Go2 and G1, both supported in OM1 today, physics-accurate, GPU-accelerated). Sim is required for caption fidelity because it provides ground-truth scene state `G`; the extraction of `G` from the simulator (object poses, agent state) is an adapter responsibility and an open item for Isaac (Section 14.5).

---

## 10. The benchmark dataset

The golden-episode dataset is the part most often underestimated and the honest justification for team scale. It is a curated, versioned, annotated set of robot episodes across tasks and embodiments. Each episode carries: the full recorded trace, ground-truth scene state where sim-generated, a behavior label, and a task-success criterion. Coverage spans a quadruped (Go2) and a humanoid (G1), in Gazebo and Isaac, with a designed spread of scenarios that stress the caption/fuse bottleneck (the obstacle-context case, multi-object scenes, ambiguous instructions, governance-rule triggers). Curation, annotation, and ground-truth labeling are manual and slow, and the fidelity and regression results rest entirely on the dataset's quality, so it gets a dedicated workstream.

---

## 11. Observability and tooling

- **Grafana panels** for fidelity over time and per-seam attribution, sitting alongside (not replacing) OM1's existing latency stack. Because traces are OTel-GenAI-aligned, they also load in Tempo/Phoenix/Langfuse.
- **Trace-diff viewer:** given two episodes (or a faithful and a counterfactual run of the same episode), show side by side where they diverged and which seam introduced the divergence. This is both a debugging tool and the visual the Experiment B demo uses.
- **CLI** (`plumbline record`, `plumbline replay`, `plumbline gate`, `plumbline fidelity`, `plumbline diff`) as the primary interface; the GitHub Action wraps `gate`.

---

## 12. Related work and differentiation

Stated up front in the README, not as defensive throat-clearing but as the signal that the project knows its field. From the novelty audit:

- **ViSIL** (information-loss metric scored by downstream VQA via VLM inference): the closest metric precedent and the lineage Plumbline's fidelity metric sits in. Differentiation: closed-loop embodied *decision* success rather than passive VQA; the decision-stability noise floor; in-runtime measurement at the fuse seam. Cite as primary metric baseline.
- **Trace-Based Assurance Framework (arXiv 2603.18096)** (Message-Action Traces, deterministic replay, fault injection, governance at the language-to-action boundary): the closest record-replay/governance precedent, but text/service-only with no perception, no modality boundary, no empirical study. Differentiation: the seams are sensor->caption->fuse->decide->act in an embodied runtime.
- **AgentRR (arXiv 2505.17716)**: generalized experience replay for text/GUI agents, explicitly not bit-perfect and not embodied. Its "check functions" are a cousin of the input-consistency matchers. Plumbline does the faithful replay AgentRR declines to.
- **Digital-twin execution tracing (arXiv 2508.11406)**: classical-robotics determinism (deterministic planners, fixed sim), not replay of nondeterministic cloud model calls. Does not touch the caption/fuse/LLM-decide loop.
- **VLA-FEB**: an offline meta-analysis of monolithic end-to-end VLAs with representation-level fusion metrics; low overlap (different object, offline, not decision-scored). Dispatched in a sentence.
- **Deterministic Simulation Testing (FoundationDB, Antithesis)** and **generic deterministic LLM-agent harnesses**: the acknowledged source of the virtual-clock and record-replay primitives. Borrowed and credited; the embodied/multimodal application is the new part.
- **Agent observability (Langfuse, LangSmith, Phoenix, AgentOps, OTel GenAI)**: score text-task quality, latency, cost; none scores embodied decision success or fusion fidelity. This is the corroboration for "baselines stay green while the robot is broken."

The single defensible novel claim, stated atomically: *deterministic, seam-by-seam record-replay of the nondeterministic perception/language/decision model calls of an embodied language-bus robot runtime, with counterfactual single-component swap, halt-on-divergence, and downstream-decision scoring corrected by a decision-stability noise floor.* No single prior work crosses the modality boundary into an embodied LLM loop this way.

---

## 13. Workstreams, roadmap, and team

### 13.1 Workstreams

- **WS1 Substrate (2 eng, systems):** seams, trace model, virtual clock, recorder, replayer (faithful + counterfactual), matchers. Owns the determinism guarantee. Critical path.
- **WS2 Trace + proxy (1 to 2 eng):** HTTP recording proxy, provider normalizers, OTel-GenAI schema, content-addressed store, safe serialization. The Zenoh tap.
- **WS3 Fidelity metrics (2 eng, ML/eval):** decision distributions, the noise floor, caption and fusion fidelity, the behavioral-equivalence judge, the golden-episode dataset. The scientific core.
- **WS4 Regression gate + observability (1 eng):** the gate, drift metric, GitHub Action, Grafana panels, trace-diff viewer.
- **WS5 OM1 adapter + sim (1 to 2 eng):** proxy config, Zenoh tap, seam classification, action schema, Gazebo and Isaac episode generation, the cross-embodiment Go2 + G1 demo.

The headcount is justified by the dataset (WS3) and cross-embodiment breadth (WS5), the two things that do not compress, not by the substrate code.

### 13.2 Roadmap

- **Weeks 1 to 2.** Freeze the substrate interfaces (Section 3) and the trace schema (Section 5). Milestone: faithful-replay a toy two-model loop bit-identically (Plumbline's own determinism test, Section 15).
- **Weeks 3 to 4.** OM1 adapter early: HTTP proxy recording a real OM1 episode in Gazebo. Validates every downstream design against a real runtime, not a mock.
- **Weeks 5 to 8.** Counterfactual replay through the OM1 adapter: swap a captioner, divergence handling live. Milestone: a swap that halts on divergence and reports the seam.
- **Weeks 9 to 12.** Fidelity layer: Experiments A and C. Publish the bandwidth curve and the captioner leaderboard.
- **Weeks 13 to 16.** Regression gate: Experiment B against the latency-dashboard and generic-tracer baselines. CI action, Grafana panels, trace-diff viewer.
- **Weeks 16+.** Cross-embodiment (G1), then optional expansions: deeper vision-seam coverage as OM1 matures it, an edge async-inference angle on the 1 Hz staleness problem, and a multi-robot coordination eval (the FABRIC-thesis extension, which the substrate is a prerequisite for).

---

## 14. Open decisions

These are genuinely undecided and should be settled early, with a default proposed for each.

### 14.1 Core language
**Default: Python core.** The fidelity layer (VLM judges, embeddings, datasets, metrics) is unavoidably Python, and the substrate's value is in the proxy and trace logic, which Python handles fine. The proxy is I/O-bound, not CPU-bound, so Go/Rust buys little. Risk: a Python proxy in the hot path adds latency to the runtime during recording; mitigate by making recording async and out-of-band, and by noting that recording overhead does not affect replay (where determinism matters). Reconsider a Rust/Go proxy only if recording latency perturbs OM1's loop enough to change behavior.

### 14.2 TLS interception
The proxy must terminate TLS to read provider traffic, which means a locally trusted CA the runtime accepts. Acceptable for a dev/eval tool on a controlled machine; documented setup. For Ollama (local, often plain HTTP) the issue disappears.

### 14.3 Streaming responses
Token-stream (SSE) responses must be captured whole and replayed with original chunk framing. Decision: store assembled body plus chunk boundaries; default replay reproduces framing, with an option to serve unframed for runtimes that do not care.

### 14.4 Clock control / determinism envelope
**Default: model-I/O determinism only**, no scheduler control, until an OM1 clock hook exists. Document the envelope precisely (Section 3.4). This is the most important honesty boundary in the project; do not let the README imply full wall-clock determinism.

### 14.5 Ground-truth extraction from Isaac/Gazebo
Caption fidelity needs scene state `G`. Extracting it (object poses, agent state, semantic labels) is simulator-specific and an adapter responsibility. Open: how much of `G` to extract and how to render it into the oracle context `render(G)` without baking in assumptions that flatter the metric. Settle with WS3 before Experiment A.

### 14.6 Action-equivalence definition
Behavioral drift depends on `div_behavior`. Open: the alignment and per-step distance for action sequences of differing length, and how much to lean on the LLM judge versus structural comparison. Settle with WS3/WS4; it determines the gate's false-positive rate.

### 14.7 VLA-FEB read
One unfinished homework item from the audit: read VLA-FEB's actual metric definitions (not the review's summary) to confirm the low-overlap verdict before publishing the related-work section. Cheap, do it first.

---

## 15. Testing and self-verification

A framework about reproducibility must be reproducible itself, and that property is the first test, not an afterthought.

- **Determinism property test:** record a toy two-model loop (a stub captioner and a stub decider with controllable nondeterminism), faithful-replay it, assert byte-identical model I/O across the run and across machines. This is CI gate zero; if it fails, nothing else matters.
- **Divergence-detection test:** inject a known divergence (swap the stub captioner for one that emits a differently-shaped output), run counterfactual replay, assert the replayer halts at the correct seam with a distance above threshold and does not serve a stale response.
- **Noise-floor calibration test:** with a fixed input and a known-temperature stub decider, assert `sigma` converges to the analytic self-divergence as `N` grows, so the floor is measuring what it claims.
- **Matcher tests:** exact, embedding, and tolerance matchers against crafted near-miss payloads.
- **Proxy fidelity test:** assert the proxy forwards and records without altering the response the runtime receives in record mode (zero-touch invariant).
- **End-to-end OM1 test (from week 3):** record a real Gazebo episode, faithful-replay it, assert the reproduced action sequence matches the recorded one.
- **Reproducibility CI:** the determinism and divergence tests run on every commit; a green build means the substrate still guarantees what it claims.

---

## 16. What this is, in one paragraph

A standalone, runtime-agnostic, open-source framework that makes a language-bus robot runtime reproducible, regression-testable, and fidelity-measurable, with OM1 as the flagship zero-touch integration. The substrate records and replays the nondeterministic model calls at the four seams of the perception-to-action loop; the metric layer scores how much task-relevant information survives the caption and fuse bottleneck, anchored to robot decision success and corrected for the decision-maker's own noise; the gate turns recorded golden episodes into CI that fails when a model, prompt, or rule change drifts behavior past the point existing latency and tracing tools can see. It borrows its record-replay primitives from deterministic simulation testing and its extrinsic-evaluation stance from the summarization-quality literature, and it is the first to carry both across the modality boundary into a running embodied LLM loop.