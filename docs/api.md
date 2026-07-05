# API reference

The load-bearing types and signatures, read off the modules. The `core/` contract is **frozen** (CLAUDE.md invariant 1): the Protocols, dataclasses, fields, and method signatures below are stable. The `fidelity` / `proxy` / `regression` / `adapters` surfaces are public but expected to evolve.

## Imports

The intended public surface is the top-level package:

```python
from plumbline import Seam, SeamEvent, Payload, Recorder, Replayer, TraceStore
```

If a symbol is not yet re-exported at the top level, import it from its submodule — these paths are stable and always resolve:

```python
from plumbline.core.seam import Seam
from plumbline.core.trace import SeamEvent, Payload
from plumbline.core.replayer import Replayer, DivergencePolicy
```

> The submodule paths (`plumbline.core.*`, `plumbline.proxy`, `plumbline.fidelity`, `plumbline.regression`, `plumbline.adapters`) are the fallback and are verified. The flat `from plumbline import …` surface is the curated convenience; if an import fails there, use the submodule path.

---

## `plumbline.core` — the frozen substrate

### `Seam` (`core/seam.py`)

An `enum.Enum` with exactly four members — the pipeline, in order:

```python
Seam.SENSOR_TO_CAPTION   # "sensor_to_caption" — raw frame/audio/state → caption text
Seam.CAPTION_TO_FUSE     # "caption_to_fuse"   — captions + rules + RAG → fused prompt
Seam.FUSE_TO_DECIDE      # "fuse_to_decide"    — fused prompt → action plan
Seam.DECIDE_TO_ACT       # "decide_to_act"     — action plan → HAL commands
```

Do not add, rename, or re-value a member.

### Data types (`core/trace.py`)

All frozen dataclasses. `JSONValue` is the recursive JSON type alias used everywhere for inline content.

```python
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

@dataclass(frozen=True)
class Payload:
    inline: JSONValue                    # small structured content
    blobs: tuple[BlobRef, ...] = ()      # large binary content, content-addressed (never inlined)

@dataclass(frozen=True)
class SeamEvent:
    episode_id: str
    seq: int                             # monotonic per-episode ordering index
    seam: Seam
    logical_tick: int                    # virtual-clock tick; all seams of one loop iteration share it
    wall_ts: float                       # original wall-clock time (recorded, NEVER drives replay)
    request: Payload                     # canonicalized request
    response: Payload                    # canonicalized response
    model_id: str | None                 # e.g. "openai/gpt-4o-2024-08-06"
    params: Mapping[str, JSONValue]      # temperature, top_p, max_tokens, seed if any
    request_digest: str                  # content hash of the canonical request (matcher key)
    latency_ms: float

@dataclass(frozen=True)
class Episode:
    episode_id: str
    events: tuple[SeamEvent, ...]
    metadata: Mapping[str, JSONValue]

@dataclass(frozen=True)
class Trace:
    episodes: tuple[Episode, ...]
```

Supporting types: `BlobRef(sha256, kind, media_type=None)`, `BlobKind` (`SAFETENSORS` | `BIN`), `CanonicalPayload(canonical_json, digest, meta)`, and the on-disk schema types `EpisodeManifest`, `ConfigSnapshot`, `SeamIndexEntry`.

Helpers:

```python
def canonical_dumps(value: JSONValue) -> str          # deterministic JSON: sorted keys, no whitespace noise, allow_nan=False
def canonicalize(payload: Payload) -> CanonicalPayload # → .digest is the matcher / faithful-replay key
```

**`make_seam_event(...)` — the recommended way to hand-build an event** (keyword-only):

```python
def make_seam_event(
    *, episode_id: str, seq: int, seam: Seam, logical_tick: int,
    request: Payload, response: Payload,
    model_id: str | None = None,
    params: Mapping[str, JSONValue] | None = None,
    wall_ts: float = 0.0, latency_ms: float = 0.0,
) -> SeamEvent
```

It fills the `request_digest` (via `canonicalize`) and defaults `params` to `{}`, so you do not compute the digest by hand.

### `Context` and `Interceptor` (`core/interceptor.py`)

```python
@dataclass(frozen=True)
class Context:
    episode_id: str
    model_id: str | None
    params: Mapping[str, JSONValue]
    logical_tick: int = 0        # runtime loop-iteration index; stamped by the loop driver

class Interceptor(Protocol):
    def on_request(self, seam: Seam, request: Payload, ctx: Context) -> None: ...
    def on_response(self, seam: Seam, response: Payload, ctx: Context) -> None: ...
    def maybe_replay(self, seam: Seam, request: Payload, ctx: Context) -> Payload | None: ...
```

`maybe_replay` returning non-`None` means "serve this from the trace, do not call the model" — the hinge between record and replay.

### `Recorder` (`core/recorder.py`)

```python
class Recorder:
    def __init__(self, store: TraceStore, clock: VirtualClock) -> None: ...
    def record(self, event: SeamEvent) -> None            # persists the event AS CAPTURED (no mutation)
    def open_episode(self, episode_id: str, metadata: Mapping[str, JSONValue]) -> None: ...
    def close_episode(self, episode_id: str) -> None: ...
```

### `Replayer` (`core/replayer.py`)

```python
class Replayer:
    def __init__(self, store: TraceStore, clock: VirtualClock,
                 matchers: Mapping[Seam, Matcher]) -> None: ...

    def faithful(self, episode_id: str) -> ReplayResult:
        # loads the recorded event sequence; diverged is always False by construction

    def counterfactual(
        self,
        episode_id: str,
        live_frontier: set[Seam],                                   # single-seam (isolated) in pure-trace replay
        overrides: Mapping[Seam, Callable[[Payload], Payload]],     # re-execute a seam: request → response
        on_divergence: DivergencePolicy,                            # REQUIRED — no default
    ) -> ReplayResult
```

`on_divergence` is a required positional/keyword argument; there is no default (omitting it raises `TypeError`). A multi-seam `live_frontier` raises `NotImplementedError` in pure-trace replay.

```python
class DivergencePolicy(enum.Enum):
    HALT = "halt"              # default choice: stop, mark diverged, report seam + distance
    GO_LIVE = "go_live"        # re-execute this seam and everything downstream live (bounded)
    RECORD_NEW = "record_new"  # go live AND record a new trace branch (re-baselining)

@dataclass(frozen=True)
class ReplayResult:
    episode_id: str
    diverged: bool
    divergence_seam: Seam | None          # first seam where live diverged from the trace
    divergence_distance: float | None     # matcher distance at that seam
    events: tuple[SeamEvent, ...]         # reproduced seam events
```

### `TraceStore` (`core/store.py`)

Filesystem store; JSON metadata + content-addressed blobs, **no pickle**.

```python
class TraceStore:
    def __init__(self, root: Path | str | None = None) -> None:   # None → a fresh temp dir
    @property
    def root(self) -> Path
    def open_episode(self, manifest: EpisodeManifest) -> None
    def append_event(self, episode_id: str, event: SeamEvent) -> None
    def close_episode(self, episode_id: str) -> None
    def load_episode(self, episode_id: str) -> Episode
    def load_manifest(self, episode_id: str) -> EpisodeManifest
    def list_episodes(self) -> tuple[str, ...]
    def put_blob(self, data: bytes, kind: BlobKind, media_type: str | None = None) -> BlobRef
    def get_blob(self, ref: BlobRef) -> bytes
    def put_config(self, snapshot: ConfigSnapshot) -> str          # returns config_hash
    def get_config(self, config_hash: str) -> ConfigSnapshot
```

On-disk layout: `episodes/<id>/manifest.json`, `episodes/<id>/events.jsonl` (one canonical `SeamEvent` per line), `blobs/<sha256>.{safetensors,bin}`, `config/<config_hash>.json`.

### `Matcher` and built-ins (`core/matcher.py`)

Matchers are the input-consistency check for counterfactual replay: given a live request and the recorded one, is the recorded downstream context still valid?

```python
@dataclass(frozen=True)
class MatchVerdict:
    is_match: bool
    distance: float    # 0.0 = identical; scale is matcher-specific (do NOT compare across matchers/seams)
    reason: str

class Matcher(Protocol):
    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict: ...

@dataclass(frozen=True)
class ExactMatcher:                                    # canonical byte/structural equality
    ...
@dataclass(frozen=True)
class EmbeddingMatcher:                                # cosine distance over free text; match if distance < threshold
    threshold: float
@dataclass(frozen=True)
class NumericToleranceMatcher:                         # tolerance compare for pose/coordinate payloads
    rtol: float
    atol: float
```

`EmbeddingMatcher` routes text through a module-level pinned embedder (a dependency-free bag-of-tokens default). Swap in a real semantic model with `set_embedder(embedder)` where `Embedder = Callable[[str], Mapping[str, float]]`.

### `VirtualClock` (`core/clock.py`)

```python
class VirtualClock:
    def __init__(self) -> None
    def now_tick(self) -> int
    def advance(self) -> int
    def bind_replay(self, episode: Episode) -> None    # serve recorded ticks during replay
```

---

## `plumbline.proxy` — record/replay proxy

The transport-agnostic record/replay core and the async HTTP proxy.

```python
@dataclass
class RecordingProxy:
    upstream: Callable[[Payload], Payload]                         # forwards to the real endpoint
    recorder: Recorder
    classifier: Callable[[Payload, Context], Seam] = classify_seam
    digest: Callable[[Payload], str] = default_digest
    episode_metadata: Mapping[str, JSONValue] = {}
    def forward(self, request: Payload, ctx: Context) -> Payload   # returns the upstream response UNALTERED
    def close(self, episode_id: str) -> None

@dataclass
class ReplayingProxy:
    store: TraceStore
    episode_id: str
    matchers: Mapping[Seam, Matcher] = {}
    on_divergence: DivergencePolicy = DivergencePolicy.HALT
    overrides: Mapping[Seam, Callable[[Payload], Payload]] = {}
    upstream: Callable[[Payload], Payload] | None = None
    classifier: Callable[[Payload, Context], Seam] = classify_seam
    digest: Callable[[Payload], str] = default_digest
    def faithful(self, request: Payload, ctx: Context) -> Payload  # serves the recorded response by digest
```

Also public: `AsyncHTTPProxy` + its transport types (`AsyncTransport`, `HTTPRequest`, `HTTPResponse`); the provider normalizers `OpenAIChatNormalizer`, `GeminiNormalizer`, `AnthropicMessagesNormalizer` (and `DEFAULT_NORMALIZERS`, `contains_image`, `extract_data_urls`); OTel-GenAI span mapping `to_span` / `seam_event_attributes`; SSE streaming helpers (`split_sse`, `assemble_openai`, …); `classify_seam`, `default_digest`, and `ProxyDivergence(seam, distance, request_digest, detail=None)`.

Note `AsyncHTTPProxy` takes an **injected** transport — a concrete TLS-terminating reverse server is not shipped; TLS termination is left to a front proxy.

---

## `plumbline.fidelity` — metrics (short-leash; math is human-reviewed)

Fidelity is scored on downstream **decision** success, corrected for the decider's own sampling noise — never on caption surface text. `DeciderFn = Callable[[str], Mapping[str, JSONValue]]` maps a context (fused prompt / caption) to an action plan.

```python
def decision_stability(decider, context: str, n: int, *, label=canonical_label,
                       divergence=total_variation, trials: int = 32, seed: int = 0) -> float
    # the noise floor: how much the decider disagrees with itself on a fixed context

def caption_loss(decider, caption: str, oracle_context: str, n: int, *,
                 label=canonical_label, divergence=total_variation) -> float
    # how much acting on the caption diverges from acting on ground truth (render(G)), beyond the floor

def fusion_loss(decider, fused_prompt: str, captions: Sequence[str], n: int, *,
                salient=default_salient, weights: Sequence[float] | None = None,
                label=canonical_label, divergence=total_variation) -> float

def decision_distribution(decider, context: str, n: int, *, label=canonical_label) -> Mapping[str, float]
def decision_drift(decider, golden_context: str, candidate_context: str, n: int, ...) -> DecisionDrift
```

Behavioral-equivalence judges (for real recordings with no ground truth):

```python
def structural_equivalence(recorded: Sequence[Payload], candidate: Sequence[Payload],
                           *, matcher: Matcher = ExactMatcher()) -> JudgeVerdict
def semantic_equivalence(sequence_a: Sequence[Payload], sequence_b: Sequence[Payload],
                         judge_model: Callable[[Payload], Payload], *, n: int = 1) -> JudgeVerdict
```

`JudgeVerdict` fields: `equivalent: bool`, `distance: float`, `reason: str`, `method: str`.
Type aliases: `Distribution = Mapping[str, float]`, `Divergence = Callable[[Mapping[str, float], Mapping[str, float]], float]` (built-ins `total_variation`, `jensen_shannon`).

> `render(G)` extraction (§14.5) and the `salient`/`weights` operation for `fusion_loss` (§14.6) are open judgment calls surfaced (not hidden) in `fidelity/metrics.py`; see [math-review-section7.md](math-review-section7.md).

---

## `plumbline.regression` — the gate

```python
@dataclass(frozen=True)   # illustrative — see golden.py
class BehaviorLabel:
    actions: tuple[Payload, ...]
    success: bool | None = None

class GoldenEpisode:  GoldenEpisode(episode_id: str, label: BehaviorLabel)
class GoldenSet:      GoldenSet(store: TraceStore)   # then .add(...) golden episodes

@dataclass
class Config:                                        # the candidate config, as seam overrides
    live_frontier: set[Seam]
    overrides: Mapping[Seam, Callable[[Payload], Payload]]
    matchers: Mapping[Seam, Matcher]
    on_divergence: DivergencePolicy = DivergencePolicy.HALT

def gate(store: TraceStore, golden: GoldenSet, config: Config, drift_threshold: float, *,
         behavior_matcher: Matcher = ExactMatcher(),
         policy: FailurePolicy = FailurePolicy.ANY,
         quantile: float = 0.95,
         decision: "DecisionGate | None" = None) -> GateResult
```

`GateSpec` bundles the same arguments for the CLI: a gate config file exposes `build() -> GateSpec`. `GateResult(passed, threshold, policy, per_episode, threshold_units)`; each `EpisodeDrift` carries `drift`, `diverged`, `divergence_seam`, `divergence_distance`, and (when a `DecisionGate` is supplied) `decision_divergence`, `sigma`. `action_sequence(events)` extracts the comparable action `Payload`s from an episode.

`DecisionGate(decider, n=32, alpha=0.05, k=3.0, ...)` is the optional noise-floor-anchored decision-divergence scorer.

---

## `plumbline.adapters` — runtime adapters

See [writing-an-adapter.md](writing-an-adapter.md) for the full walkthrough. The surface:

```python
class Adapter(Protocol):                             # the contract (adapters/base.py)
    def configure_proxy(self) -> ProxyConfig: ...
    def bus_tap(self) -> BusTap | None: ...
    def seam_of(self, request: Payload, endpoint: str) -> Seam: ...
    def action_schema(self) -> ActionSchema: ...
    def clock_hook(self) -> ClockHook | None: ...
    # plus the two reconstruct hooks — see the adapter guide

@dataclass(frozen=True)
class ProxyConfig:
    proxy_base_url: str
    env: Mapping[str, str]                # env var name → value
    config_fields: Mapping[str, str]      # config field path → value

@dataclass(frozen=True)
class Action:
    kind: str; name: str; args: Mapping[str, JSONValue]

class ActionSchema(Protocol):
    @property
    def commands(self) -> tuple[str, ...]: ...
    def parse(self, payload: Payload) -> tuple[Action, ...]: ...
```

Concrete adapters: `OM1Adapter` (flagship, Zenoh `cmd_vel` tap), `GenericAgentAdapter` (bus-less), `G1Adapter` (humanoid, gesture-only). Behavior matcher for the gate: `recommended_behavior_matcher(schema, *, rtol=1e-3, atol=1e-3) -> ActionSchemaMatcher`. Shared helper: `derived_seam_event(...)` (in `adapters/base.py`) builds a reconstructed no-model-call `SeamEvent`.
