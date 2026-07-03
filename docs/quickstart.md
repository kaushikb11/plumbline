# Quickstart

This walks the operator flow — **point base URLs at the proxy → record → replay → measure fidelity** — using the shipped Python API. Every snippet here runs against the current build (Python ≥ 3.12). The `plumbline record / replay / gate / diff / scenes` CLI (spec §11) and the CI gate (WS4) are implemented and tested; the Python API below is what those subcommands wrap.

## 1. Point your runtime at the proxy (zero source changes)

`configure_proxy()` returns the config fields (and/or env vars) that redirect the runtime's model base URL to the Plumbline proxy. No runtime source edits.

```python
from plumbline.adapters.om1 import OM1Adapter

cfg = OM1Adapter(proxy_base_url="http://localhost:8900").configure_proxy()
for path, value in cfg.config_fields.items():
    print(f"{path} = {value}")
# cortex_llm.config.base_url = http://localhost:8900/v1
print("env:", dict(cfg.env))
# env: {}
```

OM1 routes model calls through the **`cortex_llm.config.base_url`** config field (verified against OM1's source — it is *not* per-provider env vars); set that field in OM1's `config/*.json5` and its calls go through Plumbline. Other adapters may instead return `env` entries — a runtime that reads a provider base URL from the environment gets those. Either way it is external config, no source change. (See `docs/om1-integration.md`.)

> The async HTTP proxy that actually terminates these connections (`plumbline.proxy.AsyncHTTPProxy`) takes an **injected** transport — a concrete TLS-terminating reverse server is not shipped yet. The transport-agnostic record/replay *core* below is what the substrate is built on and what the tests exercise.

## 2. Record an episode

In record mode the proxy forwards each model call to the real upstream, captures it, infers the seam, records a `SeamEvent`, and returns the upstream response **unaltered** (the zero-touch invariant). Here is the in-process core with a stub "model" standing in for the cloud endpoints:

```python
from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import Payload
from plumbline.proxy import RecordingProxy

store = TraceStore()                      # filesystem trace store (defaults to a temp dir)
recorder = Recorder(store, VirtualClock())

def upstream(request: Payload) -> Payload:
    # In a real run this is the cloud provider; the proxy forwards to it verbatim.
    inline = request.inline
    assert isinstance(inline, dict)
    if inline["kind"] == "caption":
        return Payload(inline={"caption": f"obstacle {inline['tick']} m ahead"})
    return Payload(inline={"action_plan": {"action": "avoid", "args": {}}})

def classify(request: Payload, ctx: Context) -> Seam:
    inline = request.inline
    if isinstance(inline, dict) and inline.get("kind") == "caption":
        return Seam.SENSOR_TO_CAPTION
    return Seam.FUSE_TO_DECIDE

proxy = RecordingProxy(upstream, recorder, classifier=classify)

for tick in range(3):
    # Stamp the loop-iteration index on the context so every seam of one iteration
    # shares a logical_tick — this is what lets counterfactual replay group them.
    ctx = Context(episode_id="demo", model_id="stub/model", params={"temperature": 0.7}, logical_tick=tick)
    caption = proxy.forward(Payload(inline={"kind": "caption", "tick": tick}), ctx)
    proxy.forward(Payload(inline={"kind": "decide", "prompt": str(caption.inline)}), ctx)
proxy.close("demo")
```

## 3. Replay

### Faithful — bit-identical reproduction

`ReplayingProxy` serves each recorded response back by request digest. Re-drive the same loop and you get the same decisions, with no model calls made:

```python
from plumbline.proxy import ReplayingProxy

replay = ReplayingProxy(store, "demo")
replay_ctx = Context(episode_id="demo", model_id="stub/model", params={})
served = replay.faithful(Payload(inline={"kind": "caption", "tick": 0}), replay_ctx)
assert served.inline == {"caption": "obstacle 0 m ahead"}   # served from the trace
```

### Counterfactual — swap one component, isolated mode

Swap the captioner and let only that seam re-execute; everything downstream is pinned to the trace. If the swap diverges enough that the recorded fused prompt no longer applies, the run **halts** at the first downstream seam and reports the seam and distance — it never serves a stale response past a divergence. A compatible captioner (a paraphrase within the matcher threshold) reproduces without diverging; an incompatible one (disjoint content) halts.

```python
from plumbline.core.replayer import Replayer, DivergencePolicy
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher
from plumbline.core.seam import Seam

matchers = {
    Seam.CAPTION_TO_FUSE: EmbeddingMatcher(threshold=0.2),
    Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=0.2),
    Seam.DECIDE_TO_ACT: ExactMatcher(),
}

def swapped_captioner(recorded_request: Payload) -> Payload:
    return Payload(inline={"caption": "the path is clear"})   # drops the obstacle context

result = Replayer(store, VirtualClock(), matchers).counterfactual(
    "go2-gazebo-001",
    live_frontier={Seam.SENSOR_TO_CAPTION},
    overrides={Seam.SENSOR_TO_CAPTION: swapped_captioner},
    on_divergence=DivergencePolicy.HALT,   # the default
)
# result.diverged, result.divergence_seam (CAPTION_TO_FUSE), result.divergence_distance
```

**Worked, runnable examples:** `tests/test_om1_counterfactual.py` records a full Go2 Gazebo episode end to end; `tests/test_proxy_counterfactual.py` does the same through `RecordingProxy` and asserts both the no-divergence (compatible) and halt (incompatible) cases.

> **Episode shape matters.** `counterfactual` groups seam events by `logical_tick` so a swapped seam is compared against the *same loop iteration's* downstream seam. Stamp `Context.logical_tick` with the loop-iteration index (as in the record step above), so every seam of one iteration shares a tick. Faithful replay (by digest) does not depend on this.

## 4. Measure fidelity

Fidelity is scored on downstream **decision success**, corrected for the decider's own sampling noise — never on caption surface text.

```python
from plumbline.fidelity import caption_loss, fusion_loss, decision_stability, structural_equivalence

# The decision-maker: a context (fused prompt / caption) -> an action plan.
def decider(context: str) -> dict[str, object]:
    return {"action": "avoid" if "obstacle" in context else "advance", "args": {}}

# Caption fidelity: how much acting on the caption diverges from acting on ground
# truth (render(G)), beyond the noise floor. render(G) is supplied by the sim (§14.5).
loss = caption_loss(decider, caption="the path is clear", oracle_context="obstacle at 0.3 m", n=64)
print(loss)   # > 0: the caption dropped the obstacle context (the LiDAR-dog failure, as a number)

# The noise floor it is reported against:
sigma = decision_stability(decider, "obstacle at 0.3 m", n=64)
```

For real-robot recordings with no ground truth, use the behavioral-equivalence judge (`structural_equivalence` for typed action plans; `semantic_equivalence` for an LLM-as-judge routed through the proxy so the eval is itself recorded and replayable).

> **Open decisions requiring human review** before fidelity numbers are published: `render(G)` extraction (§14.5) and the `salient`/`weights` operation for fusion loss (§14.6). Both are surfaced in `fidelity/metrics.py` and guarded by `salient_artifact()`. See the `HUMAN REVIEW` banners in that module.

## 5. Gate (WS4)

The regression gate — counterfactual-replay a set of golden episodes under a candidate config, score behavior drift, fail CI past a threshold — is implemented and tested (engineering spec §8). Run it from the CLI:

```bash
plumbline gate path/to/gate_config.py     # exits non-zero on drift; wrap in CI
```

The gate config is a Python file exposing `build() -> GateSpec`; a ready example is `plumbline/bench/example_gate.py`, and the shipped GitHub Action (`.github/workflows/robot-behavior-gate.yml`) wraps this command. See [docs/results-experiment-c.md](results-experiment-c.md) for the fidelity results and `plumbline diff` / `plumbline scenes` for the trace-diff and Experiment-C authoring tools.
