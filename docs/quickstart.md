# Quickstart

The operator flow, end to end: **point your runtime at the proxy → record → replay → measure fidelity**, with runnable snippets. New to the ideas (the four seams, the record → replay → gate lifecycle)? Read [concepts.md](concepts.md) first — it's the one-page mental model.

Every snippet here runs against the current build (Python ≥ 3.12).

## 0. Run something green in one command

Before wiring anything, prove the install works — no extras, no network:

```bash
pip install -e .                          # core only (stdlib)
plumbline gate bench/om1_gazebo_gate.py   # replays a real Go2/Gazebo golden and exits 0
```

That gates the committed Gazebo golden (`om1-gazebo-maze-003`, 4,095 events): it must replay byte-identically and pass — the same check CI runs on every PR. `plumbline list` shows the episodes on disk. `record` / `replay` (below) run a proxy *server* and additionally need `pip install -e '.[proxy]'`.

## 1. Point your runtime at the proxy (zero source changes)

You don't edit the runtime — you redirect its model base URL. `configure_proxy()` returns the config fields (and/or env vars) that do it:

```python
from plumbline.adapters.om1 import OM1Adapter

cfg = OM1Adapter(proxy_base_url="http://localhost:8900").configure_proxy()
for path, value in cfg.config_fields.items():
    print(f"{path} = {value}")
# cortex_llm.config.base_url = http://localhost:8900/v1
```

OM1 routes model calls through the `cortex_llm.config.base_url` config field (verified against OM1's source — not per-provider env vars); set it in OM1's `config/*.json5` and its calls flow through Plumbline. Other adapters may instead return `env` entries. Either way it's external config — no source change. (See [om1-integration.md](om1-integration.md).)

## 2. Record an episode

In record mode the proxy forwards each model call to the real upstream, captures it, infers the seam, records a `SeamEvent`, and returns the response **unaltered** (the zero-touch invariant). Here's the in-process core with a stub standing in for the cloud endpoints:

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

**Faithful — bit-identical reproduction.** `ReplayingProxy` serves each recorded response back by request digest. Re-drive the loop and you get the same decisions, with no model calls made:

```python
from plumbline.proxy import ReplayingProxy

replay = ReplayingProxy(store, "demo")
replay_ctx = Context(episode_id="demo", model_id="stub/model", params={})
served = replay.faithful(Payload(inline={"kind": "caption", "tick": 0}), replay_ctx)
assert served.inline == {"caption": "obstacle 0 m ahead"}   # served from the trace
```

**Counterfactual — swap one component.** Let only the swapped seam re-execute; everything downstream is pinned to the trace. If the swap diverges enough that the recorded fused prompt no longer applies, the run **halts** at the first downstream seam and reports the seam and distance — it never serves a stale response past a divergence. A compatible captioner (a paraphrase within the matcher threshold) reproduces without diverging; an incompatible one (disjoint content) halts.

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
    "demo",   # the episode recorded in step 2 above
    live_frontier={Seam.SENSOR_TO_CAPTION},
    overrides={Seam.SENSOR_TO_CAPTION: swapped_captioner},
    on_divergence=DivergencePolicy.HALT,   # HALT is the recommended policy; it is required
                                           # (no default), and divergence is a result, not an error
)
# result.diverged (True), result.divergence_seam (FUSE_TO_DECIDE for this two-seam demo
# episode — it has no derived CAPTION_TO_FUSE), result.divergence_distance
```

> **Episode shape matters.** `counterfactual` groups seam events by `logical_tick`, so a swapped seam is compared against the *same loop iteration's* downstream seam — stamp `Context.logical_tick` with the loop index (as in step 2). Faithful replay (by digest) does not depend on this. Fully worked examples: `tests/test_om1_counterfactual.py` (a real Go2 Gazebo episode) and `tests/test_proxy_counterfactual.py` (both the compatible and halt cases through `RecordingProxy`).

## 4. Measure fidelity

Fidelity is scored on downstream **decision agreement** — does the caption make the decider act as it would on ground truth — corrected for the decider's own sampling noise, never on caption surface text. (Agreement with the oracle-fed decider, not task correctness; see [limitations.md](limitations.md).)

```python
from plumbline.fidelity import caption_loss, decision_stability

# The decision-maker: a context (fused prompt / caption) -> an action plan.
def decider(context: str) -> dict[str, object]:
    return {"action": "avoid" if "obstacle" in context else "advance", "args": {}}

# Caption fidelity: how much acting on the caption diverges from acting on ground
# truth, beyond the noise floor. The oracle_context (render of ground truth) is
# supplied by the sim/harness.
loss = caption_loss(decider, caption="the path is clear", oracle_context="obstacle at 0.3 m", n=64)
print(loss)   # > 0: the caption dropped the obstacle (the LiDAR-dog failure, as a number)

sigma = decision_stability(decider, "obstacle at 0.3 m", n=64)   # the noise floor it's reported against
```

For real-robot recordings with no ground truth, use the behavioral-equivalence judge: `structural_equivalence` for typed action plans, or `semantic_equivalence` for an LLM-as-judge routed through the proxy (so the eval is itself recorded and replayable).

> The `render(G)` extraction and the `salient`/`weights` operation for fusion loss are open judgment calls flagged for human review before fidelity numbers are published — see the `HUMAN REVIEW` banners in `fidelity/metrics.py` and [math-review-section7.md](math-review-section7.md).

## 5. Gate in CI

The regression gate counterfactually replays a set of golden episodes under a candidate config, scores behavioral drift, and fails past a threshold:

```bash
plumbline gate my_gate.py     # exits non-zero on drift; wrap in CI
```

The gate config is a Python file exposing `build() -> GateSpec` (the golden episodes + the change under test, as seam overrides). A ready example is [`plumbline/bench/example_gate.py`](../plumbline/bench/example_gate.py), and the shipped GitHub Action (`.github/workflows/robot-behavior-gate.yml`) wraps this command. `plumbline diff` and `plumbline scenes` give the trace-diff and Experiment-C authoring tools.

---

**Next:** the gate and record modes as native pytest ([pytest-plugin.md](pytest-plugin.md)) · the full API reference ([api.md](api.md)) · teaching Plumbline a new runtime ([writing-an-adapter.md](writing-an-adapter.md)) · what is and isn't guaranteed ([limitations.md](limitations.md)).
