# Writing an adapter

An **adapter** teaches Plumbline how to attach to one runtime. The core knows about seams, traces, and model calls; the adapter knows about one runtime's wiring — where its model base URL lives, which bus carries actions, how to classify a captured call into a seam, and how to type its action plan. Adapters live in `plumbline/adapters/` and are **not** frozen (unlike `core/`); the surface evolves as runtimes are integrated.

Read [concepts.md](concepts.md) first for the four seams and the lifecycle. This page is the how-to.

## The contract: one `Adapter` Protocol, seven methods

```python
class Adapter(Protocol):
    def configure_proxy(self) -> ProxyConfig: ...
    def bus_tap(self) -> BusTap | None: ...
    def seam_of(self, request: Payload, endpoint: str) -> Seam: ...
    def action_schema(self) -> ActionSchema: ...
    def clock_hook(self) -> ClockHook | None: ...
    def reconstruct_caption_to_fuse(self, *, episode_id, seq, logical_tick,
                                    captions, fused_prompt, wall_ts=0.0) -> SeamEvent: ...
    def reconstruct_decide_to_act(self, *, episode_id, seq, logical_tick,
                                  decision_response, wall_ts=0.0) -> SeamEvent: ...
```

Plus an `ActionSchema` (two members):

```python
class ActionSchema(Protocol):
    @property
    def commands(self) -> tuple[str, ...]: ...          # the runtime's action vocabulary
    def parse(self, payload: Payload) -> tuple[Action, ...]: ...   # decision payload → typed actions
```

The worked references are `adapters/om1.py` (flagship, has a bus) and `adapters/generic.py` (bus-less contrast). `adapters/g1.py` is a third (humanoid, gesture-only) proving the contract is embodiment-agnostic.

## Classify vs reconstruct: not all seams are equal

Two of the four seams are *classified* from a live model call the proxy captured; the other two are *reconstructed* from already-captured payloads (they have no model call of their own). This asymmetry is the concrete evidence the contract is general.

| Seam | Source | How the adapter produces it |
|------|--------|------------------------------|
| `SENSOR_TO_CAPTION` | live model call (VLM / ASR) | returned by `seam_of(...)` |
| `FUSE_TO_DECIDE` | live model call (Cortex LLM) | returned by `seam_of(...)` |
| `CAPTION_TO_FUSE` | **derived** | built by `reconstruct_caption_to_fuse(...)` |
| `DECIDE_TO_ACT` | bus tap **and/or** derived | built by `reconstruct_decide_to_act(...)`; the physical command is also tapped off the bus |

So `seam_of` only ever returns `SENSOR_TO_CAPTION` or `FUSE_TO_DECIDE` (OM1 also returns `DECIDE_TO_ACT` for the bus/action endpoint). Never return a reconstructed seam from `seam_of`. Build both reconstructed events with the shared helper:

```python
from plumbline.adapters.base import derived_seam_event
# fills model_id=None, params={}, and the request_digest for you — a pure function,
# so faithful replay of the derived seam stays byte-identical.
```

## `configure_proxy`: env vars vs `config_fields`

`ProxyConfig` is purely declarative — settings the operator applies *externally* (no runtime source changes):

```python
@dataclass(frozen=True)
class ProxyConfig:
    proxy_base_url: str
    env: Mapping[str, str]            # environment variable name → value
    config_fields: Mapping[str, str]  # config-file field path → value
```

Pick based on where the runtime actually reads its model base URL:

- **`config_fields`** — the runtime reads its endpoint from a config file. OM1 does: it uses the JSON5 field `cortex_llm.config.base_url` (verified against OM1's source — *not* per-provider env vars). So `OM1Adapter.configure_proxy()` returns `env={}` and `config_fields={'cortex_llm.config.base_url': 'http://localhost:8900/v1'}`.
- **`env`** — the runtime's client reads a base-URL environment variable. `GenericAgentAdapter` does this: it returns `env={'OPENAI_BASE_URL': '<proxy>/v1'}` and `config_fields={}`, with the var name a constructor field (`base_url_env_var`) so a caller overrides it without editing the adapter.

Either way it is external config. If a base URL already ends in `/v1`, set `append_v1=False` so you don't double it.

## `bus_tap`: return `None` when there is no bus

`bus_tap()` returns a `BusTap | None`. Return `None` when the runtime has no action bus — Plumbline's record/replay/counterfactual loop must (and does) work with the bus mechanism entirely absent; the action seam is then *derived* from the decision response instead of observed. `GenericAgentAdapter.bus_tap()` returns `None` by default (proving exactly this). OM1 returns `None` too when no Zenoh session is injected, and otherwise a `ZenohTap` scoped to the action keys (`cmd_vel`) — **action keys only**, because data-bus telemetry would be mis-recorded as `DECIDE_TO_ACT`.

## `clock_hook`: almost always `None`

Return `None` unless the runtime exposes a hook to drive its loop clock. Both OM1 and the generic adapter return `None`. This is deliberate and load-bearing: **absent a clock hook, Plumbline guarantees deterministic model I/O on replay, not deterministic wall-clock scheduling** (see [determinism-envelope.md](determinism-envelope.md)). Do not let a docstring or comment imply otherwise.

## `action_schema`: type the decision for behavioral comparison

`ActionSchema.parse(payload)` turns a decision payload into `tuple[Action, ...]`, where `Action(kind, name, args)`. `kind` is a free string — OM1 uses `"move" | "speak" | "emotion"`; the generic adapter uses `"act"` for everything. The gate compares these typed actions; wire it with `recommended_behavior_matcher(schema)` which returns an `ActionSchemaMatcher` (numeric fields compared with tolerance, structure compared exactly).

## A minimal adapter, annotated

The smallest useful adapter targets any OpenAI-compatible perception→decide loop with no bus. This is `GenericAgentAdapter` in essence:

```python
from dataclasses import dataclass, field
from collections.abc import Sequence
from plumbline.adapters.base import (
    Action, ActionSchema, BusTap, ClockHook, ProxyConfig, derived_seam_event,
)
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent
from plumbline.proxy.normalizers import contains_image


@dataclass(frozen=True)
class MyActionSchema:
    commands: tuple[str, ...] = ("move_forward", "turn_left", "turn_right", "stop")
    def parse(self, payload: Payload) -> tuple[Action, ...]:
        inline = payload.inline
        if isinstance(inline, dict) and isinstance(inline.get("action"), str):
            return (Action("act", inline["action"], {}),)
        return ()


@dataclass
class MyAdapter:
    proxy_base_url: str
    base_url_env_var: str = "OPENAI_BASE_URL"

    def configure_proxy(self) -> ProxyConfig:                    # runtime reads a base URL from env
        base = self.proxy_base_url.rstrip("/")
        return ProxyConfig(proxy_base_url=base,
                           env={self.base_url_env_var: f"{base}/v1"},
                           config_fields={})

    def bus_tap(self) -> BusTap | None:
        return None                                             # no bus: the act seam is derived

    def seam_of(self, request: Payload, endpoint: str) -> Seam:  # only the two classified seams
        if contains_image(request.inline):
            return Seam.SENSOR_TO_CAPTION                        # an image → a caption call
        return Seam.FUSE_TO_DECIDE                               # text → the decide call

    def action_schema(self) -> ActionSchema:
        return MyActionSchema()

    def clock_hook(self) -> ClockHook | None:
        return None                                             # model-I/O determinism only

    def reconstruct_caption_to_fuse(self, *, episode_id, seq, logical_tick,
                                    captions, fused_prompt, wall_ts=0.0) -> SeamEvent:
        return derived_seam_event(
            seam=Seam.CAPTION_TO_FUSE, episode_id=episode_id, seq=seq,
            logical_tick=logical_tick,
            request=Payload(inline={"captions": list(captions)}),
            response=Payload(inline={"fused_prompt": fused_prompt}), wall_ts=wall_ts)

    def reconstruct_decide_to_act(self, *, episode_id, seq, logical_tick,
                                  decision_response, wall_ts=0.0) -> SeamEvent:
        action = decision_response.inline["action"]             # pull the decided action
        return derived_seam_event(
            seam=Seam.DECIDE_TO_ACT, episode_id=episode_id, seq=seq,
            logical_tick=logical_tick,
            request=Payload(inline={"action": action}),
            response=Payload(inline={"dispatched": True}), wall_ts=wall_ts)
```

## Getting started and conformance

- Copy `plumbline/adapters/_template.py` (a skeleton with every method stubbed and commented) and fill it in.
- Run `assert_conforms(adapter)` to check your adapter satisfies the `Adapter` contract before you wire it into a recording session.
- **Build against a real recorded episode, not a mock.** The OM1 adapter is grounded in OM1's actual source and run-verified end to end; that discipline is why its interface facts (Zenoh key `cmd_vel`, the tool-call wire shape, the `cortex_llm.config.base_url` redirect) are trustworthy — see [om1-integration.md](om1-integration.md).

See also the frozen types this all builds on in [api.md](api.md).
