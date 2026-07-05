"""The adapter contract (engineering spec §9.1).

An adapter teaches Plumbline how to attach to a specific runtime. The core knows
about seams, traces, and model calls; the adapter knows about one runtime's wiring.

The full contract is ONE `Adapter` Protocol — seven methods plus the `ActionSchema`
they return:

  1. `configure_proxy`             -> ProxyConfig   (how to redirect model calls)
  2. `bus_tap`                     -> BusTap | None (optional passive action tap)
  3. `seam_of`                     -> Seam          (classify a captured model call)
  4. `action_schema`              -> ActionSchema  (parse a decision into typed Actions)
  5. `clock_hook`                  -> ClockHook | None (optional scheduler determinism)
  6. `reconstruct_caption_to_fuse` -> SeamEvent     (derive the CAPTION_TO_FUSE seam)
  7. `reconstruct_decide_to_act`   -> SeamEvent     (derive the DECIDE_TO_ACT seam)

The two `reconstruct_*` hooks are not optional extras: the four-seam producer
(`RecordingCoordinator`, §4) needs them to fill in the two seams that have no model
call of their own, so a full adapter must implement all seven. Use
`plumbline.adapters.conformance.assert_conforms` to check an implementation.

These types live outside `core/` (they are not frozen): they are the adapter
surface, expected to evolve as more runtimes are integrated.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize

# BusSample is a transport concept; it lives in transport/ to keep the tap from
# importing upward into adapters. Re-exported here so `from plumbline.adapters.base
# import BusSample` still resolves (it is part of the adapter-facing bus surface).
from plumbline.transport.bus import BusSample

__all__ = [
    "Action",
    "ActionSchema",
    "Adapter",
    "BusSample",
    "BusTap",
    "ClockHook",
    "ProxyConfig",
    "derived_seam_event",
]


def derived_seam_event(
    *,
    seam: Seam,
    episode_id: str,
    seq: int,
    logical_tick: int,
    request: Payload,
    response: Payload,
    wall_ts: float = 0.0,
) -> SeamEvent:
    """Build a RECONSTRUCTED (no-model-call) SeamEvent — the shared boilerplate for
    the CAPTION_TO_FUSE and DECIDE_TO_ACT seams an adapter derives from already-
    captured payloads (§9). `model_id=None`, `params={}`, digest over the request.

    This is a helper INSIDE `adapters/` — it touches no frozen `core/` interface, so
    it is not the cross-boundary refactor invariant 6 forbids; it just removes the
    ~40 lines of identical SeamEvent construction each adapter used to copy. Each
    adapter keeps its own semantic mapping (what goes in `request`/`response`) and
    calls this for the construction. A pure function of its inputs, so faithful
    replay of the derived seam stays byte-identical."""
    return SeamEvent(
        episode_id=episode_id,
        seq=seq,
        seam=seam,
        logical_tick=logical_tick,
        wall_ts=wall_ts,
        request=request,
        response=response,
        model_id=None,
        params={},
        request_digest=canonicalize(request).digest,
        latency_ms=0.0,
    )


@dataclass(frozen=True)
class ProxyConfig:
    """How to point a runtime's model clients at the Plumbline proxy (§9.1).

    Purely declarative: `env` and `config_fields` are settings the operator
    applies externally (environment / config file). No runtime source changes.
    """

    proxy_base_url: str
    env: Mapping[str, str]  # env var name -> value
    config_fields: Mapping[str, str]  # config field path -> value


@dataclass(frozen=True)
class Action:
    """One elemental command in an action plan, typed for comparison (§9.2)."""

    kind: str  # "move" | "skill" | "speak" | "express"
    name: str  # e.g. "move", "shake paw"
    args: Mapping[str, JSONValue]


@runtime_checkable
class BusTap(Protocol):
    """A passive subscriber to the runtime's action / data bus (§4.3, §9.1)."""

    @property
    def key_expressions(self) -> tuple[str, ...]: ...

    def subscribe(self, on_sample: Callable[[BusSample], None]) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class ActionSchema(Protocol):
    """The runtime's typed action plan, for behavioral comparison (§9.1)."""

    @property
    def commands(self) -> tuple[str, ...]: ...

    def parse(self, payload: Payload) -> tuple[Action, ...]: ...


@runtime_checkable
class ClockHook(Protocol):
    """Optional control over the runtime's loop clock for full determinism (§9.1).

    Returned by `Adapter.clock_hook` only when a runtime exposes such a hook.
    Absent it, Plumbline guarantees model-I/O determinism, not wall-clock
    scheduling (§3.4, §14.4).
    """

    def now_tick(self) -> int: ...

    def set_tick(self, tick: int) -> None: ...


@runtime_checkable
class Adapter(Protocol):
    """Wires a specific runtime's four seams into the substrate (§9.1).

    The complete, single contract: five wiring methods plus the two `reconstruct_*`
    hooks the four-seam producer (`RecordingCoordinator`, §4) calls to derive the
    CAPTION_TO_FUSE and DECIDE_TO_ACT seams (no model call of their own). All seven
    are required — a partial implementation is not a working four-seam producer.

    `@runtime_checkable`, so `isinstance(x, Adapter)` verifies method presence; use
    `plumbline.adapters.conformance.assert_conforms` for a deeper structural check.
    """

    def configure_proxy(self) -> ProxyConfig: ...

    def bus_tap(self) -> BusTap | None: ...

    def seam_of(self, request: Payload, endpoint: str) -> Seam: ...

    def action_schema(self) -> ActionSchema: ...

    def clock_hook(self) -> ClockHook | None: ...

    def reconstruct_caption_to_fuse(
        self,
        *,
        episode_id: str,
        seq: int,
        logical_tick: int,
        captions: Sequence[JSONValue],
        fused_prompt: JSONValue,
        wall_ts: float = 0.0,
    ) -> SeamEvent:
        """Derive the CAPTION_TO_FUSE seam (no model call) from a tick's captions and
        the fused prompt they feed. Typically `derived_seam_event(...)`."""
        ...

    def reconstruct_decide_to_act(
        self,
        *,
        episode_id: str,
        seq: int,
        logical_tick: int,
        decision_response: Payload,
        wall_ts: float = 0.0,
    ) -> SeamEvent:
        """Derive the DECIDE_TO_ACT seam (no model call) from the recorded decision
        response — a pure function of it, so faithful replay stays byte-identical."""
        ...
