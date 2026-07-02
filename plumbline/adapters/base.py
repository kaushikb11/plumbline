"""The adapter contract (engineering spec §9.1).

An adapter teaches Plumbline how to attach to a specific runtime. It is small by
design: five methods plus the small data types they return. The core knows about
seams, traces, and model calls; the adapter knows about one runtime's wiring.

These types live outside `core/` (they are not frozen): they are the adapter
surface, expected to evolve as more runtimes are integrated.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload


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


@dataclass(frozen=True)
class BusSample:
    """One message observed on the runtime's bus (e.g. a Zenoh sample)."""

    key_expr: str
    payload: JSONValue
    wall_ts: float


class BusTap(Protocol):
    """A passive subscriber to the runtime's action / data bus (§4.3, §9.1)."""

    @property
    def key_expressions(self) -> tuple[str, ...]: ...

    def subscribe(self, on_sample: Callable[[BusSample], None]) -> None: ...

    def close(self) -> None: ...


class ActionSchema(Protocol):
    """The runtime's typed action plan, for behavioral comparison (§9.1)."""

    @property
    def commands(self) -> tuple[str, ...]: ...

    def parse(self, payload: Payload) -> tuple[Action, ...]: ...


class ClockHook(Protocol):
    """Optional control over the runtime's loop clock for full determinism (§9.1).

    Returned by `Adapter.clock_hook` only when a runtime exposes such a hook.
    Absent it, Plumbline guarantees model-I/O determinism, not wall-clock
    scheduling (§3.4, §14.4).
    """

    def now_tick(self) -> int: ...

    def set_tick(self, tick: int) -> None: ...


class Adapter(Protocol):
    """Wires a specific runtime's seams into the substrate (§9.1)."""

    def configure_proxy(self) -> ProxyConfig: ...

    def bus_tap(self) -> BusTap | None: ...

    def seam_of(self, request: Payload, endpoint: str) -> Seam: ...

    def action_schema(self) -> ActionSchema: ...

    def clock_hook(self) -> ClockHook | None: ...
