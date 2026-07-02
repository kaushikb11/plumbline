"""Zenoh bus tap (engineering spec §4.3, §9.2).

A passive Zenoh subscriber on the runtime's action key expressions and natural-
language data-bus topics. It records what crosses the bus (action plans, HAL
commands) for the DECIDE_TO_ACT seam; it is observe-only in record mode.

The actual Zenoh session is *injected* as a `ZenohSession` Protocol, so the
substrate stays light and free of a hard `zenoh` dependency (the real
`zenoh.Session` / `zenoh.Sample` satisfy these Protocols, directly or via a thin
shim). This keeps the tap unit-testable with a fake session.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from plumbline.adapters.base import BusSample
from plumbline.core.trace import JSONValue


class ZenohSample(Protocol):
    @property
    def key_expr(self) -> str: ...

    @property
    def payload(self) -> bytes: ...


class ZenohSession(Protocol):
    def declare_subscriber(
        self, key_expr: str, handler: Callable[[ZenohSample], None]
    ) -> object: ...

    def close(self) -> None: ...


@dataclass
class ZenohTap:
    """A BusTap backed by an injected Zenoh session (§4.3)."""

    session: ZenohSession
    key_expressions: tuple[str, ...]
    _subscribers: list[object] = field(default_factory=list, init=False, repr=False)

    def subscribe(self, on_sample: Callable[[BusSample], None]) -> None:
        for key_expr in self.key_expressions:
            self._subscribers.append(
                self.session.declare_subscriber(key_expr, self._make_handler(on_sample))
            )

    def close(self) -> None:
        self.session.close()

    def _make_handler(
        self, on_sample: Callable[[BusSample], None]
    ) -> Callable[[ZenohSample], None]:
        def handle(sample: ZenohSample) -> None:
            on_sample(self._to_bus_sample(sample))

        return handle

    def _to_bus_sample(self, sample: ZenohSample) -> BusSample:
        # Action/HAL commands on the bus are often binary (CBOR/protobuf) or plain
        # text, not JSON. Fall back to a decoded string rather than crashing the
        # Zenoh subscriber thread (which would drop the sample and can kill the sub).
        payload: JSONValue
        if not sample.payload:
            payload = None
        else:
            try:
                payload = json.loads(sample.payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {"plumbline.raw_bus": bytes(sample.payload).decode("utf-8", "replace")}
        return BusSample(key_expr=sample.key_expr, payload=payload, wall_ts=time.time())
