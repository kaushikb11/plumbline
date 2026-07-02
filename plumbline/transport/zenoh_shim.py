"""Adapt a real eclipse-zenoh Session to the injected ZenohSession Protocol (§4.3).

This module deliberately does NOT `import zenoh`. The caller opens the real
session — `import zenoh; session = zenoh.open(config)` — and passes it in, so the
substrate carries no zenoh dependency and no version pin (zenoh-python is a Rust
extension and its API has churned across 0.10 / 0.11 / 1.0). The adapter bridges
the two API shapes that have stayed stable in spirit:

  - a session exposing `declare_subscriber(key_expr, handler)` and `close()`;
  - samples exposing a key expression and a byte payload.

Payload extraction is defensive because zenoh-python has exposed sample bytes
several ways (raw bytes, a `ZBytes` with `.to_bytes()`, or a buffer). Usage:

    import zenoh
    from plumbline.transport.zenoh_shim import ZenohSessionAdapter
    from plumbline.transport.zenoh_tap import ZenohTap

    session = zenoh.open(zenoh.Config())
    tap = ZenohTap(ZenohSessionAdapter(session), ("om1/agent/actions/**",))

Install the real client with:  pip install "plumbline[zenoh]".
"""

from collections.abc import Callable
from typing import Any

from plumbline.transport.zenoh_tap import ZenohSample


class _SampleAdapter:
    """Wraps a real zenoh Sample as a `ZenohSample` (str key_expr, bytes payload)."""

    def __init__(self, sample: Any) -> None:
        self._sample = sample

    @property
    def key_expr(self) -> str:
        return str(self._sample.key_expr)

    @property
    def payload(self) -> bytes:
        return _payload_bytes(self._sample.payload)


class ZenohSessionAdapter:
    """Adapts a real zenoh Session to the `ZenohSession` Protocol (§4.3).

    Structurally satisfies `ZenohSession`, so it drops straight into `ZenohTap`
    and the OM1 adapter's `bus_tap()`.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    def declare_subscriber(self, key_expr: str, handler: Callable[[ZenohSample], None]) -> object:
        def on_zenoh_sample(sample: Any) -> None:
            handler(_SampleAdapter(sample))

        subscriber: object = self._session.declare_subscriber(key_expr, on_zenoh_sample)
        return subscriber

    def close(self) -> None:
        self._session.close()


def _payload_bytes(payload: Any) -> bytes:
    """Extract raw bytes from a zenoh sample payload across API versions."""
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    to_bytes = getattr(payload, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    return bytes(payload)  # buffer-protocol / iterable-of-ints fallback
