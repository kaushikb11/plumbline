"""The real-Zenoh shim (engineering spec §4.3, §9.2).

Verifies that `ZenohSessionAdapter` bridges the real zenoh Session/Sample shape
(a `KeyExpr` that stringifies, a `ZBytes`-like payload) to the `str`/`bytes` the
tap's Protocols expect — without importing zenoh. The fakes below mimic the real
API shape so the adapter is exercised exactly as a live session would drive it.
"""

import json
from collections.abc import Callable

from plumbline.adapters.base import BusSample
from plumbline.transport.zenoh_shim import ZenohSessionAdapter, _payload_bytes
from plumbline.transport.zenoh_tap import ZenohTap

# --- fakes mimicking the real zenoh API shape (KeyExpr / ZBytes / Session) ---


class _FakeKeyExpr:
    def __init__(self, key: str) -> None:
        self._key = key

    def __str__(self) -> str:
        return self._key


class _FakeZBytes:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def to_bytes(self) -> bytes:
        return self._data


class _FakeSample:
    def __init__(self, key: str, data: bytes) -> None:
        self.key_expr = _FakeKeyExpr(key)  # a KeyExpr, not a str
        self.payload = _FakeZBytes(data)  # a ZBytes, not bytes


class _FakeRealSession:
    def __init__(self) -> None:
        self._subscribers: list[tuple[str, Callable[[_FakeSample], None]]] = []
        self.closed = False

    def declare_subscriber(self, key_expr: str, handler: Callable[[_FakeSample], None]) -> object:
        self._subscribers.append((key_expr, handler))
        return object()

    def close(self) -> None:
        self.closed = True

    def publish(self, key: str, data: bytes) -> None:
        for pattern, handler in self._subscribers:
            if _matches(pattern, key):
                handler(_FakeSample(key, data))


def _matches(pattern: str, key: str) -> bool:
    return key.startswith(pattern[:-2]) if pattern.endswith("**") else pattern == key


def test_shim_adapts_real_sample_shape_into_bus_samples() -> None:
    session = _FakeRealSession()
    tap = ZenohTap(ZenohSessionAdapter(session), ("om1/agent/actions/**",))

    received: list[BusSample] = []
    tap.subscribe(received.append)

    action_plan = {"commands": [{"type": "move", "x": 0.3}]}
    session.publish("om1/agent/actions/go2", json.dumps(action_plan).encode("utf-8"))

    assert len(received) == 1
    assert received[0].key_expr == "om1/agent/actions/go2"  # KeyExpr -> str
    assert received[0].payload == action_plan  # ZBytes -> bytes -> parsed JSON

    tap.close()
    assert session.closed is True


def test_payload_bytes_handles_zenoh_payload_variants() -> None:
    assert _payload_bytes(b"raw") == b"raw"  # already bytes
    assert _payload_bytes(bytearray(b"buf")) == b"buf"  # bytearray
    assert _payload_bytes(_FakeZBytes(b"zb")) == b"zb"  # ZBytes.to_bytes()
