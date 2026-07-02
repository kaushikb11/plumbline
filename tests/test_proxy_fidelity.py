"""Proxy fidelity / zero-touch invariant (eng spec §15, §4.2; WS2 done-criterion).

In record mode the recording proxy must forward to the upstream and return the
upstream response to the runtime *unaltered*, byte-for-byte, while recording a
matching SeamEvent. Recording must be observable-equivalent to no proxy at all.

The Protocol and factory signature below pin the expected contract (interpreted
from §4.2, which specifies behavior but no Python API — flagged).
"""

from collections.abc import Callable
from typing import Protocol, cast

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.core.trace import Payload

from tests.toyloop import canonical_bytes, load_unimplemented

UpstreamFn = Callable[[Payload], Payload]


class _RecordingProxy(Protocol):
    """Record-mode proxy: forward to upstream, record, return response unchanged."""

    def forward(self, request: Payload, ctx: Context) -> Payload: ...


# NOTE: factory shape interpreted from §4.2 — `RecordingProxy(upstream, recorder)`
# constructs a record-mode proxy wrapping the real upstream call and the recorder.
_ProxyFactory = Callable[[UpstreamFn, Recorder], _RecordingProxy]


def test_proxy_is_zero_touch_in_record_mode() -> None:
    upstream_response = Payload(inline={"action_plan": {"action": "avoid", "args": {}}})
    upstream_requests: list[Payload] = []

    def upstream(request: Payload) -> Payload:
        upstream_requests.append(request)
        return upstream_response

    make_proxy = cast(
        _ProxyFactory,
        load_unimplemented("plumbline.proxy", "RecordingProxy"),  # AttributeError until WS2 lands
    )
    store = TraceStore()
    clock = VirtualClock()
    recorder = Recorder(store, clock)
    proxy = make_proxy(upstream, recorder)

    ctx = Context(episode_id="ep-proxy", model_id="stub/decider-v1", params={"temperature": 0.7})
    request = Payload(inline={"prompt": "Observations: obstacle 0.30 m. Decide the next action."})
    returned = proxy.forward(request, ctx)

    # Zero-touch: the runtime receives exactly what the upstream returned.
    assert returned == upstream_response
    assert canonical_bytes(returned) == canonical_bytes(upstream_response)
    # The proxy forwarded exactly once, unaltered.
    assert len(upstream_requests) == 1
    assert upstream_requests[0] == request

    # And it recorded a faithful event for the call.
    recorded = store.load_episode("ep-proxy").events
    assert any(
        event.request == request and event.response == upstream_response for event in recorded
    )
