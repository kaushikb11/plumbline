"""The recording session coordinator (engineering spec §3.5, §4, §6, §9).

Records all four seams of a two-tick Go2 run into ONE episode through a single
session: the model seams via `RecordingProxy` (the session is its recorder), the
action seam via a `ZenohTap`, and CAPTION_TO_FUSE via the OM1 adapter's
reconstruction. Asserts the merged trace is coherent: globally-monotonic `seq`
and each loop iteration's four seams sharing a `logical_tick`.
"""

import json
from collections.abc import Callable

from plumbline.adapters.base import BusSample
from plumbline.adapters.om1 import OM1Adapter
from plumbline.core.interceptor import Context
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload
from plumbline.proxy import RecordingProxy
from plumbline.session import RecordingSession
from plumbline.transport.zenoh_tap import ZenohSample, ZenohTap

_SCENES = ("human", "obstacle")
_ACTION: dict[str, JSONValue] = {
    "human": {"commands": [{"type": "move", "x": 0.3}]},
    "obstacle": {"commands": [{"type": "move", "y": 0.2}]},
}


# --- a fake ZenohSession (Protocol) driving the tap synchronously ------------


class _FakeZenohSession:
    def __init__(self) -> None:
        self._subscribers: list[tuple[str, Callable[[ZenohSample], None]]] = []

    def declare_subscriber(self, key_expr: str, handler: Callable[[ZenohSample], None]) -> object:
        self._subscribers.append((key_expr, handler))
        return object()

    def close(self) -> None:
        pass

    def publish(self, key: str, data: bytes) -> None:
        for pattern, handler in self._subscribers:
            if key.startswith(pattern[:-2]):
                handler(_FakeSample(key, data))


class _FakeSample:
    def __init__(self, key: str, data: bytes) -> None:
        self.key_expr = key
        self.payload = data


def _model_call(request: Payload) -> Payload:
    inline = request.inline
    assert isinstance(inline, dict)
    if inline.get("kind") == "caption":
        return Payload(inline={"caption": f"a {inline['scene']} is ahead"})
    return Payload(inline={"action_plan": {"action": "avoid"}})


def _classify(request: Payload, ctx: Context) -> Seam:
    inline = request.inline
    if isinstance(inline, dict) and inline.get("kind") == "caption":
        return Seam.SENSOR_TO_CAPTION
    return Seam.FUSE_TO_DECIDE


def test_session_merges_four_seams_into_one_coherent_episode() -> None:
    store = TraceStore()
    adapter = OM1Adapter(proxy_base_url="http://localhost:8900", zenoh_session=_FakeZenohSession())
    session = RecordingSession(store, episode_id="go2-001", metadata={"robot": "go2"})
    session.open()

    proxy = RecordingProxy(_model_call, session, classifier=_classify)

    zenoh = _FakeZenohSession()
    tap = ZenohTap(zenoh, ("om1/agent/actions/**",))
    tap.subscribe(session.record_bus_sample)

    for tick, scene in enumerate(_SCENES):
        session.set_tick(tick)
        # SENSOR_TO_CAPTION (model seam via the proxy)
        caption = proxy.forward(
            Payload(inline={"kind": "caption", "scene": scene}),
            session.context(model_id="openai/vlm"),
        )
        caption_text = caption.inline
        assert isinstance(caption_text, dict)
        # CAPTION_TO_FUSE (reconstructed by the adapter, recorded via the session)
        session.record(
            adapter.reconstruct_caption_to_fuse(
                episode_id="go2-001",
                seq=0,
                logical_tick=session.logical_tick,
                captions=[caption_text["caption"]],
                fused_prompt={"prompt": caption_text["caption"]},
            )
        )
        # FUSE_TO_DECIDE (model seam via the proxy)
        proxy.forward(
            Payload(inline={"kind": "decide", "prompt": str(caption_text)}),
            session.context(model_id="openai/cortex"),
        )
        # DECIDE_TO_ACT (action plan observed on the bus -> tap -> session)
        zenoh.publish("om1/agent/actions/go2", json.dumps(_ACTION[scene]).encode("utf-8"))

    session.close()

    events = store.load_episode("go2-001").events
    assert len(events) == 4 * len(_SCENES)
    # Globally-monotonic, gap-free seq across proxy + tap + reconstruction.
    assert [event.seq for event in events] == list(range(len(events)))
    # Each loop iteration's four seams share a logical_tick.
    by_tick: dict[int, list[Seam]] = {}
    for event in events:
        by_tick.setdefault(event.logical_tick, []).append(event.seam)
    assert by_tick == {
        0: [Seam.SENSOR_TO_CAPTION, Seam.CAPTION_TO_FUSE, Seam.FUSE_TO_DECIDE, Seam.DECIDE_TO_ACT],
        1: [Seam.SENSOR_TO_CAPTION, Seam.CAPTION_TO_FUSE, Seam.FUSE_TO_DECIDE, Seam.DECIDE_TO_ACT],
    }
    # The action sequence is recoverable for the gate (DECIDE_TO_ACT requests).
    actions = [e.request.inline for e in events if e.seam is Seam.DECIDE_TO_ACT]
    assert actions == [_ACTION["human"], _ACTION["obstacle"]]


def test_session_lifecycle_is_idempotent() -> None:
    store = TraceStore()
    session = RecordingSession(store, episode_id="ep", metadata={})
    session.open()
    # A proxy auto-opening the same episode must not truncate it.
    session.open_episode("ep", {})
    session.record_bus_sample(_bus_sample())
    session.close()
    session.close_episode("ep")  # idempotent second close
    assert len(store.load_episode("ep").events) == 1


def _bus_sample() -> BusSample:
    return BusSample(key_expr="om1/agent/actions/go2", payload={"commands": []}, wall_ts=0.0)


def test_bus_sample_records_originating_key() -> None:
    # The key a sample arrived on is attribution (pinning the real cmd_vel key from
    # a recorded episode) — carried in non-digested params, found in the OM1 SIL run.
    store = TraceStore()
    session = RecordingSession(store, episode_id="ep", metadata={})
    session.open()
    session.record_bus_sample(_bus_sample())
    session.close()
    event = store.load_episode("ep").events[0]
    assert event.params["plumbline.bus_key"] == "om1/agent/actions/go2"


def test_bus_sample_raw_bytes_stored_content_addressed() -> None:
    # The exact wire bytes are the ground truth: stored as a BIN blob referenced
    # from the payload (so the digest covers them), while inline stays the decoded
    # comparison view. Closes the "lossy utf-8 stand-in" limitation.
    store = TraceStore()
    session = RecordingSession(store, episode_id="ep", metadata={})
    session.open()
    raw = b"\x00\x01\x00\x00" + bytes(range(48))
    session.record_bus_sample(
        BusSample(key_expr="cmd_vel", payload={"decoded": "view"}, wall_ts=0.0, raw=raw)
    )
    session.close()
    event = store.load_episode("ep").events[0]
    assert event.request.inline == {"decoded": "view"}
    assert event.request.blobs and store.get_blob(event.request.blobs[0]) == raw

    # Identical decoded views with different bytes must NOT collapse to one digest.
    other = RecordingSession(store, episode_id="ep2", metadata={})
    other.open()
    other.record_bus_sample(
        BusSample(key_expr="cmd_vel", payload={"decoded": "view"}, wall_ts=0.0, raw=raw + b"!")
    )
    other.close()
    assert store.load_episode("ep2").events[0].request_digest != event.request_digest


def test_om1_tap_decodes_cmd_vel_twist_and_keeps_raw_bytes() -> None:
    # The adapter-supplied decoder turns the CDR Twist into a typed comparison view
    # while the exact wire bytes land in a content-addressed blob.
    import struct

    store = TraceStore()
    session = RecordingSession(store, episode_id="ep", metadata={})
    session.open()
    zenoh = _FakeZenohSession()
    adapter = OM1Adapter(proxy_base_url="http://localhost:8900", zenoh_session=zenoh)
    tap = adapter.bus_tap()
    assert tap is not None
    tap.subscribe(session.record_bus_sample)

    raw = b"\x00\x01\x00\x00" + struct.pack("<6d", 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    zenoh.publish("cmd_vel", raw)
    session.close()

    event = store.load_episode("ep").events[0]
    inline = event.request.inline
    assert isinstance(inline, dict) and "geometry_msgs/Twist" in inline
    assert event.params["plumbline.bus_key"] == "cmd_vel"
    assert event.request.blobs and store.get_blob(event.request.blobs[0]) == raw


def test_bus_sample_tick_stamped_at_record_time_under_the_lock() -> None:
    # Tick and seq are assigned under one lock acquisition, so ticks are monotone
    # in seq order by construction (found on a real OM1 recording: a tap-thread
    # sample raced set_tick and landed with a stale tick later in seq order).
    store = TraceStore()
    session = RecordingSession(store, episode_id="ep", metadata={})
    session.open()
    session.set_tick(4)
    session.record_bus_sample(_bus_sample())
    session.set_tick(5)
    session.record_bus_sample(_bus_sample())
    session.close()
    events = store.load_episode("ep").events
    assert [(e.seq, e.logical_tick) for e in events] == [(0, 4), (1, 5)]
