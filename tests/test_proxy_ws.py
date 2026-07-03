"""WebSocket caption capture (§4.2; limitations gap #1).

Fake WsTransport + WsConnection (no ASGI, no `websockets`): the upstream scripts a
caption stream, the client blocks (server-pushed captions), so the record run is
deterministic — only inbound frames are captured.
"""

import asyncio
from collections.abc import Mapping, Sequence

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.proxy.ws import AsyncWSProxy, WsConnection, WsFrame, _NoUpstreamWsTransport

_URL = "wss://api.openmind.com"
_ENDPOINT = "/ws/captions"
_CTX = Context(episode_id="ws-1", model_id=None, params={}, logical_tick=0)
_CAPTIONS = (
    WsFrame(kind="text", text='{"caption": "a person is ahead"}'),
    WsFrame(kind="text", text="an obstacle to the left"),
    WsFrame(kind="bytes", data=b"\x01\x02\x03raw"),
)


class _ScriptedUpstream:
    def __init__(self, frames: Sequence[WsFrame]) -> None:
        self._frames = list(frames)
        self.sent: list[WsFrame] = []
        self.closed = False

    async def send(self, frame: WsFrame) -> None:
        self.sent.append(frame)

    async def recv(self) -> WsFrame:
        return self._frames.pop(0) if self._frames else WsFrame(kind="close", code=1000)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class _CollectingClient:
    def __init__(self, outbound: Sequence[WsFrame] = ()) -> None:
        self._outbound = list(outbound)
        self.received: list[WsFrame] = []
        self.closed = False

    async def send(self, frame: WsFrame) -> None:
        self.received.append(frame)

    async def recv(self) -> WsFrame:
        if self._outbound:
            return self._outbound.pop(0)
        await asyncio.Event().wait()  # server-pushed stream: the client sends nothing
        raise AssertionError("unreachable")

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class _FakeTransport:
    def __init__(self, upstream: WsConnection) -> None:
        self._upstream = upstream

    async def connect(
        self, url: str, *, subprotocols: Sequence[str], headers: Mapping[str, str]
    ) -> WsConnection:
        return self._upstream


def _record(store: TraceStore) -> _CollectingClient:
    recorder = Recorder(store, VirtualClock())
    client = _CollectingClient()
    proxy = AsyncWSProxy(
        transport=_FakeTransport(_ScriptedUpstream(_CAPTIONS)), recorder=recorder, store=store
    )
    asyncio.run(proxy.record(client, _CTX, upstream_url=_URL, endpoint=_ENDPOINT))
    recorder.close_episode(_CTX.episode_id)
    return client


def test_records_captions_as_sensor_to_caption() -> None:
    store = TraceStore()
    _record(store)
    events = store.load_episode(_CTX.episode_id).events
    assert len(events) == 3
    assert all(event.seam is Seam.SENSOR_TO_CAPTION for event in events)
    assert [event.seq for event in events] == [0, 1, 2]
    assert len({event.request_digest for event in events}) == 3  # unique, never collapsed


def test_zero_touch_client_receives_upstream_frames_unaltered() -> None:
    store = TraceStore()
    client = _record(store)
    assert [frame.kind for frame in client.received] == ["text", "text", "bytes", "close"]
    assert client.received[0].text == '{"caption": "a person is ahead"}'
    assert client.received[2].data == b"\x01\x02\x03raw"


def test_binary_frame_via_blob_no_pickle() -> None:
    store = TraceStore()
    _record(store)
    binary = store.load_episode(_CTX.episode_id).events[2]
    assert binary.response.blobs  # stored as a content-addressed blob, not inlined
    assert store.get_blob(binary.response.blobs[0]) == b"\x01\x02\x03raw"


def test_faithful_replay_without_upstream() -> None:
    store = TraceStore()
    _record(store)
    proxy = AsyncWSProxy(
        transport=_NoUpstreamWsTransport(), recorder=Recorder(store, VirtualClock()), store=store
    )
    client = _CollectingClient()
    asyncio.run(proxy.replay(client, _CTX, endpoint=_ENDPOINT))
    # The three recorded captions are served in seq order; the connection ends via close().
    assert [frame.kind for frame in client.received] == ["text", "text", "bytes"]
    assert client.received[2].data == b"\x01\x02\x03raw"  # reconstructed from the blob
    assert client.closed


# --- ASGI websocket server (make_ws_asgi_app / make_ws_replay_asgi_app) ------

from typing import Any  # noqa: E402

from plumbline.proxy.server import ASGIApp, make_ws_asgi_app, make_ws_replay_asgi_app  # noqa: E402


async def _drive(app: ASGIApp, receive: Any, send: Any) -> None:
    await app({"type": "websocket", "path": _ENDPOINT}, receive, send)


def _asgi_driver() -> tuple[list[dict[str, Any]], Any]:
    """A fake ASGI websocket channel: receive yields the connect handshake then
    blocks (the runtime pushes nothing); send collects outgoing messages."""
    sent: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = [{"type": "websocket.connect"}]

    async def receive() -> dict[str, Any]:
        if queue:
            return queue.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    return sent, (receive, send)


def test_ws_asgi_app_records_and_relays() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    proxy = AsyncWSProxy(
        transport=_FakeTransport(_ScriptedUpstream(_CAPTIONS)), recorder=recorder, store=store
    )
    app = make_ws_asgi_app(proxy, upstream=_URL, episode_id=_CTX.episode_id)
    sent, (receive, send) = _asgi_driver()
    asyncio.run(_drive(app, receive, send))
    recorder.close_episode(_CTX.episode_id)

    assert [m["type"] for m in sent] == [
        "websocket.accept",
        "websocket.send",
        "websocket.send",
        "websocket.send",
        "websocket.close",
    ]
    assert sent[1]["text"] == '{"caption": "a person is ahead"}'  # zero-touch relay
    assert sent[3]["bytes"] == b"\x01\x02\x03raw"
    events = store.load_episode(_CTX.episode_id).events
    assert len(events) == 3 and all(e.seam is Seam.SENSOR_TO_CAPTION for e in events)


def test_ws_replay_asgi_app_serves_recorded_frames() -> None:
    store = TraceStore()
    _record(store)  # populate the episode
    app = make_ws_replay_asgi_app(store, episode_id=_CTX.episode_id)
    sent, (receive, send) = _asgi_driver()
    asyncio.run(_drive(app, receive, send))
    assert [m["type"] for m in sent] == [
        "websocket.accept",
        "websocket.send",
        "websocket.send",
        "websocket.send",
        "websocket.close",
    ]
    assert sent[3]["bytes"] == b"\x01\x02\x03raw"  # reconstructed from the blob
