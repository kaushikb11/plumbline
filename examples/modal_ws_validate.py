"""Validate WebSocket caption capture against a REAL remote WS server (Tier 2).

Deploy modal/ws_captions.py, then:

    PLUMBLINE_WS_URL=wss://<ws>--plumbline-ws-captions-serve.modal.run/ws/captions \\
    python examples/modal_ws_validate.py

It dials the real `wss://` caption stream through `AsyncWSProxy` +
`WebsocketsTransport` (the exact production path of the WS record proxy), records
each inbound caption frame as a SENSOR_TO_CAPTION event while relaying it to the
client UNALTERED, then faithful-replays with NO upstream and asserts the served
frame sequence is identical to what the client saw live — closing the WebSocket
half of limitations gap #1 against a real remote server instead of fakes.

Needs websockets: pip install "plumbline[proxy]".
"""

import asyncio
import os

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.proxy.ws import AsyncWSProxy, WsFrame

_ENDPOINT = "/ws/captions"


class _CollectingClient:
    """The runtime side of the proxy: collects every frame relayed to it. Its recv
    blocks forever (the caption stream is server-push; the proxy cancels the
    client->upstream task when the upstream closes)."""

    def __init__(self) -> None:
        self.frames: list[WsFrame] = []
        self.closed: int | None = None

    async def send(self, frame: WsFrame) -> None:
        if frame.kind != "close":
            self.frames.append(frame)

    async def recv(self) -> WsFrame:
        await asyncio.Future()  # server-push stream: the client never sends
        raise AssertionError("unreachable")

    async def close(self, code: int = 1000) -> None:
        self.closed = code


async def record(store: TraceStore, url: str, episode_id: str) -> list[WsFrame]:
    from plumbline.proxy.server import WebsocketsTransport

    proxy = AsyncWSProxy(
        transport=WebsocketsTransport(), recorder=Recorder(store, VirtualClock()), store=store
    )
    client = _CollectingClient()
    ctx = Context(episode_id=episode_id, model_id=None, params={}, logical_tick=0)
    await proxy.record(client, ctx, upstream_url=url, endpoint=_ENDPOINT)
    return client.frames


async def replay(store: TraceStore, episode_id: str) -> list[WsFrame]:
    from plumbline.proxy.ws import _NoUpstreamWsTransport  # replay never dials out

    proxy = AsyncWSProxy(
        transport=_NoUpstreamWsTransport(), recorder=Recorder(store, VirtualClock()), store=store
    )
    client = _CollectingClient()
    ctx = Context(episode_id=episode_id, model_id=None, params={}, logical_tick=0)
    await proxy.replay(client, ctx, endpoint=_ENDPOINT)
    return client.frames


def main() -> None:
    url = os.environ["PLUMBLINE_WS_URL"]
    store = TraceStore()
    episode_id = "modal-ws-validate"

    live = asyncio.run(record(store, url, episode_id))
    events = store.load_episode(episode_id).events
    print(f"recorded {len(events)} SENSOR_TO_CAPTION frames from the live WS stream")
    for frame in live:
        print(f"  live: {frame.text}")

    served = asyncio.run(replay(store, episode_id))
    identical = [(f.kind, f.text, f.data) for f in live] == [
        (f.kind, f.text, f.data) for f in served
    ]
    print(f"replayed {len(served)} frames with no upstream")
    print(f"replay identical to live stream: {'PASS' if identical else 'FAIL'}")
    raise SystemExit(0 if identical else 1)


if __name__ == "__main__":
    main()
