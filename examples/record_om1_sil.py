"""Record a real OM1 episode, software-in-the-loop — no sim, no robot (WS5).

The SIL middle path from docs/limitations.md: run the real OM1 Go binary with a
headless config (no perception services, `agent_inputs: []`) whose Cortex points at
the Plumbline recording proxy, and whose only action is `Move` -> CDR `Twist` over
Zenoh `cmd_vel`, where the Plumbline tap is the sole subscriber (the stubbed HAL).

One process hosts everything the runbook (docs/record-om1-gazebo.md) splits across
snippets: the recording ASGI proxy (uvicorn), the Zenoh `cmd_vel` tap, and a
`RecordingCoordinator` sharing one episode across both producers.

    PLUMBLINE_UPSTREAM=https://<llm endpoint> python examples/record_om1_sil.py &
    (cd <OM1 checkout> && make run CONFIG=plumbline_sil)

Env: PLUMBLINE_UPSTREAM (required), PLUMBLINE_STORE (default ./traces-sil),
PLUMBLINE_EPISODE (default om1-sil-001), PLUMBLINE_DURATION seconds (default 60),
PLUMBLINE_PORT (default 8900). Needs `plumbline[proxy,zenoh]`.

The Cortex loop is a pure decide loop (no VLM call), so each Cortex call is one
tick (PerCallTickPolicy; see proxy/tick.py).
"""

import asyncio
import os
import struct
import time
from collections import Counter

from plumbline.adapters.om1 import OM1Adapter
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.proxy.http import AsyncHTTPProxy
from plumbline.proxy.tick import PerCallTickPolicy
from plumbline.recording import RecordingCoordinator


def _cdr_pose_stamped(x: float = 1.0, y: float = 0.5, z: float = 0.30) -> bytes:
    """A CDR-LE PoseStamped-shaped odom message matching OM1's deserializePoseStamped
    (internal/providers/unitree/go2/odom_zenoh.go): encapsulation header, stamp,
    frame_id, child_frame_id (CDR-aligned), then position xyz + orientation xyzw.

    A STATIC pose with x != 0 and z = 0.30 m reads as "standing, not moving, located"
    — exactly what the Move connector requires before it will publish cmd_vel. This is
    the stubbed HAL's other half: fake odometry in, real cmd_vel out.
    """

    def _pad(buf: bytearray, align: int) -> None:
        data_off = len(buf) - 4  # CDR alignment is relative to the data start
        buf.extend(b"\x00" * ((align - data_off % align) % align))

    now = time.time()
    buf = bytearray(b"\x00\x01\x00\x00")  # CDR little-endian encapsulation
    buf += struct.pack("<iI", int(now), int((now % 1) * 1e9))  # stamp sec/nanosec
    for name, align in ((b"odom\x00", 4), (b"base\x00", 8)):
        buf += struct.pack("<I", len(name)) + name  # length-prefixed, NUL-terminated
        _pad(buf, align)
    buf += struct.pack("<7d", x, y, z, 0.0, 0.0, 0.0, 1.0)  # pose + identity quat
    return bytes(buf)


def _cdr_paths(indices: tuple[int, ...] = tuple(range(10))) -> bytes:
    """A CDR-LE om/paths message (internal/providers/paths.go deserializePaths):
    header, stamp, frame_id, then sequence<uint32> of safe path indices. All ten
    indices (0-9) = every direction is safe — without this the Move connector bars
    every command ("cannot advance due to barrier")."""
    now = time.time()
    buf = bytearray(b"\x00\x01\x00\x00")
    buf += struct.pack("<iI", int(now), int((now % 1) * 1e9))
    name = b"paths\x00"
    buf += struct.pack("<I", len(name)) + name
    data_off = len(buf) - 4
    buf += b"\x00" * ((4 - data_off % 4) % 4)
    buf += struct.pack("<I", len(indices)) + struct.pack(f"<{len(indices)}I", *indices)
    return bytes(buf)


async def run() -> None:
    import httpx
    import uvicorn
    import zenoh
    from plumbline.proxy.server import HttpxTransport, make_asgi_app
    from plumbline.transport.zenoh_shim import ZenohSessionAdapter

    upstream = os.environ["PLUMBLINE_UPSTREAM"].rstrip("/")
    store_root = os.environ.get("PLUMBLINE_STORE", "./traces-sil")
    episode_id = os.environ.get("PLUMBLINE_EPISODE", "om1-sil-001")
    duration = float(os.environ.get("PLUMBLINE_DURATION", "60"))
    port = int(os.environ.get("PLUMBLINE_PORT", "8900"))

    store = TraceStore(root=store_root)
    raw_session = zenoh.open(zenoh.Config())
    adapter = OM1Adapter(
        proxy_base_url=f"http://127.0.0.1:{port}",
        zenoh_session=ZenohSessionAdapter(raw_session),
    )
    coordinator = RecordingCoordinator(store, episode_id=episode_id, adapter=adapter)
    coordinator.open()

    tap = adapter.bus_tap()
    assert tap is not None
    tap.subscribe(coordinator.record_bus_sample)

    # The stubbed HAL's sensing: static odometry ("standing at (1.0, 0.5)") and an
    # all-clear path map at 10 Hz. Without odom the Move connector holds every
    # command ("waiting for location data"); without paths it bars every direction
    # ("cannot advance due to barrier"). With both, it publishes real CDR Twist
    # frames on cmd_vel — which only the Plumbline tap hears.
    odom_pub = raw_session.declare_publisher("odom")
    paths_pub = raw_session.declare_publisher("om/paths")

    async def publish_hal_stub() -> None:
        while True:
            odom_pub.put(_cdr_pose_stamped())
            paths_pub.put(_cdr_paths())
            await asyncio.sleep(0.1)

    odom_task = asyncio.create_task(publish_hal_stub())

    proxy = AsyncHTTPProxy(
        transport=HttpxTransport(httpx.AsyncClient(timeout=120.0)),
        recorder=coordinator,
        store=store,
        # Pure decide loop: each Cortex call is one tick (BoundaryTickPolicy's
        # consecutive-boundary-share rule would collapse the whole run to tick 0).
        tick_policy=PerCallTickPolicy(boundary_seam=Seam.FUSE_TO_DECIDE),
    )
    app = make_asgi_app(proxy, upstream=upstream, episode_id=episode_id)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serve_task = asyncio.create_task(server.serve())
    print(f"recording {upstream} -> episode {episode_id!r} at {store.root}")
    print(f"proxy on 127.0.0.1:{port}; zenoh tap on {adapter.action_key_expressions}")
    print(f"start OM1 now (make run CONFIG=plumbline_sil); recording for {duration:.0f}s ...")

    await asyncio.sleep(duration)
    odom_task.cancel()
    server.should_exit = True
    await serve_task
    tap.close()
    coordinator.close()

    events = store.load_episode(episode_id).events
    by_seam = Counter(event.seam.value for event in events)
    ticks = len({event.logical_tick for event in events})
    print(f"\nrecorded {len(events)} seam events over {ticks} ticks: {dict(by_seam)}")


if __name__ == "__main__":
    asyncio.run(run())
