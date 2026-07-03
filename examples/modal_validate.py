"""Validate Plumbline against REAL models on Modal — no robot (Tier 1).

Deploy modal/llm.py + modal/vlm.py, then:

    PLUMBLINE_VLM_URL=https://<ws>--plumbline-vlm-serve.modal.run \\
    PLUMBLINE_LLM_URL=https://<ws>--plumbline-llm-serve.modal.run \\
    python examples/modal_validate.py

It drives scenes through the recording proxy to the real Modal VLM + LLM at
temperature > 0, then faithful-replays and asserts BYTE-IDENTICAL model I/O — the
substrate's core claim, against real nondeterministic models instead of stubs. The
repeated identical vision request also exercises the digest-occurrence fix (a static
scene sampled twice records two distinct captions, replayed in order).

Needs httpx: pip install "plumbline[proxy]".
"""

import asyncio
import json
import os
from collections.abc import Sequence

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.core.trace import canonicalize
from plumbline.proxy.http import AsyncHTTPProxy, AsyncTransport, HTTPRequest, HTTPResponse
from plumbline.proxy.recording import ReplayingProxy
from plumbline.proxy.tick import BoundaryTickPolicy

_JSON_HEADERS = {"content-type": "application/json"}
# A 1x1 PNG data URL — a placeholder frame; supply real corridor images for real captions.
_FRAME = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "Move",
        "description": "Move the robot",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "turn left",
                        "turn right",
                        "move forwards",
                        "move back",
                        "stand still",
                    ],
                }
            },
            "required": ["action"],
        },
    },
}


def _vision_body() -> bytes:
    return json.dumps(
        {
            "model": "captioner",
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "In one sentence: is the path ahead blocked or clear?",
                        },
                        {"type": "image_url", "image_url": {"url": _FRAME}},
                    ],
                }
            ],
        }
    ).encode()


def _cortex_body(caption: str) -> bytes:
    return json.dumps(
        {
            "model": "cortex",
            "temperature": 0.7,
            "messages": [
                {"role": "user", "content": f"Observation: {caption}. Decide the next move."}
            ],
            "tools": [_MOVE_TOOL],
        }
    ).encode()


def _content(response: HTTPResponse) -> str:
    data = json.loads(response.body)
    try:
        return data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


async def record_episode(
    transport: AsyncTransport,
    store: TraceStore,
    *,
    vlm_url: str,
    llm_url: str,
    ticks: int,
    episode_id: str = "modal-validate",
) -> str:
    """Drive `ticks` perception→decision cycles through the recording proxy to the
    (real or fake) VLM + LLM. Returns the recorded episode id."""
    recorder = Recorder(store, VirtualClock())
    proxy = AsyncHTTPProxy(
        transport=transport, recorder=recorder, store=store, tick_policy=BoundaryTickPolicy()
    )
    for _ in range(ticks):
        ctx = Context(episode_id=episode_id, model_id=None, params={}, logical_tick=0)
        vision = await proxy.record(
            HTTPRequest("POST", f"{vlm_url}/v1/chat/completions", _JSON_HEADERS, _vision_body()),
            ctx,
        )
        caption = _content(vision)
        await proxy.record(
            HTTPRequest(
                "POST", f"{llm_url}/v1/chat/completions", _JSON_HEADERS, _cortex_body(caption)
            ),
            ctx,
        )
    recorder.close_episode(episode_id)
    return episode_id


def verify_faithful_replay(store: TraceStore, episode_id: str) -> bool:
    """Re-serve each recorded request via faithful replay and check the served
    response is byte-identical to the recorded one (repeated identical requests are
    served in record order)."""
    replay = ReplayingProxy(store, episode_id)
    ctx = Context(episode_id=episode_id, model_id=None, params={}, logical_tick=0)
    for event in store.load_episode(episode_id).events:
        served = replay.faithful(event.request, ctx)
        if canonicalize(served).digest != canonicalize(event.response).digest:
            return False
    return True


def main() -> None:
    vlm_url = os.environ["PLUMBLINE_VLM_URL"].rstrip("/")
    llm_url = os.environ["PLUMBLINE_LLM_URL"].rstrip("/")

    import httpx
    from plumbline.proxy.server import HttpxTransport

    transport = HttpxTransport(httpx.AsyncClient(timeout=900.0))  # covers GPU cold start
    store = TraceStore()
    episode_id = asyncio.run(
        record_episode(transport, store, vlm_url=vlm_url, llm_url=llm_url, ticks=3)
    )
    events: Sequence[object] = store.load_episode(episode_id).events
    print(f"recorded {len(events)} seam events over 3 ticks (real Modal models, temperature 0.7)")
    ok = verify_faithful_replay(store, episode_id)
    print(f"faithful replay byte-identical vs real models: {'PASS' if ok else 'FAIL'}")
    print("(the three identical vision requests also exercised the digest-occurrence fix)")


if __name__ == "__main__":
    main()
