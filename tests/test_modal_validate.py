"""Offline check of the Modal validation driver (no network): proves the record ->
faithful-replay byte-identical logic with a fake transport, so the harness is
CI-verified even though the real run points at Modal."""

import asyncio
import json

import examples.modal_validate as mv
from plumbline.core.store import TraceStore
from plumbline.proxy.http import HTTPRequest, HTTPResponse
from plumbline.proxy.normalizers import contains_image


class _FakeTransport:
    async def send(self, request: HTTPRequest) -> HTTPResponse:
        body = json.loads(request.body)
        payload: dict[str, object]
        if contains_image(body):
            payload = {"choices": [{"message": {"content": "a clear corridor ahead"}}]}
        else:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "Move",
                                        "arguments": '{"action": "move forwards"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        return HTTPResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
        )


def test_record_and_faithful_replay_byte_identical() -> None:
    store = TraceStore()
    episode_id = asyncio.run(
        mv.record_episode(
            _FakeTransport(), store, vlm_url="http://vlm", llm_url="http://llm", ticks=3
        )
    )
    events = store.load_episode(episode_id).events
    assert len(events) == 6  # 3 ticks x (vision + cortex)
    # The three identical vision requests share a digest; faithful replay serves them
    # in record order (the digest-occurrence fix) — byte-identical throughout.
    assert mv.verify_faithful_replay(store, episode_id) is True


def test_main_is_callable() -> None:
    assert callable(mv.main)
