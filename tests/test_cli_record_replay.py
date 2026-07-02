"""The `plumbline record` / `replay` proxy-server subcommands (spec §4.2, §11).

The replay ASGI app is driven in-process via httpx ASGITransport (no socket); the
CLI wiring is checked by stubbing the uvicorn runner (`cli._serve`) so no server
actually starts.
"""

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
from plumbline import cli
from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.store import TraceStore
from plumbline.proxy.http import AsyncHTTPProxy, HTTPRequest, HTTPResponse
from plumbline.proxy.server import ASGIApp, make_replay_asgi_app

_REQ = {"model": "gpt-4o", "messages": [{"role": "user", "content": "go?"}]}
_RESP = {"id": "r", "choices": [{"message": {"role": "assistant", "content": "avoid"}}]}


class _FakeTransport:
    def __init__(self, response: HTTPResponse) -> None:
        self._response = response

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        return self._response


def _recorded_store(root: Path) -> TraceStore:
    store = TraceStore(root=root)
    upstream = HTTPResponse(
        200, {"content-type": "application/json"}, json.dumps(_RESP).encode(), None
    )
    proxy = AsyncHTTPProxy(
        transport=_FakeTransport(upstream), recorder=Recorder(store, VirtualClock()), store=store
    )
    request = HTTPRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=json.dumps(_REQ).encode(),
    )
    asyncio.run(proxy.record(request, Context(episode_id="ep", model_id=None, params={})))
    proxy.close("ep")
    return store


async def _post(app: ASGIApp, body: Mapping[str, object]) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://replay"
    ) as client:
        return await client.post("/v1/chat/completions", json=body)


def test_replay_app_serves_recorded_response(tmp_path: Path) -> None:
    app = make_replay_asgi_app(_recorded_store(tmp_path / "traces"), episode_id="ep")
    response = asyncio.run(_post(app, _REQ))
    assert response.status_code == 200
    assert response.json() == _RESP


def test_replay_app_returns_404_on_unrecorded_request(tmp_path: Path) -> None:
    app = make_replay_asgi_app(_recorded_store(tmp_path / "traces"), episode_id="ep")
    other = {"model": "gpt-4o", "messages": [{"role": "user", "content": "DIFFERENT"}]}
    response = asyncio.run(_post(app, other))
    assert response.status_code == 404


def test_cli_record_wires_the_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_serve(app: object, host: str, port: int) -> None:
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr(cli, "_serve", fake_serve)
    code = cli.main(
        [
            "record",
            "--upstream",
            "https://api.openai.com",
            "--store",
            str(tmp_path),
            "--episode",
            "ep",
        ]
    )
    assert code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8900
    assert callable(captured["app"])


def test_cli_replay_wires_the_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_serve(app: object, host: str, port: int) -> None:
        captured.update(app=app, port=port)

    monkeypatch.setattr(cli, "_serve", fake_serve)
    code = cli.main(["replay", "--store", str(tmp_path), "--episode", "ep", "--port", "9001"])
    assert code == 0
    assert captured["port"] == 9001
    assert callable(captured["app"])
