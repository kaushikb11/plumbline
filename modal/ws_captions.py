"""Modal app — a WebSocket caption server (mimics OpenMind's `wss://` caption stream).

Lets you exercise Plumbline's WebSocket capture (`AsyncWSProxy` / `make_ws_asgi_app`)
against a REAL remote WS server. Point the WS record proxy's `upstream` at this app's
`wss://…/ws/captions` URL.

    modal deploy modal/ws_captions.py
"""

import modal
from fastapi import FastAPI, WebSocket

app = modal.App("plumbline-ws-captions")
image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi==0.115.0")

web = FastAPI()

_CAPTIONS = [
    "a corridor is clear ahead",
    "an obstacle is on the left",
    "a person is directly ahead",
]


@web.websocket("/ws/captions")
async def captions(websocket: WebSocket) -> None:
    await websocket.accept()
    # A scripted caption stream; a real deployment would caption incoming frames.
    for caption in _CAPTIONS:
        await websocket.send_json({"caption": caption})
    await websocket.close()


@app.function(image=image)
@modal.asgi_app()
def serve() -> FastAPI:
    return web
