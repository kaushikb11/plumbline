"""The integrated record -> counterfactual -> gate journey on the recorder's OWN
output (§4, §6, §8; limitations gap #3).

Drives a fake OM1-shaped runtime through the integrated recorder — the HTTP proxy
with a BoundaryTickPolicy and a RecordingCoordinator, and NO x-plumbline-tick header —
then loads the PRODUCED episode and runs Replayer.counterfactual + gate() on it. This
is the flow that had only ever been tested on hand-built SeamEvent fixtures.
"""

import asyncio
import json
from collections.abc import Callable, Mapping

from plumbline.adapters.om1 import OM1ActionSchema, OM1Adapter
from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.replayer import DivergencePolicy, Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent
from plumbline.proxy.http import AsyncHTTPProxy, HTTPRequest, HTTPResponse
from plumbline.proxy.normalizers import contains_image
from plumbline.proxy.tick import BoundaryTickPolicy
from plumbline.recording import RecordingCoordinator
from plumbline.regression import Config, GoldenSet, gate

_URL = "https://api.openai.com/v1/chat/completions"
_FRAME = "data:image/png;base64,/9j/4AAQSkZJRg=="
_THRESHOLD = 0.2
_SCENES = ("human_ahead", "obstacle_left", "owner_waving")

_CAPTION_A: Mapping[str, str] = {
    "human_ahead": "a person stands one meter directly ahead and appears calm and curious",
    "obstacle_left": "a solid obstacle sits forty centimeters to the left side of the robot",
    "owner_waving": "the owner is waving a hand and smiling warmly toward the quadruped robot",
}
_CAPTION_B_COMPAT: Mapping[str, str] = {
    "human_ahead": "a human stands one meter directly ahead and appears calm and curious",
    "obstacle_left": "a solid obstacle sits forty centimeters to the left flank of the robot",
    "owner_waving": "the owner is waving a hand and smiling warmly toward the quadruped dog",
}
_CAPTION_B_INCOMPAT: Mapping[str, str] = {
    "human_ahead": "empty hallway extends forward with clear flooring and no hazards nearby",
    "obstacle_left": "open space surrounds every direction with nothing blocking forward motion",
    "owner_waving": "nothing notable in view; a quiet environment without any people present",
}
_MOVE: Mapping[str, str] = {
    "human_ahead": "move forwards",
    "obstacle_left": "turn left",
    "owner_waving": "stand still",
}


def _cortex_response(move: str) -> JSONValue:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "Move", "arguments": json.dumps({"action": move})}}
                    ]
                }
            }
        ]
    }


class _FakeTransport:
    """The upstream: an image request returns the scene's caption; a chat request
    returns the scene's Cortex Move tool call."""

    def __init__(self, captions: Mapping[str, str], moves: Mapping[str, str]) -> None:
        self._captions = captions
        self._moves = moves

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        body = json.loads(request.body)
        scene = body["scene"]
        if contains_image(body):
            payload: JSONValue = {"choices": [{"message": {"content": self._captions[scene]}}]}
        else:
            payload = _cortex_response(self._moves[scene])
        return HTTPResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
        )


def _vision_body(scene: str) -> bytes:
    return json.dumps(
        {
            "model": "vlm",
            "scene": scene,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the scene."},
                        {"type": "image_url", "image_url": {"url": _FRAME}},
                    ],
                }
            ],
        }
    ).encode()


def _cortex_body(scene: str) -> bytes:
    return json.dumps(
        {"model": "cortex", "scene": scene, "messages": [{"role": "user", "content": "decide"}]}
    ).encode()


def _record_episode(store: TraceStore, episode_id: str = "go2-integrated") -> str:
    coordinator = RecordingCoordinator(
        store, episode_id=episode_id, adapter=OM1Adapter(proxy_base_url="x")
    )
    proxy = AsyncHTTPProxy(
        transport=_FakeTransport(_CAPTION_A, _MOVE),
        recorder=coordinator,
        store=store,
        tick_policy=BoundaryTickPolicy(),
    )

    async def drive() -> None:
        for scene in _SCENES:
            ctx = Context(
                episode_id=episode_id, model_id=None, params={}, logical_tick=0
            )  # NO tick override
            await proxy.record(HTTPRequest("POST", _URL, {}, _vision_body(scene)), ctx)
            await proxy.record(HTTPRequest("POST", _URL, {}, _cortex_body(scene)), ctx)

    asyncio.run(drive())
    coordinator.close()
    return episode_id


def _captioner_override(captions: Mapping[str, str]) -> Callable[[Payload], Payload]:
    def override(request: Payload) -> Payload:
        inline = request.inline
        assert isinstance(inline, dict)
        scene = inline["scene"]
        assert isinstance(scene, str)
        return Payload(inline={"choices": [{"message": {"content": captions[scene]}}]})

    return override


def _matchers() -> dict[Seam, Matcher]:
    return {
        Seam.CAPTION_TO_FUSE: EmbeddingMatcher(threshold=_THRESHOLD),
        Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=_THRESHOLD),
        Seam.DECIDE_TO_ACT: ExactMatcher(),
    }


def _config(captions: Mapping[str, str]) -> Config:
    return Config(
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner_override(captions)},
        matchers=_matchers(),
    )


def _actions(events: tuple[SeamEvent, ...]) -> tuple[object, ...]:
    schema = OM1ActionSchema()
    out: list[object] = []
    for event in events:
        if event.seam is Seam.DECIDE_TO_ACT:
            out.extend(schema.parse(event.request))
    return tuple(out)


def test_integrated_recorder_produces_full_four_seam_episode() -> None:
    store = TraceStore()
    episode = store.load_episode(_record_episode(store))
    assert len(episode.events) == 12  # 4 seams x 3 ticks
    # Auto-ticked WITHOUT a header: three distinct ticks (gap #2 regression guard).
    assert sorted({e.logical_tick for e in episode.events}) == [0, 1, 2]
    for tick in (0, 1, 2):
        seams = [e.seam for e in episode.events if e.logical_tick == tick]
        assert seams == [
            Seam.SENSOR_TO_CAPTION,
            Seam.CAPTION_TO_FUSE,
            Seam.FUSE_TO_DECIDE,
            Seam.DECIDE_TO_ACT,
        ]
    assert [e.seq for e in episode.events] == list(range(12))  # gap-free pipeline order


def test_produced_episode_faithful_replay_matches() -> None:
    store = TraceStore()
    episode_id = _record_episode(store)
    result = Replayer(store, VirtualClock(), {}).faithful(episode_id)
    assert result.diverged is False
    assert _actions(result.events) == _actions(store.load_episode(episode_id).events)


def test_counterfactual_on_produced_episode_compatible_reproduces() -> None:
    store = TraceStore()
    episode_id = _record_episode(store)
    result = Replayer(store, VirtualClock(), _matchers()).counterfactual(
        episode_id,
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner_override(_CAPTION_B_COMPAT)},
        on_divergence=DivergencePolicy.HALT,
    )
    assert result.diverged is False
    assert sum(e.seam is Seam.DECIDE_TO_ACT for e in result.events) == len(_SCENES)


def test_counterfactual_on_produced_episode_incompatible_halts() -> None:
    store = TraceStore()
    episode_id = _record_episode(store)
    result = Replayer(store, VirtualClock(), _matchers()).counterfactual(
        episode_id,
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner_override(_CAPTION_B_INCOMPAT)},
        on_divergence=DivergencePolicy.HALT,
    )
    assert result.diverged is True
    assert result.divergence_seam in (Seam.CAPTION_TO_FUSE, Seam.FUSE_TO_DECIDE)


def test_gate_on_produced_episode_passes_compatible_fails_incompatible() -> None:
    store = TraceStore()
    episode_id = _record_episode(store)
    golden = GoldenSet(store)
    golden.add(episode_id)  # captures the PRODUCED action sequence
    assert gate(store, golden, _config(_CAPTION_B_COMPAT), drift_threshold=0.1).passed is True
    incompatible = gate(store, golden, _config(_CAPTION_B_INCOMPAT), drift_threshold=0.1)
    assert incompatible.passed is False
    assert incompatible.per_episode[0].diverged is True
