"""OM1 end-to-end: record a Go2 Gazebo episode, faithful-replay, compare actions.

Engineering spec §15 (end-to-end OM1 test) and §9.2 (OM1 adapter). A real run
records the four seams from a live OM1 stack — the VLM/ASR and Cortex calls via
the HTTP proxy, the action plans via the Zenoh tap. No OM1 / Zenoh / Gazebo is
available here, so this drives the adapter's own surface (`seam_of`, `bus_tap`,
`action_schema`, `reconstruct_caption_to_fuse`) over a synthetic-but-faithful
Go2 episode: image -> caption, caption -> fused prompt, fused prompt -> Cortex
action plan, action plan -> bus. It records, faithful-replays, and asserts the
reproduced action sequence matches the recorded one.

The Zenoh session is faked (the substrate carries no `zenoh` dependency); the
fake satisfies the injected `ZenohSession` Protocol exactly as a real session
would.
"""

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from plumbline.adapters.base import Action, Adapter, BusSample
from plumbline.adapters.om1 import OM1Adapter
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.transport.zenoh_tap import ZenohSample

from tests.toyloop import model_io_bytes

_PROXY = "http://localhost:8900"
_VISION_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_CORTEX_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_ACTION_KEY = "om1/agent/actions/go2"
_CAMERA_FRAME = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="


# --- a fake Zenoh session satisfying the injected ZenohSession Protocol -------


@dataclass(frozen=True)
class _FakeSample:
    key_expr: str
    payload: bytes


class _FakeZenohSession:
    def __init__(self) -> None:
        self._subs: list[tuple[str, Callable[[ZenohSample], None]]] = []
        self.closed = False

    def declare_subscriber(self, key_expr: str, handler: Callable[[ZenohSample], None]) -> object:
        self._subs.append((key_expr, handler))
        return object()

    def close(self) -> None:
        self.closed = True

    def publish(self, key_expr: str, payload: bytes) -> None:
        for subscribed, handler in self._subs:
            if _key_matches(subscribed, key_expr):
                handler(_FakeSample(key_expr=key_expr, payload=payload))


def _key_matches(pattern: str, key: str) -> bool:
    if pattern.endswith("**"):
        return key.startswith(pattern[:-2])
    return pattern == key


# --- a synthetic but OM1-faithful Go2 Gazebo episode ------------------------


@dataclass(frozen=True)
class _Tick:
    caption: str
    action_plan: JSONValue


def _episode() -> tuple[_Tick, ...]:
    return (
        _Tick(
            caption="a human is 1.2 m ahead and looks curious",
            action_plan={"commands": [{"type": "move", "x": 0.3, "y": 0.0, "yaw": 0.1}]},
        ),
        _Tick(
            caption="an obstacle is 0.4 m to the left",
            action_plan={"commands": [{"type": "move", "x": 0.0, "y": 0.2, "yaw": -0.3}]},
        ),
        _Tick(
            caption="the owner is waving",
            action_plan={
                "commands": [
                    {"type": "skill", "name": "shake paw"},
                    {"type": "speak", "text": "hello"},
                ]
            },
        ),
    )


def _vision_request() -> JSONValue:
    return {
        "model": "openai/vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the scene for the robot."},
                    {"type": "image_url", "image_url": {"url": _CAMERA_FRAME}},
                ],
            }
        ],
    }


def _vision_response(caption: str) -> JSONValue:
    return {"id": "vlm-1", "model": "vlm", "choices": [{"message": {"content": caption}}]}


def _cortex_request(caption: str) -> JSONValue:
    return {
        "model": "openai/cortex",
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "You are a Go2 quadruped. Avoid obstacles."},
            {"role": "user", "content": f"Observation: {caption}. Decide the next action."},
        ],
    }


def _cortex_response(action_plan: JSONValue) -> JSONValue:
    return {"id": "cortex-1", "choices": [{"message": {"content": json.dumps(action_plan)}}]}


def _action_sequence(events: Sequence[SeamEvent], adapter: Adapter) -> tuple[Action, ...]:
    schema = adapter.action_schema()
    actions: list[Action] = []
    for event in events:
        if event.seam is Seam.DECIDE_TO_ACT:
            actions.extend(schema.parse(event.request))
    return tuple(actions)


# --- the test ---------------------------------------------------------------


def test_adapter_contract_and_configuration() -> None:
    adapter: Adapter = OM1Adapter(proxy_base_url=_PROXY)  # conforms to the Protocol

    config = adapter.configure_proxy()
    # Zero source changes: configuration only, pointing providers at the proxy.
    assert config.env["OPENAI_BASE_URL"] == f"{_PROXY}/v1"
    assert config.env["ANTHROPIC_BASE_URL"] == _PROXY  # SDK appends /v1/messages
    assert config.env["OLLAMA_HOST"] == _PROXY
    assert config.proxy_base_url == _PROXY

    # Determinism envelope (§3.4): no scheduler control yet.
    assert adapter.clock_hook() is None
    # No bus tap without a session.
    assert adapter.bus_tap() is None

    # seam_of classification across the four interception points.
    assert (
        adapter.seam_of(Payload(inline=_vision_request()), _VISION_ENDPOINT)
        is Seam.SENSOR_TO_CAPTION
    )
    assert (
        adapter.seam_of(
            Payload(inline={"audio": "..."}), "https://api.openai.com/v1/audio/transcriptions"
        )
        is Seam.SENSOR_TO_CAPTION
    )
    assert (
        adapter.seam_of(Payload(inline=_cortex_request("x")), _CORTEX_ENDPOINT)
        is Seam.FUSE_TO_DECIDE
    )
    assert adapter.seam_of(Payload(inline={"commands": []}), _ACTION_KEY) is Seam.DECIDE_TO_ACT


def test_record_and_faithful_replay_reproduces_action_sequence() -> None:
    session = _FakeZenohSession()
    adapter = OM1Adapter(proxy_base_url=_PROXY, zenoh_session=session)
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    episode_id = "go2-gazebo-001"
    recorder.open_episode(episode_id, {"robot": "go2", "sim": "gazebo"})

    recorded: list[SeamEvent] = []
    seq = itertools.count()
    current_tick = {"t": 0}

    def record(event: SeamEvent) -> None:
        recorder.record(event)
        recorded.append(event)

    def make_event(seam: Seam, request: JSONValue, response: JSONValue) -> SeamEvent:
        req = Payload(inline=request)
        return SeamEvent(
            episode_id=episode_id,
            seq=next(seq),
            seam=seam,
            logical_tick=current_tick["t"],
            wall_ts=float(current_tick["t"]),
            request=req,
            response=Payload(inline=response),
            model_id=None,
            params={},
            request_digest=canonicalize(req).digest,
            latency_ms=0.0,
        )

    # The bus tap turns published action plans into DECIDE_TO_ACT events (§4.3).
    tap = adapter.bus_tap()
    assert tap is not None

    def on_bus_sample(sample: BusSample) -> None:
        action_request = Payload(inline=sample.payload)
        recorded_event = SeamEvent(
            episode_id=episode_id,
            seq=next(seq),
            seam=adapter.seam_of(action_request, sample.key_expr),
            logical_tick=current_tick["t"],
            wall_ts=sample.wall_ts,
            request=action_request,
            response=Payload(inline={"executed": True}),
            model_id=None,
            params={},
            request_digest=canonicalize(action_request).digest,
            latency_ms=0.0,
        )
        record(recorded_event)

    tap.subscribe(on_bus_sample)

    for index, tick in enumerate(_episode()):
        current_tick["t"] = index

        # SENSOR_TO_CAPTION: the VLM vision call (classified by image content).
        vision_req = _vision_request()
        vision_seam = adapter.seam_of(Payload(inline=vision_req), _VISION_ENDPOINT)
        record(make_event(vision_seam, vision_req, _vision_response(tick.caption)))

        # CAPTION_TO_FUSE: reconstructed from the caption + the fused prompt.
        fused_prompt = _cortex_request(tick.caption)
        record(
            adapter.reconstruct_caption_to_fuse(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=index,
                captions=[tick.caption],
                fused_prompt=fused_prompt,
                wall_ts=float(index),
            )
        )

        # FUSE_TO_DECIDE: the Cortex chat completion (no image -> decision call).
        cortex_seam = adapter.seam_of(Payload(inline=fused_prompt), _CORTEX_ENDPOINT)
        record(make_event(cortex_seam, fused_prompt, _cortex_response(tick.action_plan)))

        # DECIDE_TO_ACT: the action plan published on the Zenoh action bus.
        session.publish(_ACTION_KEY, json.dumps(tick.action_plan).encode("utf-8"))

    recorder.close_episode(episode_id)

    # Per-tick seam order is the pipeline order.
    assert tuple(e.seam for e in recorded[:4]) == (
        Seam.SENSOR_TO_CAPTION,
        Seam.CAPTION_TO_FUSE,
        Seam.FUSE_TO_DECIDE,
        Seam.DECIDE_TO_ACT,
    )

    # Faithful-replay the episode.
    replayer = Replayer(store, VirtualClock(), {})
    result = replayer.faithful(episode_id)
    assert result.diverged is False

    # The reproduced action sequence matches the recorded one (the §15 assertion).
    recorded_actions = _action_sequence(recorded, adapter)
    replayed_actions = _action_sequence(result.events, adapter)
    assert len(recorded_actions) == 4  # 3 moves/skills + 1 speak across 3 ticks
    assert replayed_actions == recorded_actions
    assert recorded_actions[0] == Action("move", "move", {"x": 0.3, "y": 0.0, "yaw": 0.1})
    assert recorded_actions[2] == Action("skill", "shake paw", {})

    # And the model-seam I/O is byte-identical (the determinism guarantee).
    assert model_io_bytes(result.events) == model_io_bytes(recorded)
