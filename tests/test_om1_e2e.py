"""OM1 end-to-end (§9.2, §15): record a synthetic-but-faithful OM1 episode using the
REAL shapes — a Gemini/OpenAI tool-call `Move` decision reconstructed into the
cmd_vel action seam — then faithful-replay and compare actions. Interface facts are
established in docs/om1-integration.md (verified against OM1's source).
"""

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from plumbline.adapters.base import Action, Adapter
from plumbline.adapters.om1 import OM1ActionSchema, OM1Adapter
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.transport.zenoh_tap import ZenohSample

from tests.toyloop import model_io_bytes

_PROXY = "http://localhost:8900"
_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_CMD_VEL_KEY = "cmd_vel"
_FRAME = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="

# (VLM caption, discrete Move label the Cortex LLM decides) per tick.
_TICKS: tuple[tuple[str, str], ...] = (
    ("the corridor ahead is clear", "move forwards"),
    ("a wall is close on the right", "turn left"),
    ("a person is directly ahead", "stand still"),
)


class _FakeZenohSession:
    def __init__(self) -> None:
        self._subs: list[tuple[str, Callable[[ZenohSample], None]]] = []

    def declare_subscriber(self, key_expr: str, handler: Callable[[ZenohSample], None]) -> object:
        self._subs.append((key_expr, handler))
        return object()

    def close(self) -> None:
        pass

    def publish(self, key_expr: str, payload: bytes) -> None:
        for subscribed, handler in self._subs:
            prefix = subscribed[:-2] if subscribed.endswith("**") else subscribed
            if key_expr.startswith(prefix.rstrip("/")) or subscribed == key_expr:
                handler(_FakeSample(key_expr, payload))


@dataclass(frozen=True)
class _FakeSample:
    key_expr: str
    payload: bytes


def _vision_request() -> JSONValue:
    return {
        "model": "vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe."},
                    {"type": "image_url", "image_url": {"url": _FRAME}},
                ],
            }
        ],
    }


def _cortex_response(action_label: str) -> JSONValue:
    """A Cortex chat response with an OpenAI-style Move tool call."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "Move",
                                "arguments": json.dumps({"action": action_label}),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _action_sequence(events: Sequence[SeamEvent], adapter: Adapter) -> tuple[Action, ...]:
    schema = adapter.action_schema()
    actions: list[Action] = []
    for event in events:
        if event.seam is Seam.DECIDE_TO_ACT:
            actions.extend(schema.parse(event.request))
    return tuple(actions)


def test_adapter_contract_and_configuration() -> None:
    adapter: Adapter = OM1Adapter(proxy_base_url=_PROXY)
    config = adapter.configure_proxy()
    # Redirect is a config-field override (cortex_llm.config.base_url), not env vars.
    assert config.config_fields["cortex_llm.config.base_url"] == f"{_PROXY}/v1"
    assert config.env == {}
    assert adapter.clock_hook() is None
    assert adapter.bus_tap() is None
    assert adapter.seam_of(Payload(inline=_vision_request()), _ENDPOINT) is Seam.SENSOR_TO_CAPTION
    assert adapter.seam_of(Payload(inline={"messages": []}), _ENDPOINT) is Seam.FUSE_TO_DECIDE
    assert adapter.seam_of(Payload(inline={}), _CMD_VEL_KEY) is Seam.DECIDE_TO_ACT


def test_move_action_schema_parses_tool_call() -> None:
    schema = OM1ActionSchema()
    parsed = schema.parse(Payload(inline=_cortex_response("move forwards")))
    assert parsed == (Action("move", "move forwards", {}),)


def test_record_and_faithful_replay_reproduces_action_sequence() -> None:
    adapter = OM1Adapter(proxy_base_url=_PROXY)
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    episode_id = "om1-sim-001"
    recorder.open_episode(episode_id, {"robot": "go2", "sim": "gazebo"})
    recorded: list[SeamEvent] = []
    seq = itertools.count()

    def record(event: SeamEvent) -> None:
        recorder.record(event)
        recorded.append(event)

    for tick, (caption, label) in enumerate(_TICKS):
        vision_req = Payload(inline=_vision_request())
        record(
            SeamEvent(
                episode_id,
                next(seq),
                adapter.seam_of(vision_req, _ENDPOINT),
                tick,
                float(tick),
                vision_req,
                Payload(inline={"choices": [{"message": {"content": caption}}]}),
                None,
                {},
                canonicalize(vision_req).digest,
                0.0,
            )
        )
        fused: JSONValue = {"model": "cortex", "messages": [{"role": "user", "content": caption}]}
        record(
            adapter.reconstruct_caption_to_fuse(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=tick,
                captions=[caption],
                fused_prompt=fused,
                wall_ts=float(tick),
            )
        )
        cortex_req = Payload(inline=fused)
        cortex_resp = Payload(inline=_cortex_response(label))
        record(
            SeamEvent(
                episode_id,
                next(seq),
                adapter.seam_of(cortex_req, _ENDPOINT),
                tick,
                float(tick),
                cortex_req,
                cortex_resp,
                None,
                {},
                canonicalize(cortex_req).digest,
                0.0,
            )
        )
        record(
            adapter.reconstruct_decide_to_act(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=tick,
                decision_response=cortex_resp,
                wall_ts=float(tick),
            )
        )
    recorder.close_episode(episode_id)

    assert tuple(e.seam for e in recorded[:4]) == (
        Seam.SENSOR_TO_CAPTION,
        Seam.CAPTION_TO_FUSE,
        Seam.FUSE_TO_DECIDE,
        Seam.DECIDE_TO_ACT,
    )
    result = Replayer(store, VirtualClock(), {}).faithful(episode_id)
    assert result.diverged is False
    recorded_actions = _action_sequence(recorded, adapter)
    assert _action_sequence(result.events, adapter) == recorded_actions
    assert recorded_actions[0] == Action("move", "move forwards", {})
    assert Action("move", "stand still", {}) in recorded_actions
    assert model_io_bytes(result.events) == model_io_bytes(recorded)


def test_bus_tap_classifies_cmd_vel_as_action() -> None:
    session = _FakeZenohSession()
    adapter = OM1Adapter(proxy_base_url=_PROXY, zenoh_session=session)
    tap = adapter.bus_tap()
    assert tap is not None
    seams: list[Seam] = []
    tap.subscribe(lambda sample: seams.append(adapter.seam_of(Payload(inline={}), sample.key_expr)))
    session.publish(_CMD_VEL_KEY, b"\x00\x01\x00\x00")  # a CDR Twist header, byte payload
    assert seams == [Seam.DECIDE_TO_ACT]


def test_decode_cmd_vel_twist_matches_om1_wire_layout() -> None:
    # Mirror of OM1's serializeTwist (cmd_vel.go): CDR-LE header + 6 float64.
    import struct

    from plumbline.adapters.om1 import decode_cmd_vel_twist

    raw = b"\x00\x01\x00\x00" + struct.pack("<6d", 0.5, 0.0, 0.0, 0.0, 0.0, -0.3)
    decoded = decode_cmd_vel_twist(raw)
    assert decoded == {
        "geometry_msgs/Twist": {
            "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": -0.3},
        }
    }
    assert decode_cmd_vel_twist(b"\x00\x01\x00\x00short") is None  # not a Twist
    assert decode_cmd_vel_twist(b"\xff" + raw[1:]) is None  # wrong header
