"""G1 end-to-end: record a synthetic humanoid episode, faithful-replay, compare
actions (engineering spec §9.3, §15). No real G1 / Zenoh — a fake session, same
pattern as test_om1_e2e.py, over the REAL G1 surface: tool-call decisions
(speak/emotion/robot_action) and the CDR sport-mode gesture request on the bare
`api/sport/request` key (arm/zenoh.go).
"""

import itertools
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from plumbline.adapters.base import Action, Adapter, BusSample
from plumbline.adapters.g1 import G1Adapter
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import BlobKind, JSONValue, Payload, SeamEvent, canonicalize

from tests.toyloop import model_io_bytes

_PROXY = "http://localhost:8900"
_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_ACTION_KEY = "api/sport/request"  # source-verified (arm/zenoh.go sportRequestTopic)
_FRAME = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="


class _FakeZenohSession:
    def __init__(self) -> None:
        self._subs: list[tuple[str, Callable[[_FakeSample], None]]] = []

    def declare_subscriber(self, key_expr: str, handler: Callable[["_FakeSample"], None]) -> object:
        self._subs.append((key_expr, handler))
        return object()

    def close(self) -> None:
        pass

    def publish(self, key_expr: str, payload: bytes) -> None:
        for subscribed, handler in self._subs:
            prefix = subscribed[:-2] if subscribed.endswith("**") else subscribed
            if key_expr.startswith(prefix):
                handler(_FakeSample(key_expr, payload))


@dataclass(frozen=True)
class _FakeSample:
    key_expr: str
    payload: bytes


def _sport_request(api_id: int, parameter: str) -> bytes:
    # Mirror of arm/zenoh.go serializeUnitreeRequest.
    param = parameter.encode() + b"\x00"
    buf = bytearray(b"\x00\x01\x00\x00")
    buf += struct.pack("<q", 0) + struct.pack("<q", api_id) + struct.pack("<q", 0)
    buf += struct.pack("<I", 0) + b"\x00" + b"\x00\x00\x00"
    buf += struct.pack("<I", len(param)) + param
    buf += b"\x00" * ((4 - (len(buf) - 4) % 4) % 4)
    buf += struct.pack("<I", 0)
    return bytes(buf)


def _tool_call_response(*calls: tuple[str, str]) -> JSONValue:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": f"t{i}",
                            "type": "function",
                            "function": {"name": name, "arguments": f'{{"action": "{value}"}}'},
                        }
                        for i, (name, value) in enumerate(calls)
                    ]
                }
            }
        ]
    }


# (caption, tool calls, gesture published on the bus — None for no physical output)
_TICKS: tuple[tuple[str, tuple[tuple[str, str], ...], str | None], ...] = (
    ("the owner extends a hand", (("robot_action", "shake_hand"),), "shake_hand"),
    (
        "someone waves from the door",
        (("robot_action", "face_wave"), ("emotion", "happy")),
        "face_wave",
    ),
    ("a question is asked", (("speak", "hello there"),), None),  # TTS only, no bus traffic
)


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


def _action_sequence(events: Sequence[SeamEvent], adapter: Adapter) -> tuple[Action, ...]:
    schema = adapter.action_schema()
    actions: list[Action] = []
    for event in events:
        if event.seam is Seam.DECIDE_TO_ACT and not event.request.blobs:
            actions.extend(schema.parse(event.request))
    return tuple(actions)


def test_adapter_contract_and_configuration() -> None:
    adapter: Adapter = G1Adapter(proxy_base_url=_PROXY)
    config = adapter.configure_proxy()
    # OM1-family redirect is a config-field override, not env vars (docs/om1-integration.md).
    assert config.config_fields["cortex_llm.config.base_url"] == f"{_PROXY}/v1"
    assert config.env == {}
    assert adapter.clock_hook() is None
    assert adapter.bus_tap() is None
    assert adapter.seam_of(Payload(inline=_vision_request()), _ENDPOINT) is Seam.SENSOR_TO_CAPTION
    assert adapter.seam_of(Payload(inline={"m": []}), _ENDPOINT) is Seam.FUSE_TO_DECIDE
    assert adapter.seam_of(Payload(inline={}), _ACTION_KEY) is Seam.DECIDE_TO_ACT
    assert adapter.seam_of(Payload(inline={}), "om/avatar/request") is Seam.DECIDE_TO_ACT


def test_record_and_faithful_replay_reproduces_action_sequence() -> None:
    session = _FakeZenohSession()
    adapter = G1Adapter(proxy_base_url=_PROXY, zenoh_session=session)
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    episode_id = "g1-sim-001"
    recorder.open_episode(episode_id, {"robot": "g1"})
    recorded: list[SeamEvent] = []
    seq = itertools.count()
    tick_ref = {"t": 0}

    def record(event: SeamEvent) -> None:
        recorder.record(event)
        recorded.append(event)

    def model_event(seam: Seam, request: JSONValue, response: JSONValue) -> SeamEvent:
        req = Payload(inline=request)
        return SeamEvent(
            episode_id,
            next(seq),
            seam,
            tick_ref["t"],
            float(tick_ref["t"]),
            req,
            Payload(inline=response),
            None,
            {},
            canonicalize(req).digest,
            0.0,
        )

    tap = adapter.bus_tap()
    assert tap is not None

    def on_sample(sample: BusSample) -> None:
        blobs = (store.put_blob(sample.raw, BlobKind.BIN),) if sample.raw else ()
        req = Payload(inline=sample.payload, blobs=blobs)
        record(
            SeamEvent(
                episode_id,
                next(seq),
                adapter.seam_of(req, sample.key_expr),
                tick_ref["t"],
                sample.wall_ts,
                req,
                req,
                None,
                {"plumbline.bus_key": sample.key_expr},
                canonicalize(req).digest,
                0.0,
            )
        )

    tap.subscribe(on_sample)

    for index, (caption, calls, gesture) in enumerate(_TICKS):
        tick_ref["t"] = index
        vision = _vision_request()
        record(
            model_event(
                adapter.seam_of(Payload(inline=vision), _ENDPOINT),
                vision,
                {"choices": [{"message": {"content": caption}}]},
            )
        )
        fused: JSONValue = {"model": "cortex", "messages": [{"role": "user", "content": caption}]}
        record(
            adapter.reconstruct_caption_to_fuse(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=index,
                captions=[caption],
                fused_prompt=fused,
                wall_ts=float(index),
            )
        )
        decision = _tool_call_response(*calls)
        record(model_event(adapter.seam_of(Payload(inline=fused), _ENDPOINT), fused, decision))
        record(
            adapter.reconstruct_decide_to_act(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=index,
                decision_response=Payload(inline=decision),
                wall_ts=float(index),
            )
        )
        if gesture is not None:  # the physical gesture request on the real key
            session.publish(_ACTION_KEY, _sport_request(9001, f'{{"action": "{gesture}"}}'))

    recorder.close_episode(episode_id)

    assert tuple(e.seam for e in recorded[:5]) == (
        Seam.SENSOR_TO_CAPTION,
        Seam.CAPTION_TO_FUSE,
        Seam.FUSE_TO_DECIDE,
        Seam.DECIDE_TO_ACT,
        Seam.DECIDE_TO_ACT,  # the bus-tapped sport request
    )
    # The tap's typed decode: a comparable unitree_api/Request view + raw bytes as blob.
    tapped = recorded[4]
    inline = tapped.request.inline
    assert isinstance(inline, dict) and "unitree_api/Request" in inline
    assert tapped.request.blobs and store.get_blob(tapped.request.blobs[0]).startswith(
        b"\x00\x01\x00\x00"
    )

    result = Replayer(store, VirtualClock(), {}).faithful(episode_id)
    assert result.diverged is False
    recorded_actions = _action_sequence(recorded, adapter)
    assert _action_sequence(result.events, adapter) == recorded_actions
    assert recorded_actions[0] == Action("skill", "shake_hand", {})
    assert Action("express", "happy", {}) in recorded_actions
    assert Action("speak", "speak", {"text": "hello there"}) in recorded_actions
    assert model_io_bytes(result.events) == model_io_bytes(recorded)
