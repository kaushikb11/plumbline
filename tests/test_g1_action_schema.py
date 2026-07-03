"""G1 humanoid action schema (engineering spec §9.3) — the REAL vocabulary.

Grounded in OM1's source (plugins/actions/unitree/g1/arm/zenoh.go, emotion, speak):
tool calls `speak`/`emotion`/`robot_action`, each with an {"action": "<value>"}
argument; the physical output is a discrete gesture on `api/sport/request`.
"""

import struct

from plumbline.adapters.base import Action
from plumbline.adapters.g1 import G1_GESTURES, G1ActionSchema, decode_sport_request
from plumbline.core.trace import JSONValue, Payload


def _tool_response(*calls: tuple[str, str]) -> Payload:
    return Payload(
        inline={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": f"t{i}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": f'{{"action": "{value}"}}',
                                },
                            }
                            for i, (name, value) in enumerate(calls)
                        ]
                    }
                }
            ]
        }
    )


def test_command_vocabulary_is_the_real_tool_call_set() -> None:
    assert G1ActionSchema().commands == ("speak", "emotion", "robot_action")


def test_parse_real_tool_calls() -> None:
    plan = G1ActionSchema().parse(
        _tool_response(("robot_action", "shake_hand"), ("emotion", "happy"), ("speak", "hello!"))
    )
    assert plan == (
        Action("skill", "shake_hand", {}),
        Action("express", "happy", {}),
        Action("speak", "speak", {"text": "hello!"}),
    )


def test_parse_tolerates_unknown_and_malformed() -> None:
    assert G1ActionSchema().parse(_tool_response(("unknown_tool", "x"))) == ()
    assert G1ActionSchema().parse(Payload(inline={"other": 1})) == ()


def test_gesture_vocabulary_matches_om1_enum() -> None:
    # arm/zenoh.go ArmAction.EnumValues (idle is a connector no-op).
    assert "shake_hand" in G1_GESTURES and "talking_20s" in G1_GESTURES and "idle" in G1_GESTURES
    assert "walk" not in G1_GESTURES  # the G1 config has no locomotion action


def _sport_request(api_id: int, parameter: str) -> bytes:
    # Mirror of arm/zenoh.go serializeUnitreeRequest.
    param = parameter.encode() + b"\x00"
    buf = bytearray(b"\x00\x01\x00\x00")
    buf += struct.pack("<q", 0)  # identity.id
    buf += struct.pack("<q", api_id)  # identity.api_id
    buf += struct.pack("<q", 0)  # lease.id
    buf += struct.pack("<I", 0)  # policy.priority
    buf += b"\x00"  # policy.noreply
    buf += b"\x00\x00\x00"  # pad to data offset 32
    buf += struct.pack("<I", len(param)) + param
    data_len = len(buf) - 4
    buf += b"\x00" * ((4 - data_len % 4) % 4)
    buf += struct.pack("<I", 0)  # binary sequence length
    return bytes(buf)


def test_decode_sport_request_wire_layout() -> None:
    raw = _sport_request(9001, '{"action": "salute"}')
    decoded: JSONValue = decode_sport_request(raw)
    assert decoded == {"unitree_api/Request": {"api_id": 9001, "parameter": {"action": "salute"}}}
    assert decode_sport_request(b"\x00\x01\x00\x00short") is None
    assert decode_sport_request(b"\xff" + raw[1:]) is None  # wrong header
