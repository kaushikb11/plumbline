"""G1 humanoid action schema (engineering spec §9.3)."""

from collections.abc import Sequence

from plumbline.adapters.base import Action
from plumbline.adapters.g1 import G1ActionSchema
from plumbline.core.trace import JSONValue, Payload


def _parse(commands: Sequence[JSONValue]) -> tuple[Action, ...]:
    return G1ActionSchema().parse(Payload(inline={"commands": list(commands)}))


def test_command_vocabulary() -> None:
    assert G1ActionSchema().commands == ("walk", "turn", "gesture", "speak", "pose")


def test_parse_humanoid_commands() -> None:
    assert _parse([{"type": "walk", "vx": 0.3, "vy": 0.0, "vyaw": 0.1}]) == (
        Action("walk", "walk", {"vx": 0.3, "vy": 0.0, "vyaw": 0.1}),
    )
    assert _parse([{"type": "turn", "vyaw": 0.5}]) == (Action("turn", "turn", {"vyaw": 0.5}),)
    assert _parse([{"type": "gesture", "name": "wave"}]) == (Action("gesture", "wave", {}),)
    assert _parse([{"type": "speak", "text": "hello"}]) == (
        Action("speak", "speak", {"text": "hello"}),
    )
    assert _parse([{"type": "pose", "name": "bow"}]) == (Action("pose", "bow", {}),)


def test_parse_preserves_order_and_tolerates_malformed() -> None:
    plan = _parse([{"type": "walk", "vx": 0.1}, {"type": "speak", "text": "go"}])
    assert [a.kind for a in plan] == ["walk", "speak"]
    # unknown type / non-dict command / non-list commands / non-dict inline -> tolerant
    assert _parse([{"type": "unknown"}, "not-a-dict"]) == ()
    assert G1ActionSchema().parse(Payload(inline={"commands": "nope"})) == ()
    assert G1ActionSchema().parse(Payload(inline={"other": 1})) == ()
