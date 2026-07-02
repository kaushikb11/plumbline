"""Generic-adapter record + store round-trip (engineering spec §9.1, §15) — no bus.

Records a synthetic perception->caption->fuse->decide loop through the generic
adapter (the action seam DERIVED from the decision response, not observed on a
bus), then loads it back via `Replayer.faithful` (a trace load, not a re-execution
— the re-drive proof is `test_reexecution.py`) and asserts the derived action
sequence and byte-identical stored model I/O survive the round-trip. Evidence the
substrate's trace/replay path does not require OM1's Zenoh mechanism.
"""

import itertools
from collections.abc import Sequence

from plumbline.adapters.base import Action, Adapter
from plumbline.adapters.generic import GenericAgentAdapter
from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize

from tests.toyloop import model_io_bytes

_PROXY = "http://localhost:8900"
_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_FRAME = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
_TICKS = (
    ("a person is one meter ahead", "stop"),
    ("the path ahead is clear", "move_forward"),
    ("an obstacle is on the left", "turn_right"),
)


def _vision_request() -> JSONValue:
    return {
        "model": "vlm",
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


def _vision_response(caption: str) -> JSONValue:
    return {"id": "vlm-1", "choices": [{"message": {"content": caption}}]}


def _decide_request(caption: str) -> JSONValue:
    return {
        "model": "llm",
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "Decide one action."},
            {"role": "user", "content": f"Observation: {caption}."},
        ],
    }


def _decide_response(action: str) -> JSONValue:
    return {"id": "llm-1", "choices": [{"message": {"content": action}}]}


def _action_sequence(events: Sequence[SeamEvent], adapter: Adapter) -> tuple[Action, ...]:
    schema = adapter.action_schema()
    actions: list[Action] = []
    for event in events:
        if event.seam is Seam.DECIDE_TO_ACT:
            actions.extend(schema.parse(event.request))
    return tuple(actions)


def test_record_and_faithful_replay_reproduces_action_sequence() -> None:
    adapter = GenericAgentAdapter(proxy_base_url=_PROXY)
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    episode_id = "generic-agent-001"
    recorder.open_episode(episode_id, {"runtime": "generic"})
    seq = itertools.count()
    recorded: list[SeamEvent] = []

    def record_model(seam: Seam, request: JSONValue, response: JSONValue, tick: int) -> None:
        req = Payload(inline=request)
        event = SeamEvent(
            episode_id=episode_id,
            seq=next(seq),
            seam=seam,
            logical_tick=tick,
            wall_ts=float(tick),
            request=req,
            response=Payload(inline=response),
            model_id=None,
            params={},
            request_digest=canonicalize(req).digest,
            latency_ms=0.0,
        )
        recorder.record(event)
        recorded.append(event)

    def record_derived(event: SeamEvent) -> None:
        recorder.record(event)
        recorded.append(event)

    for tick, (caption, action) in enumerate(_TICKS):
        vision = _vision_request()
        record_model(
            adapter.seam_of(Payload(inline=vision), _ENDPOINT),
            vision,
            _vision_response(caption),
            tick,
        )

        fused = _decide_request(caption)
        record_derived(
            adapter.reconstruct_caption_to_fuse(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=tick,
                captions=[caption],
                fused_prompt=fused,
                wall_ts=float(tick),
            )
        )

        decide_response = _decide_response(action)
        record_model(
            adapter.seam_of(Payload(inline=fused), _ENDPOINT), fused, decide_response, tick
        )

        # DECIDE_TO_ACT derived from the decision response — no bus.
        record_derived(
            adapter.reconstruct_decide_to_act(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=tick,
                decision_response=Payload(inline=decide_response),
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
    replayed_actions = _action_sequence(result.events, adapter)
    assert replayed_actions == recorded_actions
    assert recorded_actions == (
        Action("act", "stop", {}),
        Action("act", "move_forward", {}),
        Action("act", "turn_right", {}),
    )
    assert model_io_bytes(result.events) == model_io_bytes(recorded)
