"""Counterfactual captioner swap on a synthetic G1 episode (eng spec §6, §9.3).

Same divergence machinery as the OM1 counterfactual, over the humanoid embodiment:
a compatible paraphrase reproduces; an incompatible caption halts at the first
downstream seam and reports it, serving no stale action past it (invariant 5).
"""

import itertools
import json
from collections.abc import Callable, Mapping

from plumbline.adapters.g1 import G1Adapter
from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy, Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize

_PROXY = "http://localhost:8900"
_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_FRAME = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
_THRESHOLD = 0.2
_SCENES = ("clear_ahead", "person_left", "owner_waving")

_CAPTION_A: Mapping[str, str] = {
    "clear_ahead": "the corridor directly ahead is entirely open and clear for several meters",
    "person_left": "a person is standing about one meter to the left of the humanoid robot",
    "owner_waving": "the owner is waving a raised hand and smiling warmly toward the robot",
}
_CAPTION_B_COMPAT: Mapping[str, str] = {
    "clear_ahead": "the corridor directly ahead is entirely open and clear for several metres",
    "person_left": "a human is standing about one meter to the left of the humanoid robot",
    "owner_waving": "the owner is waving a raised hand and smiling warmly toward the machine",
}
_CAPTION_B_INCOMPAT: Mapping[str, str] = {
    "clear_ahead": "a large solid object completely blocks the way immediately in front",
    "person_left": "empty quiet room with plain walls and nothing else present nearby",
    "owner_waving": "open space everywhere, no people and no obstacles whatsoever around",
}
# Real G1 decisions: tool calls with an {"action": ...} argument (no locomotion).
_ACTION: Mapping[str, JSONValue] = {
    "clear_ahead": {"name": "emotion", "action": "curious"},
    "person_left": {"name": "robot_action", "action": "face_wave"},
    "owner_waving": {"name": "robot_action", "action": "shake_hand"},
}


def _decision_response(scene: str) -> JSONValue:
    call = _ACTION[scene]
    assert isinstance(call, dict)
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "t0",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps({"action": call["action"]}),
                            },
                        }
                    ]
                }
            }
        ]
    }


def _vision_request(scene: str) -> JSONValue:
    return {
        "model": "vlm",
        "scene": scene,
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


def _cortex_request(caption: str) -> JSONValue:
    return {
        "model": "cortex",
        "messages": [{"role": "user", "content": f"Observation: {caption}."}],
    }


def _event(
    episode_id: str, seq: int, seam: Seam, tick: int, request: Payload, response: Payload
) -> SeamEvent:
    return SeamEvent(
        episode_id,
        seq,
        seam,
        tick,
        float(tick),
        request,
        response,
        None,
        {},
        canonicalize(request).digest,
        0.0,
    )


def _record(adapter: G1Adapter, store: TraceStore) -> str:
    recorder = Recorder(store, VirtualClock())
    episode_id = "g1-counterfactual"
    recorder.open_episode(episode_id, {"robot": "g1"})
    seq = itertools.count()
    for tick, scene in enumerate(_SCENES):
        caption = _CAPTION_A[scene]
        vision_req = Payload(inline=_vision_request(scene))
        recorder.record(
            _event(
                episode_id,
                next(seq),
                adapter.seam_of(vision_req, _ENDPOINT),
                tick,
                vision_req,
                Payload(inline={"choices": [{"message": {"content": caption}}]}),
            )
        )
        fused = _cortex_request(caption)
        recorder.record(
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
        decision = _decision_response(scene)
        recorder.record(
            _event(
                episode_id,
                next(seq),
                adapter.seam_of(cortex_req, _ENDPOINT),
                tick,
                cortex_req,
                Payload(inline=decision),
            )
        )
        recorder.record(
            adapter.reconstruct_decide_to_act(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=tick,
                decision_response=Payload(inline=decision),
                wall_ts=float(tick),
            )
        )
    recorder.close_episode(episode_id)
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


def test_counterfactual_compatible_captioner_reproduces() -> None:
    store = TraceStore()
    episode_id = _record(G1Adapter(proxy_base_url=_PROXY), store)
    result = Replayer(store, VirtualClock(), _matchers()).counterfactual(
        episode_id,
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner_override(_CAPTION_B_COMPAT)},
        on_divergence=DivergencePolicy.HALT,
    )
    assert result.diverged is False
    assert sum(e.seam is Seam.DECIDE_TO_ACT for e in result.events) == len(_SCENES)


def test_counterfactual_incompatible_captioner_halts_and_reports_seam() -> None:
    store = TraceStore()
    episode_id = _record(G1Adapter(proxy_base_url=_PROXY), store)
    result = Replayer(store, VirtualClock(), _matchers()).counterfactual(
        episode_id,
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner_override(_CAPTION_B_INCOMPAT)},
        on_divergence=DivergencePolicy.HALT,
    )
    assert result.diverged is True
    assert result.divergence_seam in (Seam.CAPTION_TO_FUSE, Seam.FUSE_TO_DECIDE)
    assert result.divergence_distance is not None and result.divergence_distance > _THRESHOLD
    assert all(e.seam is not Seam.FUSE_TO_DECIDE for e in result.events)
    assert all(e.seam is not Seam.DECIDE_TO_ACT for e in result.events)
