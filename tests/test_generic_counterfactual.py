"""Counterfactual captioner swap on a generic (bus-less) episode (eng spec §6, §9.1).

The same divergence machinery as the OM1 counterfactual, but on a runtime with no
bus and a DERIVED action seam: a compatible paraphrase reproduces; an incompatible
caption (dropping the task-relevant fact) HALTS at the first downstream seam and
reports it, serving no stale decision past it (invariant 5, §6.4).
"""

import itertools
from collections.abc import Callable, Mapping

from plumbline.adapters.generic import GenericAgentAdapter
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
_SCENES = ("person_ahead", "obstacle_left", "path_clear")

_CAPTION_A: Mapping[str, str] = {
    "person_ahead": "a person stands roughly one meter directly ahead and appears calm and curious",
    "obstacle_left": "a solid obstacle sits about forty centimeters to the left side of the robot",
    "path_clear": "the corridor ahead is entirely open and clear with no objects blocking the way",
}
_CAPTION_B_COMPAT: Mapping[str, str] = {
    "person_ahead": "a human stands roughly one meter directly ahead and appears calm and curious",
    "obstacle_left": "a solid obstacle sits about forty centimeters to the left flank of the robot",
    "path_clear": "the corridor ahead is entirely open and clear with no objects blocking the path",
}
_CAPTION_B_INCOMPAT: Mapping[str, str] = {
    "person_ahead": "an empty quiet room with plain walls and nothing else",
    "obstacle_left": "clear open space everywhere, no barriers whatsoever around",
    "path_clear": "a huge boulder totally jams the passage right here",
}
_ACTION: Mapping[str, str] = {
    "person_ahead": "stop",
    "obstacle_left": "turn_right",
    "path_clear": "move_forward",
}


def _vision_request(scene: str) -> JSONValue:
    return {
        "model": "vlm",
        "scene": scene,  # the swapped captioner reads this to caption the same input
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
        "messages": [{"role": "user", "content": f"Observation: {caption}. Decide."}],
    }


def _decide_response(action: str) -> JSONValue:
    return {"id": "llm-1", "choices": [{"message": {"content": action}}]}


def _record(adapter: GenericAgentAdapter, store: TraceStore) -> str:
    recorder = Recorder(store, VirtualClock())
    episode_id = "generic-counterfactual"
    recorder.open_episode(episode_id, {"runtime": "generic"})
    seq = itertools.count()
    for tick, scene in enumerate(_SCENES):
        caption = _CAPTION_A[scene]
        vision_req = Payload(inline=_vision_request(scene))
        recorder.record(
            SeamEvent(
                episode_id=episode_id,
                seq=next(seq),
                seam=adapter.seam_of(vision_req, _ENDPOINT),
                logical_tick=tick,
                wall_ts=float(tick),
                request=vision_req,
                response=Payload(inline=_vision_response(caption)),
                model_id=None,
                params={},
                request_digest=canonicalize(vision_req).digest,
                latency_ms=0.0,
            )
        )
        fused = _decide_request(caption)
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
        decide_req = Payload(inline=fused)
        decide_response = _decide_response(_ACTION[scene])
        recorder.record(
            SeamEvent(
                episode_id=episode_id,
                seq=next(seq),
                seam=adapter.seam_of(decide_req, _ENDPOINT),
                logical_tick=tick,
                wall_ts=float(tick),
                request=decide_req,
                response=Payload(inline=decide_response),
                model_id=None,
                params={},
                request_digest=canonicalize(decide_req).digest,
                latency_ms=0.0,
            )
        )
        recorder.record(
            adapter.reconstruct_decide_to_act(
                episode_id=episode_id,
                seq=next(seq),
                logical_tick=tick,
                decision_response=Payload(inline=decide_response),
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
        return Payload(inline=_vision_response(captions[scene]))

    return override


def _matchers() -> dict[Seam, Matcher]:
    return {
        Seam.CAPTION_TO_FUSE: EmbeddingMatcher(threshold=_THRESHOLD),
        Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=_THRESHOLD),
        Seam.DECIDE_TO_ACT: ExactMatcher(),
    }


def test_counterfactual_compatible_captioner_reproduces() -> None:
    store = TraceStore()
    episode_id = _record(GenericAgentAdapter(proxy_base_url=_PROXY), store)
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
    episode_id = _record(GenericAgentAdapter(proxy_base_url=_PROXY), store)
    result = Replayer(store, VirtualClock(), _matchers()).counterfactual(
        episode_id,
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner_override(_CAPTION_B_INCOMPAT)},
        on_divergence=DivergencePolicy.HALT,
    )
    assert result.diverged is True
    assert result.divergence_seam in (Seam.CAPTION_TO_FUSE, Seam.FUSE_TO_DECIDE)
    assert result.divergence_distance is not None
    assert result.divergence_distance > _THRESHOLD
    assert all(e.seam is not Seam.FUSE_TO_DECIDE for e in result.events)
    assert all(e.seam is not Seam.DECIDE_TO_ACT for e in result.events)
