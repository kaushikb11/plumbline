"""Counterfactual captioner swap on a recorded OM1 episode (eng spec §6, §9.2).

Records a Go2 Gazebo episode (the four seams), then runs an *isolated*
counterfactual replay (§6.2) with `live_frontier = {SENSOR_TO_CAPTION}`: only the
captioner re-executes; every downstream seam must still match the trace or the
DivergencePolicy fires (§6.3).

Two cases verify the divergence machinery on OM1-shaped data:
  - compatible captioner (a paraphrase): the recorded fused prompt still applies,
    the run reproduces without diverging.
  - incompatible captioner (drops the scene's task-relevant content — the
    LiDAR-dog hazard): the recorded fused prompt no longer applies, so the run
    HALTS at the first downstream seam and reports the seam + distance (§6.4),
    serving no stale decision past it.

The vision requests are scene-tagged so a swapped captioner produces a per-scene
caption from the same input the recorded captioner saw.
"""

import itertools
import json
from collections.abc import Callable, Mapping

from plumbline.adapters.om1 import OM1Adapter
from plumbline.core.clock import VirtualClock
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy, Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize

_PROXY = "http://localhost:8900"
_VISION_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_CORTEX_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_ACTION_KEY = "om1/agent/actions/go2"
_FRAME = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
_THRESHOLD = 0.2

_SCENES = ("human_ahead", "obstacle_left", "owner_waving")

# Captioner A — the recorded ground-truth captions.
_CAPTION_A: Mapping[str, str] = {
    "human_ahead": (
        "a person stands roughly one meter directly ahead and appears calm and clearly curious"
    ),
    "obstacle_left": "a solid obstacle sits about forty centimeters to the left side of the robot",
    "owner_waving": (
        "the owner is waving a raised hand and smiling warmly toward the quadruped robot"
    ),
}
# Compatible captioner B — a paraphrase (one word changed per scene; shares tokens).
_CAPTION_B_COMPAT: Mapping[str, str] = {
    "human_ahead": (
        "a human stands roughly one meter directly ahead and appears calm and clearly curious"
    ),
    "obstacle_left": "a solid obstacle sits about forty centimeters to the left flank of the robot",
    "owner_waving": (
        "the owner is waving a raised hand and smiling warmly toward the quadruped dog"
    ),
}
# Incompatible captioner B — disjoint tokens; drops the scene's task-relevant fact.
_CAPTION_B_INCOMPAT: Mapping[str, str] = {
    "human_ahead": (
        "empty hallway extends forward with clear flooring and no hazards detected nearby"
    ),
    "obstacle_left": (
        "open space surrounds every direction without anything blocking forward motion at all"
    ),
    "owner_waving": (
        "nothing notable is happening in view; quiet environment without people present anywhere"
    ),
}
_ACTION: Mapping[str, JSONValue] = {
    "human_ahead": {"commands": [{"type": "move", "x": 0.3, "y": 0.0, "yaw": 0.1}]},
    "obstacle_left": {"commands": [{"type": "move", "x": 0.0, "y": 0.2, "yaw": -0.3}]},
    "owner_waving": {"commands": [{"type": "skill", "name": "shake paw"}]},
}


def _vision_request(scene: str) -> JSONValue:
    return {
        "model": "openai/vlm",
        "scene": scene,  # the swapped captioner reads this to caption the same input
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the scene for the robot."},
                    {"type": "image_url", "image_url": {"url": _FRAME}},
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


def _cortex_response(plan: JSONValue) -> JSONValue:
    return {"id": "cortex-1", "choices": [{"message": {"content": json.dumps(plan)}}]}


def _event(
    episode_id: str, seq: int, seam: Seam, tick: int, request: Payload, response: Payload
) -> SeamEvent:
    return SeamEvent(
        episode_id=episode_id,
        seq=seq,
        seam=seam,
        logical_tick=tick,
        wall_ts=float(tick),
        request=request,
        response=response,
        model_id=None,
        params={},
        request_digest=canonicalize(request).digest,
        latency_ms=0.0,
    )


def _record_gazebo_episode(adapter: OM1Adapter, store: TraceStore) -> str:
    recorder = Recorder(store, VirtualClock())
    episode_id = "go2-gazebo-counterfactual"
    recorder.open_episode(episode_id, {"robot": "go2", "sim": "gazebo"})
    seq = itertools.count()
    for tick, scene in enumerate(_SCENES):
        caption = _CAPTION_A[scene]

        vision_req = Payload(inline=_vision_request(scene))
        recorder.record(
            _event(
                episode_id,
                next(seq),
                adapter.seam_of(vision_req, _VISION_ENDPOINT),
                tick,
                vision_req,
                Payload(inline=_vision_response(caption)),
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
        recorder.record(
            _event(
                episode_id,
                next(seq),
                adapter.seam_of(cortex_req, _CORTEX_ENDPOINT),
                tick,
                cortex_req,
                Payload(inline=_cortex_response(_ACTION[scene])),
            )
        )

        action_req = Payload(inline=_ACTION[scene])
        recorder.record(
            _event(
                episode_id,
                next(seq),
                adapter.seam_of(action_req, _ACTION_KEY),
                tick,
                action_req,
                Payload(inline={"executed": True}),
            )
        )
    recorder.close_episode(episode_id)
    return episode_id


def _captioner_override(captions: Mapping[str, str]) -> Callable[[Payload], Payload]:
    """A swapped captioner: caption the same scene the recorded captioner saw."""

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


def test_counterfactual_compatible_captioner_reproduces_without_divergence() -> None:
    adapter = OM1Adapter(proxy_base_url=_PROXY)
    store = TraceStore()
    episode_id = _record_gazebo_episode(adapter, store)

    replayer = Replayer(store, VirtualClock(), _matchers())
    result = replayer.counterfactual(
        episode_id,
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner_override(_CAPTION_B_COMPAT)},
        on_divergence=DivergencePolicy.HALT,
    )

    assert result.diverged is False
    assert result.divergence_seam is None
    assert result.divergence_distance is None
    # The full episode is reproduced: every tick reaches the action seam.
    assert sum(e.seam is Seam.DECIDE_TO_ACT for e in result.events) == len(_SCENES)


def test_counterfactual_incompatible_captioner_halts_and_reports_seam() -> None:
    adapter = OM1Adapter(proxy_base_url=_PROXY)
    store = TraceStore()
    episode_id = _record_gazebo_episode(adapter, store)

    replayer = Replayer(store, VirtualClock(), _matchers())
    result = replayer.counterfactual(
        episode_id,
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner_override(_CAPTION_B_INCOMPAT)},
        on_divergence=DivergencePolicy.HALT,
    )

    # Divergence is a reported result (§6.4), at the first downstream seam where
    # the recorded fused prompt no longer applies.
    assert result.diverged is True
    assert result.divergence_seam in (Seam.CAPTION_TO_FUSE, Seam.FUSE_TO_DECIDE)
    assert result.divergence_distance is not None
    assert result.divergence_distance > _THRESHOLD
    # Halt-on-divergence: no fabricated decision or action served past it (§6.4).
    assert all(e.seam is not Seam.FUSE_TO_DECIDE for e in result.events)
    assert all(e.seam is not Seam.DECIDE_TO_ACT for e in result.events)
