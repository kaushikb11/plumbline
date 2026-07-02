"""Counterfactual replay over a proxy-recorded episode (eng spec §6, §4.2).

Regression test for the seam-grouping fix: `Replayer.counterfactual` groups seam
events by `logical_tick`, so a swapped seam can be compared against the same loop
iteration's downstream seam. That requires every seam of one iteration to share a
tick. The loop driver stamps `Context.logical_tick`; the proxy records it (rather
than inventing a per-call tick), so episodes recorded *through the proxy* now
support counterfactual replay.

Only the two model seams are recorded here (no CAPTION_TO_FUSE), so the first
downstream seam present after the captioner swap is FUSE_TO_DECIDE — which is
exactly where the run halts.
"""

from collections.abc import Callable

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, Matcher
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy, Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import Payload
from plumbline.proxy import RecordingProxy

_SCENES = ("scene_human", "scene_obstacle")
_RECORDED = {
    "scene_human": "a person stands one meter directly ahead and appears calm and clearly curious",
    "scene_obstacle": "a solid obstacle sits about forty centimeters to the left side of the robot",
}
_COMPATIBLE = {  # paraphrase: one word changed, shares almost all tokens
    "scene_human": "a human stands one meter directly ahead and appears calm and clearly curious",
    "scene_obstacle": (
        "a solid obstacle sits about forty centimeters to the left flank of the robot"
    ),
}
_INCOMPATIBLE = "empty hallway extends forward with clear flooring and no hazards detected anywhere"


def _upstream(request: Payload) -> Payload:
    inline = request.inline
    assert isinstance(inline, dict)
    if inline.get("kind") == "caption":
        scene = inline["scene"]
        assert isinstance(scene, str)
        return Payload(inline={"caption": _RECORDED[scene]})
    return Payload(inline={"action_plan": {"action": "avoid", "args": {}}})


def _classify(request: Payload, ctx: Context) -> Seam:
    inline = request.inline
    if isinstance(inline, dict) and inline.get("kind") == "caption":
        return Seam.SENSOR_TO_CAPTION
    return Seam.FUSE_TO_DECIDE


def _record_episode() -> TraceStore:
    store = TraceStore()
    proxy = RecordingProxy(_upstream, Recorder(store, VirtualClock()), classifier=_classify)
    for tick, scene in enumerate(_SCENES):
        ctx = Context(episode_id="ep", model_id="vlm", params={}, logical_tick=tick)
        caption = proxy.forward(Payload(inline={"kind": "caption", "scene": scene}), ctx)
        proxy.forward(Payload(inline={"kind": "decide", "prompt": str(caption.inline)}), ctx)
    proxy.close("ep")
    return store


def _captioner(captions: dict[str, str] | str) -> Callable[[Payload], Payload]:
    def override(request: Payload) -> Payload:
        if isinstance(captions, str):
            return Payload(inline={"caption": captions})
        inline = request.inline
        assert isinstance(inline, dict)
        scene = inline["scene"]
        assert isinstance(scene, str)
        return Payload(inline={"caption": captions[scene]})

    return override


def _matchers() -> dict[Seam, Matcher]:
    return {
        Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=0.2),
        Seam.DECIDE_TO_ACT: ExactMatcher(),
    }


def test_proxy_recorded_episode_groups_seams_by_logical_tick() -> None:
    events = _record_episode().load_episode("ep").events
    by_tick: dict[int, set[Seam]] = {}
    for event in events:
        by_tick.setdefault(event.logical_tick, set()).add(event.seam)
    # Both model seams of each iteration share a tick (not one tick per call).
    assert by_tick == {
        0: {Seam.SENSOR_TO_CAPTION, Seam.FUSE_TO_DECIDE},
        1: {Seam.SENSOR_TO_CAPTION, Seam.FUSE_TO_DECIDE},
    }


def test_compatible_captioner_swap_reproduces_without_divergence() -> None:
    store = _record_episode()
    result = Replayer(store, VirtualClock(), _matchers()).counterfactual(
        "ep",
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner(dict(_COMPATIBLE))},
        on_divergence=DivergencePolicy.HALT,
    )
    assert result.diverged is False
    assert result.divergence_seam is None
    assert sum(e.seam is Seam.FUSE_TO_DECIDE for e in result.events) == len(_SCENES)


def test_incompatible_captioner_swap_halts_at_fuse_to_decide() -> None:
    store = _record_episode()
    result = Replayer(store, VirtualClock(), _matchers()).counterfactual(
        "ep",
        live_frontier={Seam.SENSOR_TO_CAPTION},
        overrides={Seam.SENSOR_TO_CAPTION: _captioner(_INCOMPATIBLE)},
        on_divergence=DivergencePolicy.HALT,
    )
    assert result.diverged is True
    assert result.divergence_seam is Seam.FUSE_TO_DECIDE
    assert result.divergence_distance is not None
    assert result.divergence_distance > 0.2
    assert all(e.seam is not Seam.FUSE_TO_DECIDE for e in result.events)
