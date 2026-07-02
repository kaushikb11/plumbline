"""Captioner-for-decisions leaderboard — Experiment C (engineering spec §4, §7.6).

The thesis, as a checkable result: the captioner that preserves the decision wins,
even against a more fluent caption that drops the task-relevant fact. Uses a
deterministic content-only decider (sigma = 0) so the loss values are exact.
"""

import json
from collections.abc import Mapping
from pathlib import Path

from plumbline.bench.leaderboard import (
    CaptionerSpec,
    LabeledScene,
    load_scenes,
    run_captioner_leaderboard,
)
from plumbline.core.trace import JSONValue

# One scene: ground truth is a close obstacle -> the correct decision is to stop.
_SCENE = LabeledScene(
    scene_id="hallway-1",
    image="data:image/png;base64,AAAA",
    render_g="a solid obstacle is directly ahead within one meter",
)


def _content_only_decider(context: str) -> Mapping[str, JSONValue]:
    """Decides on the informational content: an obstacle -> stop, else move on."""
    blocked = "obstacle" in context or "blocked" in context
    return {"action": "stop" if blocked else "move_forward"}


def test_leaderboard_ranks_by_decision_fidelity_not_fluency() -> None:
    accurate = CaptionerSpec("accurate", lambda _s: "obstacle right ahead")
    # More fluent, but drops the obstacle -> the robot would move into it.
    fluent_but_blind = CaptionerSpec(
        "fluent", lambda _s: "a beautifully lit corridor stretches invitingly forward"
    )

    board = run_captioner_leaderboard(
        [_SCENE], [fluent_but_blind, accurate], _content_only_decider, n=16
    )

    # The decision-preserving captioner wins, though it is the less fluent one.
    assert board.best.name == "accurate"
    assert board.best.mean_caption_loss == 0.0
    assert board.best.decision_fidelity == 1.0
    assert board.scores[-1].name == "fluent"
    assert board.scores[-1].mean_caption_loss == 1.0  # decision flipped -> maximal loss


def test_leaderboard_averages_over_scenes() -> None:
    scenes = [
        LabeledScene("s1", "data:,x", "an obstacle is directly ahead"),
        LabeledScene("s2", "data:,y", "the path is clear and open"),
    ]
    # Captions the obstacle scene wrong ("clear") but the clear scene right.
    spec = CaptionerSpec("mixed", lambda scene: "clear path" if scene.scene_id == "s1" else "clear")
    score = run_captioner_leaderboard(scenes, [spec], _content_only_decider, n=16).best
    assert score.per_scene["s1"] == 1.0  # wrong on the obstacle scene
    assert score.per_scene["s2"] == 0.0  # right on the clear scene
    assert score.mean_caption_loss == 0.5


def test_oracle_distribution_is_sampled_once_per_scene_not_per_captioner() -> None:
    # The optimization: D(render_g) and sigma depend only on the scene, so they are
    # sampled once per scene (n for D(render_g) + 2n for the floor, which draws 2N to
    # match the numerator's sample size), while each caption is sampled per captioner.
    calls: dict[str, int] = {}

    def counting_decider(context: str) -> Mapping[str, JSONValue]:
        calls[context] = calls.get(context, 0) + 1
        return {"action": "stop" if "obstacle" in context else "move_forward"}

    scenes = [
        LabeledScene("s1", "data:,x", "an obstacle ahead"),
        LabeledScene("s2", "data:,y", "clear path open"),
    ]
    captioners = [
        CaptionerSpec("a", lambda scene: f"cap-a-{scene.scene_id}"),
        CaptionerSpec("b", lambda scene: f"cap-b-{scene.scene_id}"),
    ]
    n = 4
    run_captioner_leaderboard(scenes, captioners, counting_decider, n=n)

    render_gs = {scene.render_g for scene in scenes}
    oracle_calls = sum(count for context, count in calls.items() if context in render_gs)
    caption_calls = sum(count for context, count in calls.items() if context.startswith("cap-"))
    # 3n per scene (n for D(render_g) + 2n for the floor), NOT multiplied by captioners.
    assert oracle_calls == 3 * n * len(scenes)
    # n per (captioner, scene) for D(caption).
    assert caption_calls == n * len(captioners) * len(scenes)


def test_load_scenes_reads_a_json_list(tmp_path: Path) -> None:
    path = tmp_path / "scenes.json"
    path.write_text(
        json.dumps([{"scene_id": "a", "image": "data:,z", "render_g": "clear"}]),
        encoding="utf-8",
    )
    scenes = load_scenes(path)
    assert len(scenes) == 1
    assert scenes[0].scene_id == "a"
    assert scenes[0].render_g == "clear"
