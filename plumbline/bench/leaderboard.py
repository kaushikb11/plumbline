"""Captioner-for-decisions leaderboard — Experiment C (engineering spec §4, §7.6).

Rank candidate VLM captioners by downstream *decision* success, not caption
surface quality: for each labeled scene a captioner produces a caption, and its
`caption_loss` (§7.3) measures how far a decider acting on that caption diverges
from acting on ground truth — where `render(G)` is the scene's own label. The best
captioner is the one whose captions preserve the decision, which is not the same
as the most fluent caption. That is the whole point of Experiment C.

No robot and no simulator: ground truth comes from the dataset's labels, and the
captioner and decider are real models reached through an OpenAI-compatible
endpoint (a local Ollama or a hosted provider) — see `plumbline.bench.openai_client`.

    from plumbline.bench.leaderboard import CaptionerSpec, load_scenes, run_captioner_leaderboard
    from plumbline.bench.openai_client import chat_captioner, chat_decider
    import httpx

    url = "http://localhost:11434/v1"          # local Ollama, free, no keys
    client = httpx.Client(base_url=url)
    scenes = load_scenes("scenes.json")        # PhysBench-derived labeled scenes
    decider = chat_decider(client, url, "llama3.2")
    captioners = [
        CaptionerSpec("llava", chat_captioner(client, url, "llava")),
        CaptionerSpec("llama3.2-vision", chat_captioner(client, url, "llama3.2-vision")),
    ]
    board = run_captioner_leaderboard(scenes, captioners, decider, n=16)
    print(board.as_table())
"""

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from plumbline.fidelity import (
    DeciderFn,
    DecisionLabel,
    Distribution,
    Divergence,
    canonical_label,
    decision_distribution,
    decision_stability,
    total_variation,
)

# A captioner: a labeled scene -> a caption (a real VLM in practice).
Captioner = Callable[["LabeledScene"], str]


@dataclass(frozen=True)
class LabeledScene:
    """One labeled scene: an image and its ground-truth oracle context (§7.3, §14.5).

    `render_g` is `render(G)` — a caption-agnostic description of ground truth (the
    dataset label), NOT phrased like a captioner would, so it does not flatter the
    metric.
    """

    scene_id: str
    image: str  # data URL or http(s) URL, passed to the VLM as image_url
    render_g: str


@dataclass(frozen=True)
class CaptionerSpec:
    name: str
    caption: Captioner


@dataclass(frozen=True)
class CaptionerScore:
    name: str
    mean_caption_loss: float  # E[caption_loss] over scenes; lower is better for decisions
    per_scene: Mapping[str, float]

    @property
    def decision_fidelity(self) -> float:
        return 1.0 - self.mean_caption_loss


@dataclass(frozen=True)
class Leaderboard:
    scores: tuple[CaptionerScore, ...]  # ranked best-first (lowest caption_loss)

    @property
    def best(self) -> CaptionerScore:
        return self.scores[0]

    def as_table(self) -> str:
        return "\n".join(
            f"{rank}. {score.name}: decision_fidelity={score.decision_fidelity:.3f} "
            f"(mean caption_loss={score.mean_caption_loss:.3f})"
            for rank, score in enumerate(self.scores, start=1)
        )


def run_captioner_leaderboard(
    scenes: Sequence[LabeledScene],
    captioners: Sequence[CaptionerSpec],
    decider: DeciderFn,
    *,
    n: int = 32,
    label: DecisionLabel = canonical_label,
    divergence: Divergence = total_variation,
) -> Leaderboard:
    """Score each captioner by mean `caption_loss` across the scenes and rank them
    best-first (§4 Experiment C).

    The oracle decision distribution `D(render(G))` and the noise floor `sigma`
    depend only on the scene, not the captioner, so they are sampled once per scene
    and shared — this is `caption_loss` (§7.3) inlined to avoid re-sampling the
    (real, billed) decider once per captioner. For a deterministic decider the
    result is identical to per-captioner `caption_loss`; for a stochastic one the
    shared oracle estimate is also more consistent across captioners.
    """
    oracle: dict[str, tuple[Distribution, float]] = {}
    for scene in scenes:
        oracle[scene.scene_id] = (
            decision_distribution(decider, scene.render_g, n, label=label),
            decision_stability(decider, scene.render_g, n, label=label, divergence=divergence),
        )

    scored: list[CaptionerScore] = []
    for spec in captioners:
        per_scene: dict[str, float] = {}
        for scene in scenes:
            d_caption = decision_distribution(decider, spec.caption(scene), n, label=label)
            d_oracle, sigma = oracle[scene.scene_id]
            per_scene[scene.scene_id] = max(0.0, divergence(d_caption, d_oracle) - sigma)
        mean = statistics.fmean(per_scene.values()) if per_scene else 0.0
        scored.append(CaptionerScore(spec.name, mean, per_scene))
    scored.sort(key=lambda score: score.mean_caption_loss)
    return Leaderboard(tuple(scored))


def load_scenes(path: str | Path) -> tuple[LabeledScene, ...]:
    """Load labeled scenes from a JSON list of {scene_id, image, render_g}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("scenes JSON must be a list of {scene_id, image, render_g} objects")
    return tuple(
        LabeledScene(
            scene_id=str(entry["scene_id"]),
            image=str(entry["image"]),
            render_g=str(entry["render_g"]),
        )
        for entry in raw
    )
