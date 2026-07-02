"""Benchmark: golden-episode fixtures and the captioner leaderboard (spec §4, §10).

Exports the dependency-free leaderboard API. The real-model client
(`plumbline.bench.openai_client`) and the example gate config are imported
directly, since they pull optional extras.
"""

from plumbline.bench.leaderboard import (
    Captioner,
    CaptionerScore,
    CaptionerSpec,
    LabeledScene,
    Leaderboard,
    load_scenes,
    run_captioner_leaderboard,
)

__all__ = [
    "Captioner",
    "CaptionerScore",
    "CaptionerSpec",
    "LabeledScene",
    "Leaderboard",
    "load_scenes",
    "run_captioner_leaderboard",
]
