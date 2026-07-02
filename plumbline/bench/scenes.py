"""Author a scenes.json for Experiment C from a folder of images + labels.

Turns your own photos (~15 obstacle / clear scenes) into the labeled input
`run_captioner_leaderboard` / `load_scenes` expect, so a real run is a folder plus
a one-line label per image — see docs/experiment-c.md.

    labels.json:  {"hall-01.jpg": "a solid obstacle is 0.4 m dead ahead; blocked",
                   "hall-02.jpg": "the corridor is clear for at least 3 m"}

    plumbline scenes ./images labels.json -o scenes.json

Keep `render_g` caption-agnostic and identical regardless of the captioner under
test (§14.5) — a terse factual ground-truth description, not fluent prose.
"""

import base64
import json
import mimetypes
from collections.abc import Mapping
from pathlib import Path

from plumbline.bench.leaderboard import LabeledScene


def encode_image(path: str | Path) -> str:
    """Encode an image file as a `data:<mime>;base64,...` URL for the VLM."""
    file = Path(path)
    mime = mimetypes.guess_type(file.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(file.read_bytes()).decode("ascii")


def build_scenes(image_dir: str | Path, labels: Mapping[str, str]) -> tuple[LabeledScene, ...]:
    """One `LabeledScene` per (image filename -> render_g) entry; `scene_id` is the
    file stem. Raises if a named image is missing."""
    directory = Path(image_dir)
    scenes: list[LabeledScene] = []
    for filename, render_g in labels.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        scenes.append(
            LabeledScene(scene_id=path.stem, image=encode_image(path), render_g=str(render_g))
        )
    return tuple(scenes)


def write_scenes_json(scenes: tuple[LabeledScene, ...], out_path: str | Path) -> None:
    """Write scenes in the format `load_scenes` reads back."""
    payload = [
        {"scene_id": scene.scene_id, "image": scene.image, "render_g": scene.render_g}
        for scene in scenes
    ]
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
