"""Experiment C on real models via Ollama — the runnable example behind
docs/results-experiment-c.md. No robot, no simulator.

It generates simple corridor scenes (some blocked by an obstacle on the floor),
then ranks two perception front-ends of the SAME vision model by *downstream
decision fidelity*: a wide field of view vs. a narrow (up-pitched) one that is
cropped so it cannot see the floor where the obstacle sits. The narrow front-end
drops the obstacle from its caption, and the metric charges it exactly where that
missing information flips the robot's decision.

Requires an Ollama server plus pillow + httpx:

    pip install "plumbline[proxy]" pillow
    ollama pull moondream && ollama pull llama3.2:1b
    python examples/experiment_c.py

Override the endpoint/models with PLUMBLINE_OLLAMA_URL / PLUMBLINE_VLM /
PLUMBLINE_DECIDER. For real cloud models, point the captioner and decider at separate
endpoints — e.g. Modal (modal/README.md):

    PLUMBLINE_VLM_URL=<vlm url>/v1 PLUMBLINE_VLM=captioner \\
    PLUMBLINE_LLM_URL=<llm url>/v1 PLUMBLINE_DECIDER=cortex python examples/experiment_c.py

Ground truth is the dataset labels (render(G)), not the sim.
"""

import base64
import io
import os

import httpx
from PIL import Image, ImageDraw
from plumbline.bench.leaderboard import CaptionerSpec, LabeledScene, run_captioner_leaderboard
from plumbline.bench.openai_client import chat_captioner, chat_decider

BASE_URL = os.environ.get("PLUMBLINE_OLLAMA_URL", "http://localhost:11434/v1")
# The captioner (VLM) and decider (LLM) can live at separate OpenAI-compatible
# endpoints — e.g. two Modal deployments (modal/vlm.py, modal/llm.py). Both default to
# BASE_URL so a single local Ollama still works unchanged.
VLM_URL = os.environ.get("PLUMBLINE_VLM_URL", BASE_URL)
LLM_URL = os.environ.get("PLUMBLINE_LLM_URL", BASE_URL)
VLM = os.environ.get("PLUMBLINE_VLM", "moondream")
DECIDER = os.environ.get("PLUMBLINE_DECIDER", "llama3.2:1b")
# Overridable because captioners differ in what phrasing they can answer: moondream
# handles the blocked-or-clear framing (docs/results-experiment-c.md), while e.g.
# Qwen2-VL-2B sees the obstacle but still answers "clear" under it — an object-grounded
# prompt ("state whether there is an object on the floor") is needed for the FOV
# comparison to measure FOV, not prompt-binding failure.
PROMPT = os.environ.get(
    "PLUMBLINE_CAPTION_PROMPT",
    "Look at this robot's forward camera. In ONE sentence: "
    "is the path ahead blocked by an object, or clear?",
)
_BLOCKED = "a large solid object is directly ahead on the floor, blocking the path"
_CLEAR = "the corridor ahead is open with no objects in the way"


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _corridor(obstacle: bool, box_color: tuple[int, int, int]) -> Image.Image:
    w = h = 320
    image = Image.new("RGB", (w, h), (210, 215, 220))
    draw = ImageDraw.Draw(image)
    floor = [(0, h), (w, h), (int(w * 0.66), int(h * 0.55)), (int(w * 0.34), int(h * 0.55))]
    far_wall = [
        (int(w * 0.34), int(h * 0.55)),
        (int(w * 0.66), int(h * 0.55)),
        (int(w * 0.62), 0),
        (int(w * 0.38), 0),
    ]
    draw.polygon(floor, fill=(150, 150, 155))
    draw.polygon(far_wall, fill=(190, 195, 200))
    if obstacle:
        draw.rectangle([int(w * 0.40), int(h * 0.52), int(w * 0.60), int(h * 0.85)], fill=box_color)
    return image


def build_scenes() -> tuple[LabeledScene, ...]:
    return (
        LabeledScene("obstacle-01", _data_url(_corridor(True, (25, 25, 25))), _BLOCKED),
        LabeledScene("obstacle-02", _data_url(_corridor(True, (120, 70, 40))), _BLOCKED),
        LabeledScene("clear-01", _data_url(_corridor(False, (0, 0, 0))), _CLEAR),
        LabeledScene("clear-02", _data_url(_corridor(False, (0, 0, 0))), _CLEAR),
    )


def narrow_fov(scene: LabeledScene, frac: float = 0.5) -> LabeledScene:
    """An up-pitched / narrow camera: crop away the lower frame — and with it the
    floor obstacle — so this perception front-end never sees the object."""
    encoded = scene.image.split(";base64,", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    w, h = image.size
    return LabeledScene(
        scene.scene_id, _data_url(image.crop((0, 0, w, int(h * frac)))), scene.render_g
    )


def main() -> None:
    scenes = build_scenes()
    client = httpx.Client(timeout=180.0)
    wide = chat_captioner(client, VLM_URL, VLM, instruction=PROMPT)
    narrow_base = chat_captioner(client, VLM_URL, VLM, instruction=PROMPT)

    def narrow(scene: LabeledScene) -> str:
        return narrow_base(narrow_fov(scene))

    decider = chat_decider(client, LLM_URL, DECIDER, temperature=0.2)
    board = run_captioner_leaderboard(
        scenes,
        [
            CaptionerSpec(f"{VLM}/wide-fov", wide),
            CaptionerSpec(f"{VLM}/narrow-fov", narrow),
        ],
        decider,
        n=8,
    )
    print(board.as_table())
    print(f"\nbest captioner for decisions: {board.best.name}")
    for score in board.scores:
        per_scene = {scene_id: round(loss, 3) for scene_id, loss in score.per_scene.items()}
        print(f"  {score.name} per-scene loss: {per_scene}")


if __name__ == "__main__":
    main()
