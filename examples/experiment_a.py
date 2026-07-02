"""Experiment A on real models via Ollama — the caption verbosity/fidelity curve.

Generates corridor scenes with an obstacle, asks a real VLM for a detailed caption,
then sweeps a degradation knob (progressively truncating the caption) and plots
downstream DECISION fidelity against a surface text-similarity metric. The point: a
surface metric is blind to WHICH words carry the decision, so it is a poor proxy for
decision preservation — decision fidelity can collapse while surface similarity is
still high. No robot, no simulator.

    pip install -e ".[proxy]" pillow
    ollama pull moondream && ollama pull llama3.2:1b
    python examples/experiment_a.py

Overrides: PLUMBLINE_OLLAMA_URL / PLUMBLINE_VLM / PLUMBLINE_DECIDER. This is a
synthetic illustration of the METRIC divergence (§4, §14.5), not an absolute
bandwidth constant; `render_g` is the caption-agnostic dataset label.

Honest caveat: `truncate` drops trailing tokens, so the divergence magnitude
depends on where the VLM places the obstacle clause in its caption (a model that
front-loads it will show little divergence). The robust, knob-independent finding
is the blindness of the surface metric — see the unit tests — not the exact number.
"""

import base64
import io
import os

import httpx
from PIL import Image, ImageDraw
from plumbline.bench.leaderboard import LabeledScene
from plumbline.bench.openai_client import chat_captioner, chat_decider
from plumbline.bench.verbosity import linspace, run_verbosity_sweep

BASE_URL = os.environ.get("PLUMBLINE_OLLAMA_URL", "http://localhost:11434/v1")
VLM = os.environ.get("PLUMBLINE_VLM", "moondream")
DECIDER = os.environ.get("PLUMBLINE_DECIDER", "llama3.2:1b")
# Ask for a DETAILED, multi-clause caption so there are trailing tokens to drop.
PROMPT = (
    "Describe this robot's forward camera view in detail: the scene, the lighting, "
    "and critically whether any object blocks the path ahead and where it is."
)


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _corridor_with_obstacle() -> Image.Image:
    w = h = 320
    image = Image.new("RGB", (w, h), (210, 215, 220))
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(0, h), (w, h), (int(w * 0.66), int(h * 0.55)), (int(w * 0.34), int(h * 0.55))],
        fill=(150, 150, 155),
    )
    draw.rectangle([int(w * 0.40), int(h * 0.52), int(w * 0.60), int(h * 0.85)], fill=(25, 25, 25))
    return image


def main() -> None:
    scenes = [
        LabeledScene(
            "obstacle-01",
            _data_url(_corridor_with_obstacle()),
            "a solid object is 0.4 m directly ahead on the floor; the path is blocked",
        )
    ]
    client = httpx.Client(timeout=180.0)
    captioner = chat_captioner(client, BASE_URL, VLM, instruction=PROMPT)
    decider = chat_decider(client, BASE_URL, DECIDER, temperature=0.2)

    curve = run_verbosity_sweep(scenes, captioner, decider, levels=linspace(0.0, 1.0, 11), n=8)
    print(curve.as_table())
    print(f"\ndivergence (max surface_similarity - decision_fidelity): {curve.divergence:.3f}")
    knee = curve.knee()
    if knee is not None:
        print(
            f"knee: at level {knee.level:.2f} the decision has collapsed "
            f"(fidelity {knee.decision_fidelity:.2f}) while surface_similarity is still "
            f"{knee.surface_similarity:.2f}"
        )


if __name__ == "__main__":
    main()
