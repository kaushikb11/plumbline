"""Experiment B on real models via Ollama — "existing observability says fine,
Plumbline says broken, and the robot WAS broken". No robot, no simulator.

Records two runs over the SAME obstacle corridor that differ ONLY in the perception
front-end: a wide field of view that sees the floor obstacle (-> an avoid action) vs
a narrow, up-pitched crop that drops it from the caption (-> move forward). Same VLM
weights, same decider, comparable latency, well-formed calls throughout — so a
latency dashboard and an OTel-GenAI tracer both stay GREEN while Plumbline's
behavior monitor goes RED on the action inversion.

    pip install "plumbline[proxy]" pillow
    ollama pull moondream && ollama pull llama3.2:1b
    python examples/experiment_b.py

Override the endpoint/models with PLUMBLINE_OLLAMA_URL / PLUMBLINE_VLM /
PLUMBLINE_DECIDER.

Honest caveats: synthetic corridor scenes, small local models, per-run and
nondeterministic — the decision inversion is guarded, not guaranteed every run. The
narrow crop feeds fewer vision tokens, so the candidate can run *faster*, not just
slower — either can push mean model latency outside the monitor's tolerance and trip
it; rerun. The Experiment-B property (baselines green while
Plumbline red) is proven deterministically in tests/test_baselines.py; this file is
the live-model demonstration, not the proof. No wall-clock / scheduler determinism
is claimed (invariant 4, §3.4) — only the model I/O and the derived action sequence.
"""

import base64
import io
import os
import time
from collections.abc import Mapping

import httpx
from PIL import Image, ImageDraw
from plumbline.adapters.generic import GenericAgentAdapter
from plumbline.bench.leaderboard import LabeledScene
from plumbline.bench.openai_client import chat_captioner, chat_decider
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.observability.baselines import compare_against_baselines

try:
    from examples._env import friendly_endpoint
except ImportError:  # `python examples/experiment_b.py`: examples/ is on sys.path, not repo root
    from _env import friendly_endpoint

BASE_URL = os.environ.get("PLUMBLINE_OLLAMA_URL", "http://localhost:11434/v1")
VLM = os.environ.get("PLUMBLINE_VLM", "moondream")
DECIDER = os.environ.get("PLUMBLINE_DECIDER", "llama3.2:1b")
PROMPT = (
    "Look at this robot's forward camera. In ONE sentence: "
    "is the path ahead blocked by an object, or clear?"
)


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _corridor(obstacle: bool) -> Image.Image:
    w = h = 320
    image = Image.new("RGB", (w, h), (210, 215, 220))
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(0, h), (w, h), (int(w * 0.66), int(h * 0.55)), (int(w * 0.34), int(h * 0.55))],
        fill=(150, 150, 155),
    )
    if obstacle:
        draw.rectangle(
            [int(w * 0.40), int(h * 0.52), int(w * 0.60), int(h * 0.85)], fill=(25, 25, 25)
        )
    return image


def narrow_fov(scene: LabeledScene, frac: float = 0.5) -> LabeledScene:
    """An up-pitched / narrow camera: crop away the lower frame — and the floor
    obstacle with it — so this front-end never sees the object."""
    encoded = scene.image.split(";base64,", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    w, h = image.size
    return LabeledScene(
        scene.scene_id, _data_url(image.crop((0, 0, w, int(h * frac)))), scene.render_g
    )


def build_run(
    episode_id: str,
    *,
    caption: str,
    decision: Mapping[str, JSONValue],
    cap_latency_ms: float,
    dec_latency_ms: float,
) -> list[SeamEvent]:
    """Assemble the four seams for one run from already-computed values (no I/O) —
    the pure core, so the property is offline-testable."""
    adapter = GenericAgentAdapter(proxy_base_url=BASE_URL)
    cap_request = Payload(inline={"instruction": PROMPT, "scene_id": episode_id})
    decide_request = Payload(inline={"prompt": caption})
    return [
        SeamEvent(
            episode_id=episode_id,
            seq=0,
            seam=Seam.SENSOR_TO_CAPTION,
            logical_tick=0,
            wall_ts=0.0,
            request=cap_request,
            response=Payload(inline={"caption": caption}),
            model_id=VLM,
            params={"temperature": 0.0},
            request_digest=canonicalize(cap_request).digest,
            latency_ms=cap_latency_ms,
        ),
        adapter.reconstruct_caption_to_fuse(
            episode_id=episode_id, seq=1, logical_tick=0, captions=[caption], fused_prompt=caption
        ),
        SeamEvent(
            episode_id=episode_id,
            seq=2,
            seam=Seam.FUSE_TO_DECIDE,
            logical_tick=0,
            wall_ts=0.0,
            request=decide_request,
            response=Payload(inline=dict(decision)),
            model_id=DECIDER,
            params={"temperature": 0.0},
            request_digest=canonicalize(decide_request).digest,
            latency_ms=dec_latency_ms,
        ),
        adapter.reconstruct_decide_to_act(
            episode_id=episode_id,
            seq=3,
            logical_tick=0,
            decision_response=Payload(inline=dict(decision)),
        ),
    ]


def main() -> None:
    scene = LabeledScene("obstacle-01", _data_url(_corridor(obstacle=True)), "blocked")
    client = httpx.Client(timeout=180.0)
    captioner = chat_captioner(client, BASE_URL, VLM, instruction=PROMPT)
    decider = chat_decider(client, BASE_URL, DECIDER, temperature=0.0)

    start = time.perf_counter()
    wide_caption = captioner(scene)
    cap_g = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    narrow_caption = captioner(narrow_fov(scene))
    cap_c = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    golden_decision = decider(wide_caption)
    dec_g = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    candidate_decision = decider(narrow_caption)
    dec_c = (time.perf_counter() - start) * 1000.0

    golden = build_run(
        "golden",
        caption=wide_caption,
        decision=golden_decision,
        cap_latency_ms=cap_g,
        dec_latency_ms=dec_g,
    )
    candidate = build_run(
        "candidate",
        caption=narrow_caption,
        decision=candidate_decision,
        cap_latency_ms=cap_c,
        dec_latency_ms=dec_c,
    )

    print("Experiment B — obstacle corridor; golden = wide FOV, candidate = narrow FOV\n")
    print(f"  golden   : caption={wide_caption!r} -> action={golden_decision.get('action')}")
    print(f"  candidate: caption={narrow_caption!r} -> action={candidate_decision.get('action')}")
    golden_ms, candidate_ms = (cap_g + dec_g) / 2, (cap_c + dec_c) / 2
    print(f"  mean model latency golden={golden_ms:.0f}ms candidate={candidate_ms:.0f}ms\n")

    comparison = compare_against_baselines(golden, candidate)
    for verdict in comparison.verdicts:
        print(f"  [{'GREEN' if verdict.healthy else 'RED  '}] {verdict.name}: {verdict.detail}")
    print(f"\n  caught_by = {comparison.caught_by}")
    print(f"  missed_by = {comparison.missed_by}")

    if golden_decision.get("action") == candidate_decision.get("action"):
        print("\n  NOTE: this run did not reproduce the decision inversion (noise) — rerun.")
        print("  The property is proven deterministically in tests/test_baselines.py.")
    else:
        print("\n  Existing observability green; Plumbline red; the robot moved at the obstacle.")


if __name__ == "__main__":
    with friendly_endpoint("Ollama", BASE_URL, hint="Is Ollama running? (ollama serve)"):
        main()
