# Experiment C — a real-model result (no robot, no simulator)

This is Plumbline's fidelity metric run end-to-end on **real models** — a real
vision model captioning real images, a real LLM deciding, sampled N times, scored
by `caption_loss` with the decision-stability noise floor. It runs on a laptop
with [Ollama](https://ollama.com), no robot and no simulator: ground truth comes
from the dataset labels (`render(G)`), not a sim.

Reproduce it with [`examples/experiment_c.py`](../examples/experiment_c.py):

```bash
pip install -e ".[proxy]" pillow
ollama pull moondream && ollama pull llama3.2:1b
python examples/experiment_c.py
```

## The setup

Four generated corridor scenes — two with a solid obstacle on the floor ahead, two
clear. Two perception front-ends of the **same** vision model (`moondream`) are
ranked by downstream decision fidelity against the **same** decider
(`llama3.2:1b`, temperature 0.2, n=8):

- **wide-fov** — the full forward image.
- **narrow-fov** — the image cropped to its upper half, as if the camera were
  pitched up or narrow. It cannot see the floor where the obstacle sits, so it
  drops the obstacle from its caption ("a gray and white striped wall" — object
  gone).

This is the caption/perception bottleneck as a controlled variable: same model,
same decider, same scenes — only the *information the perception front-end
preserves* differs.

## The result

```
1. moondream/wide-fov:   decision_fidelity=0.814 (mean caption_loss=0.186)
2. moondream/narrow-fov: decision_fidelity=0.752 (mean caption_loss=0.248)
```

| scene | wide-fov loss | narrow-fov loss |
|---|---|---|
| obstacle-01 | 0.109 | **0.234** |
| obstacle-02 | 0.105 | **0.355** |
| clear-01 | 0.004 | 0.004 |
| clear-02 | 0.523 | 0.398 |

**The wide field of view wins**, and the signal is exactly where it should be:
**on the two obstacle scenes, the narrow front-end's caption_loss is 2–3× higher**
(0.234 vs 0.109; 0.355 vs 0.105). Dropping the floor obstacle from the caption
flips the robot's decision from *stop* to *move*, and the metric charges precisely
that. This is the LiDAR-dog / caption-bottleneck failure as a number — a
*perception* limitation surfaced as a *decision* cost, which a text-quality or
latency check would miss entirely.

## Honest caveats

This validates the **mechanism and the metric on real models**; it is not a clean
benchmark, and it is reported raw:

- **The clear scenes are noisy.** `clear-02` shows ~0.5/0.4 loss for *both*
  captioners — that is `llama3.2:1b` sampling variance plus `moondream`
  hallucinating "a hole in the floor" on the empty corridors, which occasionally
  makes the tiny decider hesitate. That noise shrinks the aggregate gap (0.062)
  relative to the per-obstacle-scene gap. A stronger decider (`llama3.2` 3B) and
  higher `n` reduce it.
- **Synthetic images, tiny models.** The scenes are drawn corridors, not
  photographs, and the models are small (chosen to fit a laptop). Real photos and
  larger models sharpen the number; the *mechanism* is what is demonstrated here.
- **`render(G)` is the dataset label.** Kept caption-agnostic and identical across
  captioners (§14.5), so the metric cannot flatter itself.

For a noise-free illustration of the same thesis through the full network stack
(real HTTP, real N-sampling) with a scripted model stand-in, see the
`caption_loss` / leaderboard unit tests. The mechanism there is clean; this page
is the honest real-model version, noise and all.
