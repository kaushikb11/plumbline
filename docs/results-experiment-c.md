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

## The cloud-GPU replication (Modal, larger models)

The same experiment run against real GPU-served models on
[Modal](../modal/README.md) — `Qwen2.5-VL-7B-Instruct` captioner,
`Qwen2.5-3B-Instruct` decider (both vLLM, `modal deploy modal/vlm.py modal/llm.py`):

```bash
PLUMBLINE_VLM_URL=https://<ws>--plumbline-vlm-serve.modal.run/v1 PLUMBLINE_VLM=captioner \
PLUMBLINE_LLM_URL=https://<ws>--plumbline-llm-serve.modal.run/v1 PLUMBLINE_DECIDER=cortex \
python examples/experiment_c.py
```

```
1. captioner/wide-fov:   decision_fidelity=1.000 (mean caption_loss=0.000)
2. captioner/narrow-fov: decision_fidelity=0.500 (mean caption_loss=0.500)
```

| scene | wide-fov loss | narrow-fov loss |
|---|---|---|
| obstacle-01 | 0.0 | **1.0** |
| obstacle-02 | 0.0 | **1.0** |
| clear-01 | 0.0 | 0.0 |
| clear-02 | 0.0 | 0.0 |

With models strong enough to be reliable at this task, the separation is exact:
the narrow front-end is charged maximal loss on precisely the two scenes where
its cropped view drops the obstacle and the decision flips, and nothing else. At
temperature 0.2 this decider is effectively deterministic, so the distributions
are point masses and per-scene loss is 0-or-1 (σ ≈ 0); the laptop run above shows
the same mechanism under real sampling noise.

Two failures found *by the metric* on the way to this table are worth recording:

- **`Qwen2-VL-2B` is decision-blind here.** Asked "is the path blocked or
  clear?", it answers "clear" on the obstacle scene — while, asked to *describe*
  the image, it reports "a block or a box" in the center. It sees the object but
  cannot bind it to the blocked/clear judgment, so *both* front-ends were charged
  loss 1.0 on the obstacle scenes and the leaderboard (correctly) refused to
  separate them. A caption-surface metric would score its fluent captions well.
- **The decider needs the implication, not just the fact.** Given "there is an
  object on the floor ahead, a black box", `Qwen2.5-3B` still decides
  *move_forward*; only "…blocking the path" makes it back up. Task-relevant
  information must survive in decision-usable form — the founders' LiDAR-dog
  fix ("richer captioning") reproduced with off-the-shelf models.
