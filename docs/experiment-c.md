# Experiment C — the captioner-for-decisions leaderboard (no robot)

Rank candidate VLM captioners by **downstream decision success**, not caption
surface quality — the result that "the best caption by NLP metrics is not the best
caption for behavior" (engineering spec §4, §7.6). This runs on a laptop with real
models and **no robot and no simulator**: ground truth comes from labeled images,
and the models are reached through an OpenAI-compatible endpoint (a local Ollama —
free, no keys — or a hosted provider).

## What it produces

A ranked leaderboard where each captioner's score is its mean `caption_loss`
(§7.3) — how far a decider *acting on that captioner's caption* diverges from
acting on ground truth, corrected for the decider's own sampling noise:

```
1. terse-accurate:  decision_fidelity=0.94 (mean caption_loss=0.06)
2. fluent-verbose:  decision_fidelity=0.71 (mean caption_loss=0.29)
```

## Prerequisites

```bash
pip install -e ".[proxy]"          # httpx (the model client)
# Free local models (recommended): https://ollama.com
ollama pull llava                   # a VLM captioner (or: moondream, llama3.2-vision)
ollama pull llama3.2                # the Cortex decider (any small chat model)
```

Ollama serves an OpenAI-compatible API at `http://localhost:11434/v1`, which is
exactly what `chat_captioner` / `chat_decider` (and the proxy's OpenAI normalizer)
speak. No API keys.

## Step 1 — build `scenes.json`

Each scene is an image plus its **ground-truth oracle context** `render_g`
(= `render(G)`). Format:

```json
[
  {"scene_id": "hall-01",
   "image": "data:image/jpeg;base64,/9j/4AAQ...",
   "render_g": "a solid obstacle is 0.4 m directly ahead; the path is blocked"},
  {"scene_id": "hall-02",
   "image": "data:image/jpeg;base64,/9j/4AAQ...",
   "render_g": "the corridor ahead is clear for at least 3 m"}
]
```

`image` is a data URL (base64) or an `http(s)` URL the model can fetch. To encode
local image files:

```python
import base64, json, pathlib

def data_url(path: str, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(pathlib.Path(path).read_bytes()).decode()

scenes = [
    {"scene_id": "hall-01", "image": data_url("images/hall-01.jpg"),
     "render_g": "a solid obstacle is 0.4 m directly ahead; the path is blocked"},
    # ...
]
pathlib.Path("scenes.json").write_text(json.dumps(scenes))
```

**Start with ~10–20 hand-labeled images** (obstacle vs. clear scenes map cleanly to
the stop/move decision). Scale later to a benchmark: HomeSafeBench (safety-hazard
labels) or PhysBench (physical-scene labels) on Hugging Face — deriving `render_g`
from their labels is the §14.5 work (see the honesty note below).

## Step 2 — run it

Save as `run_leaderboard.py`:

```python
import httpx
from plumbline.bench.leaderboard import CaptionerSpec, load_scenes, run_captioner_leaderboard
from plumbline.bench.openai_client import chat_captioner, chat_decider

URL = "http://localhost:11434/v1"          # local Ollama (free); or a hosted base URL
client = httpx.Client(base_url=URL, timeout=120.0)

scenes = load_scenes("scenes.json")
decider = chat_decider(client, URL, "llama3.2")   # one fixed decider for all captioners
captioners = [
    CaptionerSpec("llava",     chat_captioner(client, URL, "llava")),
    CaptionerSpec("moondream", chat_captioner(client, URL, "moondream")),
]

board = run_captioner_leaderboard(scenes, captioners, decider, n=16)
print(board.as_table())
print("best captioner for decisions:", board.best.name)
```

```bash
python run_leaderboard.py
```

## Step 3 — read the result

- `decision_fidelity = 1 - mean_caption_loss` — higher is better.
- `score.per_scene[scene_id]` — the per-scene loss, to find *which* scenes a
  captioner fails on.
- A loss near 0 means acting on that caption reproduces the ground-truth decision;
  a loss near 1 means the caption flipped the decision (dropped the task-relevant
  fact — the LiDAR-dog failure, as a number).
- The floor is handled for you: `caption_loss` subtracts the decider's
  decision-stability `sigma`, so a gap only counts if it exceeds the decider's own
  self-disagreement (§7.2). Loss is never negative.

## Cost and performance

Each `(captioner, scene)` costs **1 caption call + ~3·n decider calls**
(sampling `D(caption)`, `D(render_g)`, and the noise floor). With `n=16`, two
captioners, twenty scenes that is ~2,000 model calls — free but slow on Ollama.

- Start with `n=8` and ~10 scenes to shake out the pipeline, then scale.
- ⚠️ The oracle distribution and floor are currently recomputed per captioner even
  though they depend only on the scene — so adding captioners multiplies the oracle
  sampling. Precomputing them once per scene is a worthwhile optimization before a
  large run (ask, and it is a small change to `run_captioner_leaderboard`).

## §14.5 — keep `render_g` honest

`render_g` is the one place this experiment can flatter itself. It must be a
**caption-agnostic, structured description of ground truth** — an object/obstacle
inventory from the dataset label — and **identical regardless of which captioner is
under test**. Do *not* phrase it the way a good captioner would (fluent prose),
or a captioner that happens to match that phrasing scores artificially well. Prefer
terse, factual `render_g` ("nearest obstacle 0.4 m, dead ahead") over narration.

## Hosted providers instead of Ollama

Point `URL` at any OpenAI-compatible base and pass an `api_key`:

```python
client = httpx.Client(
    base_url="https://api.openai.com/v1",
    headers={"Authorization": "Bearer $OPENAI_API_KEY"},
    timeout=120.0,
)
captioners = [CaptionerSpec("gpt-4o", chat_captioner(client, "https://api.openai.com/v1", "gpt-4o"))]
```

Anthropic and Gemini work the same via their OpenAI-compatible endpoints (or the
proxy's Gemini/Anthropic normalizers). Mind the token budget — see the call count
above.

## Recording the run (optional, for reproducibility)

To make the eval itself replayable, route `client` through the recording proxy
(`plumbline.proxy.server`) so every captioner and decider call is captured to a
trace. Then a re-run serves the recorded responses instead of re-billing the
models. See `docs/quickstart.md` for the proxy wiring.
