# Concepts — the one-page mental model

Read this first. It is the whole idea in one arc: **what the four seams are**, and **how record → faithful-replay → counterfactual → gate fit together.** Everything else in the docs is detail on one of these pieces.

## Why this exists

A language-bus robot runtime (OpenMind's [OM1](https://github.com/OpenMind/OM1) is the reference) turns sensor streams into natural-language captions, fuses them into one prompt at roughly 1 Hz, and hands that prompt to a Cortex LLM that decides what the robot does. Every model in that loop — the VLM captioner, the ASR, the Cortex LLM — is a nondeterministic external dependency (usually a cloud API sampling at non-zero temperature). So you cannot reproduce a run, cannot regression-test a model or prompt change, and cannot measure how much task-relevant information survives the language bottleneck.

Plumbline records the nondeterministic model calls at the loop's four seams and replays them, so the decision/action sequence is reproducible — then it lets you swap one component and measure what changed.

## The four seams

The perception-to-action loop is a linear pipeline; each seam's input is the previous seam's output:

```
sensors ─▶ caption (VLM / ASR) ─▶ fuse (captions + rules + RAG → one prompt)
        ─▶ decide (Cortex LLM → action plan) ─▶ act (orchestrator → HAL) ─▶ sensors …
```

| Seam | Captured request | Captured response | How it is captured |
|------|------------------|-------------------|--------------------|
| `SENSOR_TO_CAPTION` | raw frame / audio / state | caption text | HTTP proxy (a live model call) |
| `CAPTION_TO_FUSE` | captions + rules + RAG | fused prompt | **derived** — no model call of its own |
| `FUSE_TO_DECIDE` | fused prompt | action plan | HTTP proxy (a live model call) |
| `DECIDE_TO_ACT` | action plan | HAL commands | bus tap (Zenoh / ROS2), and derived |

The bus is already text and already the architecture's narrow waist, so **three of four seams come from a recording HTTP proxy and one from a bus tap.** Two seams (`SENSOR_TO_CAPTION`, `FUSE_TO_DECIDE`) are *classified* from a live model call; the other two are *reconstructed* from already-captured payloads (see [writing-an-adapter.md](writing-an-adapter.md)). These four names are frozen — they are part of the contribution.

## The lifecycle

One arc, four stages, each validating the next:

```
        ┌─────────────────────────────────────────────────────────────────────┐
        │  1. RECORD                                                           │
        │  Runtime's model calls go through the proxy (base URL redirect,      │
        │  zero source changes). Each call: forward to the real endpoint,      │
        │  capture request+response, infer the seam, emit a SeamEvent,         │
        │  return the upstream response UNALTERED (the zero-touch invariant).  │
        │  → a trace on disk: episodes/<id>/events.jsonl (JSON, never pickle)  │
        └───────────────────────────────┬─────────────────────────────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                                ▼
  ┌──────────────────────────────┐          ┌──────────────────────────────────────┐
  │ 2. FAITHFUL REPLAY           │          │ 3. COUNTERFACTUAL REPLAY               │
  │ Serve every seam from the    │          │ Re-run ONE seam live (the "live         │
  │ trace by request digest.     │          │ frontier", e.g. a swapped captioner);   │
  │ Re-drive the runtime and you │          │ pin the rest to the trace. If the swap  │
  │ get bit-identical model I/O  │          │ changes a downstream seam's input past  │
  │ and the same decisions —     │          │ its matcher's tolerance, HALT, record   │
  │ no model calls made.         │          │ the divergence seam + distance, and     │
  │ The reproducibility oracle.  │          │ serve nothing stale past it.            │
  └──────────────┬───────────────┘          └────────────────────┬───────────────────┘
                 │                                                │
                 └────────────────────────┬───────────────────────┘
                                          ▼
                    ┌────────────────────────────────────────────────┐
                    │ 4. GATE                                        │
                    │ Counterfactual-replay a set of golden episodes  │
                    │ under a candidate config (a swapped model /     │
                    │ edited prompt as seam overrides). Score behavior │
                    │ drift from the accepted action sequence; fail   │
                    │ CI past a threshold. Optionally score DECISION   │
                    │ divergence anchored to the noise floor.         │
                    └────────────────────────────────────────────────┘
```

**Record** validates the substrate; **faithful replay** is the reproducibility oracle (byte-identical model I/O); **counterfactual replay** isolates the effect of one component change and halts on divergence; **the gate** turns all of it into a CI check that catches a silent behavior regression a latency dashboard cannot see.

## Two hard rules that shape everything

- **The determinism envelope is model-I/O only.** On replay every model call receives the recorded request and returns the recorded response, so the decision/action sequence is reproduced. Plumbline does **not** control the runtime's wall-clock scheduler unless an adapter exposes a clock hook. Loop *timing* may vary across replays; model *I/O* does not. See [determinism-envelope.md](determinism-envelope.md).
- **Halt-on-divergence is the default, and divergence is a result, not an error.** In counterfactual replay, when a downstream seam's live input no longer matches the recording, Plumbline stops, records the seam and distance, and returns that — it never silently serves a stale recorded response past a divergence.

## Where to go next

- [quickstart.md](quickstart.md) — run the pieces above with real code.
- [api.md](api.md) — the exact types and signatures.
- [writing-an-adapter.md](writing-an-adapter.md) — teach Plumbline a new runtime.
- [limitations.md](limitations.md) — the honest scope audit: what works, what's scoped, what isn't built.
