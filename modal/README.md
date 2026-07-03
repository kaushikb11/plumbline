# Testing Plumbline on Modal (no robot)

A [Modal](https://modal.com) account is enough to validate most of Plumbline against
**real, nondeterministic models** — no robot, no Gazebo. Three tiers, cheap → ambitious.

## Prereqs
- `pip install modal && modal setup` (auth once).
- `pip install -e ".[proxy]"` in this repo for the driver (needs `httpx`).

## Tier 1 — substrate + fidelity against real models (do this first)

Deploy the two OpenAI-compatible model endpoints (H100/A10G, **scale-to-zero**, per-second
billing — a few dollars):

```bash
modal deploy modal/llm.py    # the "Cortex" decider (tool-calling LLM)
modal deploy modal/vlm.py    # the captioner (vision-language model)
# each prints a URL: https://<workspace>--plumbline-{llm,vlm}-serve.modal.run
```

Then drive scenes through the recording proxy to the real models and validate:

```bash
PLUMBLINE_VLM_URL=<vlm url> PLUMBLINE_LLM_URL=<llm url> python examples/modal_validate.py
```

It records perception→decision cycles at **temperature 0.7** (real nondeterminism), then
faithful-replays and prints **PASS** iff the model I/O is byte-identical — the substrate's
core claim, validated against reality instead of stubs. The three identical vision requests
also exercise the digest-occurrence fix (a static scene sampled repeatedly records distinct
captions, replayed in record order). The same Modal VLM URL works as the captioner in
`examples/experiment_c.py` to rank real captioners by decision fidelity.

**Tuning:** `llm.py`/`vlm.py` are templates. Pick a model your GPU fits (A10G ≈ 24 GB),
and match the vLLM version + `--tool-call-parser` to the model family (Qwen2.5 → `hermes`,
Llama → `llama3_json`, …). These pins may need a bump as vLLM evolves.

## Tier 2 — WebSocket caption capture

```bash
modal deploy modal/ws_captions.py
# -> wss://<workspace>--plumbline-ws-captions-serve.modal.run/ws/captions

PLUMBLINE_WS_URL=wss://<workspace>--plumbline-ws-captions-serve.modal.run/ws/captions \
python examples/modal_ws_validate.py   # prints PASS iff replay is byte-identical
```

The driver dials the real `wss://` stream through `AsyncWSProxy` + `WebsocketsTransport`
(`plumbline/proxy/server.py`), records each caption frame as a `SENSOR_TO_CAPTION` event,
then faithful-replays with no upstream and asserts the served frames are byte-identical.
This validates the WebSocket half of limitations gap #1 against a real remote server —
and it caught a real bug the fakes missed (recorded JSON frames were re-serialized on
replay, changing their bytes; frames are now stored verbatim).

## Tier 3 — real OM1 + Gazebo (stretch)

`modal/gazebo_om1.py` is an **honest skeleton** (⚠️ NOT run/verified) of a custom Modal GPU
image — ROS2 + `go2_gazebo_sim` + `zenoh-bridge-ros2dds` + the OM1 Go binary pointed at the
Modal model URLs, headless EGL, up-to-24 h timeout. This is the one thing that makes the OM1
adapter **run-verified** (confirming the three `UNVERIFIED` facts in
[docs/om1-integration.md](../docs/om1-integration.md)). It maps to the wiring in
[docs/record-om1-gazebo.md](../docs/record-om1-gazebo.md).

It is a starting point, not a working script — expect to iterate on:
- **The base image** — the exact ROS2/Gazebo base + how to build the OM1 Go binary and the
  Go2 sim package (`TODO(UNVERIFIED)` markers in the file).
- **Headless rendering** — Gazebo needs EGL (`--headless-rendering`, OGRE2) or xvfb; sensor
  cameras need the GPU.
- **Real-time on serverless** — a physics sim on a Modal container is untested; watch the
  function timeout and whether the loop keeps real-time.
- **The record wiring (steps 3–5)** — start the Plumbline proxy + `RecordingCoordinator` +
  the `cmd_vel` Zenoh tap, and point OM1's `cortex_llm.config.base_url` at the proxy. The
  Python is in `docs/record-om1-gazebo.md`; the skeleton sketches where it goes.
- **The `UNVERIFIED` facts** — the exact `cmd_vel` Zenoh key, the tool-call wire shape, and
  the OpenMind portal URL. A successful run is what pins them.

If real-time Gazebo on Modal proves impractical, the **software-in-the-loop** middle path
(run OM1 + Zenoh with a stubbed HAL, no physics) captures a real OM1 episode's model + Zenoh
action seams at far lower effort — see [docs/limitations.md](../docs/limitations.md).
