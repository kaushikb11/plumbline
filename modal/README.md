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
```

Point a WS record proxy's `upstream` at that URL — `make_ws_asgi_app` +
`WebsocketsTransport` in `plumbline/proxy/server.py` (both shipped) capture the real WS
caption stream as `SENSOR_TO_CAPTION` events, faithfully replayable via
`make_ws_replay_asgi_app`. This closes the WebSocket half of limitations gap #1.

## Tier 3 — real OM1 + Gazebo (stretch)

A custom Modal GPU image — ROS2 + `go2_gazebo_sim` + `zenoh-bridge-ros2dds` + the OM1 Go
binary pointed at the Modal model URLs, headless EGL rendering, 24 h timeout. This is the
one thing that makes the OM1 adapter **run-verified** (confirming the three `UNVERIFIED`
facts in [docs/om1-integration.md](../docs/om1-integration.md)). Heavy and not scaffolded
here — see [docs/limitations.md](../docs/limitations.md) and
[docs/record-om1-gazebo.md](../docs/record-om1-gazebo.md).
