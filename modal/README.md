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

## Tier 3 — real OM1 + Gazebo (RUN AND PASSING)

`modal/gazebo_om1.py` runs the full closed loop headlessly on a Modal T4: Gazebo
physics (`go2_sim` + champ from [OpenMind/OM1-sim](https://github.com/OpenMind/OM1-sim))
+ `zenoh-bridge-ros2dds` + the real OM1 Go binary (pinned commit, built in-image) + a
live Cortex through the recording proxy.

```bash
modal run modal/gazebo_om1.py::doctor                          # 7 component checks
modal run modal/gazebo_om1.py::record --llm-url <llm url> --seconds 150
modal volume get plumbline-traces <episode>-store .            # pull the trace
```

Result (episode `om1-gazebo-004`): 90 live-LLM decisions, 2,407 real `cmd_vel` CDR
`Twist` frames tapped, the simulated Go2 **walked 3.455 m**, faithful replay
byte-identical over 2,587 events — locally reproduced byte-identical on a different
machine/arch. The run summary reports seam counts, observed bus keys, connector-gate
counts, and meters traveled.

Hard-won facts encoded in the app (see its docstring): `USE_SIM` must stay **unset**
(OM1's sim mode cloud-tethers `cmd_vel` with no override), the sim package is
`go2_sim` (OM1's docs name is stale), OM1-sim's own `cyclonedds.xml` + bridge config
are required in a container (loopback iface, no multicast, participant-index ceiling),
and two sim gaps are shimmed zero-touch: champ's planar odom lacks the body height
OM1's Move connector requires ("standing"), and `om_path` publishes `/om/paths/r{K}`
while OM1 subscribes bare `om/paths`.
