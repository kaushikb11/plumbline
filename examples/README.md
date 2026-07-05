# Plumbline examples

Runnable drivers that exercise Plumbline against **real** models and runtimes. They
are demonstrations, not tests — the deterministic proofs live in `tests/`. Each
example here needs some external infra (a model endpoint, or ROS/Zenoh + the OM1
binary); when a prerequisite is missing they now print **one actionable line and
exit(1)** instead of a raw traceback. See [Prerequisites](#prerequisites) per row.

## Start here — two things that work with zero setup

Before touching any of the examples below (all of which need infra), prove the
install with these, which need **no network and no extras**:

```bash
pip install -e .                          # core only (stdlib)
plumbline gate bench/om1_gazebo_gate.py   # replays a real Go2/Gazebo golden trace, exits 0 (green)
```

That gates the committed Gazebo golden episode — the same check CI runs on every
PR — and is a genuine green result end to end. Then read
[`docs/quickstart.md`](../docs/quickstart.md) for the record → replay → measure
lifecycle against the shipped Python API.

## Which example covers which canonical task

| Canonical task | Example |
| --- | --- |
| **Record → faithful replay** (byte-identical model I/O) | `modal_validate.py` (HTTP), `modal_ws_validate.py` (WebSocket), `record_om1_sil.py` (real OM1) |
| **Counterfactual replay + halt-on-divergence + the CI gate** | `experiment_b_om1.py` |
| **Fidelity metric** (caption/fusion loss, decision fidelity) | `experiment_a.py` (verbosity curve), `experiment_c.py` (captioner leaderboard) |
| **"Baselines green, Plumbline red"** (behavior monitor vs latency/tracer) | `experiment_b.py`, `experiment_b_om1.py` |

## The examples

Install the extra each group needs (`pip install -e '.[proxy]'`, add `,zenoh` for
the OM1 recorder), then run from the repo root.

### Needs a local model endpoint — Ollama

Default endpoint `http://localhost:11434/v1`; override with the env vars below.
Setup once: `ollama serve` then `ollama pull moondream && ollama pull llama3.2:1b`.

| Example | Purpose | Prerequisites | Run |
| --- | --- | --- | --- |
| `experiment_a.py` | Caption verbosity/fidelity curve — shows a surface text metric is blind to which words carry the decision. | Ollama running. Optional: `PLUMBLINE_OLLAMA_URL`, `PLUMBLINE_VLM`, `PLUMBLINE_DECIDER`. | `python examples/experiment_a.py` |
| `experiment_b.py` | Same corridor, wide vs narrow FOV: a latency dashboard + OTel tracer stay green while Plumbline's behavior monitor goes red on the action inversion. | Ollama running. Same optional overrides. | `python examples/experiment_b.py` |
| `experiment_c.py` | Ranks two perception front-ends of one VLM by downstream decision fidelity (the leaderboard behind `docs/results-experiment-c.md`). | Ollama running. Optional: `PLUMBLINE_VLM_URL` / `PLUMBLINE_LLM_URL` (point captioner/decider at separate endpoints, e.g. Modal), `PLUMBLINE_CAPTION_PROMPT`. | `python examples/experiment_c.py` |

### Needs a remote model endpoint — Modal (or any OpenAI-compatible / `wss://`)

Deploy the servers under `modal/` first (see `modal/README.md`).

| Example | Purpose | Prerequisites | Run |
| --- | --- | --- | --- |
| `modal_validate.py` | Drives scenes through the recording proxy to a real Modal VLM+LLM at temperature > 0, then faithful-replays and asserts **byte-identical** model I/O. | `PLUMBLINE_VLM_URL`, `PLUMBLINE_LLM_URL` (both **required**). | `PLUMBLINE_VLM_URL=… PLUMBLINE_LLM_URL=… python examples/modal_validate.py` |
| `modal_ws_validate.py` | Records a live `wss://` caption stream through the WS proxy while relaying it unaltered, then replays with no upstream and asserts the frame sequence is identical. | `PLUMBLINE_WS_URL` (**required**). | `PLUMBLINE_WS_URL=wss://… python examples/modal_ws_validate.py` |

### Needs a recorded OM1 trace + a live LLM

| Example | Purpose | Prerequisites | Run |
| --- | --- | --- | --- |
| `experiment_b_om1.py` | Gate over a **real recorded OM1** episode: PASS on the unchanged config, FAIL (regression caught, seam-attributed) when a governance rule is injected and re-run against the live Cortex model — while both text baselines stay green. | `PLUMBLINE_LLM_URL` (**required**); a trace store produced by `record_om1_sil.py`. Optional: `PLUMBLINE_STORE` (default `./traces-sil`), `PLUMBLINE_EPISODE` (default `om1-sil-002`). | `PLUMBLINE_LLM_URL=… python examples/experiment_b_om1.py` |

### Needs ROS/Zenoh + the OM1 Go binary

| Example | Purpose | Prerequisites | Run |
| --- | --- | --- | --- |
| `record_om1_sil.py` | Records a real OM1 episode software-in-the-loop: hosts the recording proxy, a Zenoh `cmd_vel` tap, and a stubbed HAL, sharing one episode. Produces the trace `experiment_b_om1.py` consumes. | `pip install -e '.[proxy,zenoh]'`; the OM1 Go binary + a Zenoh session. `PLUMBLINE_UPSTREAM` (**required**). Optional: `PLUMBLINE_STORE`, `PLUMBLINE_EPISODE`, `PLUMBLINE_DURATION`, `PLUMBLINE_PORT`. | `PLUMBLINE_UPSTREAM=… python examples/record_om1_sil.py &` then `cd <OM1> && make run CONFIG=plumbline_sil` |

## Companion files (not runnable examples)

- `om1_sil_config.json5` — the headless OM1 config `record_om1_sil.py` expects the
  OM1 binary to run with (`agent_inputs: []`, Cortex pointed at the proxy).
- `_env.py` — the shared prereq-UX helper (`require_env`, `friendly_endpoint`) that
  turns a missing env var or an unreachable endpoint into the one-line message above.
