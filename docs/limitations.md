# Scope & limitations (honest audit)

This project's ethos is that *a tool built to detect overclaiming should not overclaim*.
This is the honest map — from a deep soundness audit of every pillar — of what works
today, what is true only within a scoped regime, and what is not yet built. Read it
before assuming a headline capability.

## What genuinely works today (SOUND)

- **Faithful record → replay of HTTP model calls** — byte-identical model I/O,
  zero-touch, wired through the `plumbline record` / `replay` CLI. If your runtime's
  model calls are HTTP (OpenAI-compatible), this delivers.
- **Comparative fidelity measurement** — ranking perception/captioner front-ends by
  downstream *decision* divergence, credited only beyond a correctly-scaled (2N
  split-half) decision-stability noise floor. Demonstrated live on Ollama in
  Experiment C. The math is sound and the rankings/deltas are meaningful.
- **Honesty discipline** — the model-I/O-only determinism envelope is stated
  correctly everywhere; no pickle; typed, dependency-free core.

## True only within a regime (scoped, narrower than a headline read)

- **Reproducibility holds iff** every nondeterminism source crosses a *captured seam*
  AND all other per-tick state is a pure function of captured model outputs.
  Deterministic memory (conversation history built from captured captions/decisions)
  IS reproduced — a real strength. Uncaptured state (RAG retrieved inside the fuser,
  a timestamp/nonce in the prompt, async sensor-arrival order deciding *which* frame
  is fused) is NOT — it surfaces as a loud `KeyError` on re-drive, never as silent
  fabrication. The `None` clock hook means wall-clock timing, and in an async loop the
  *selection of which model calls happen*, is uncontrolled.
- **Fidelity numbers are relative, not absolute.** `caption_loss` depends on the
  caller-supplied `render(G)`, the decider, and the action binning (§14.5/§14.6, both
  open decisions). A ranking or a regression delta *within one fixed harness* is
  meaningful; a single absolute `caption_loss` (or `decision_fidelity = 1 − loss`) is
  not portable across users/tasks. It also conflates caption *phrasing* sensitivity
  with *information* loss (no phrasing guard yet, unlike the fusion `salient_artifact`),
  and degenerates for continuous action spaces under the default label (inject a
  tolerance label from the ActionSchema).
- **The counterfactual is a single-seam linear-chain stand-in.** It compares the
  swapped seam's live-vs-recorded output with a matcher; it does NOT re-run the fuser
  or decider. Multi-seam frontiers raise `NotImplementedError`.

## Gaps closed (the four red items are now addressed)

1. **WebSocket caption capture — CLOSED (residual: RTSP video upload only).**
   `proxy/ws.py` (`AsyncWSProxy` + injected `WsTransport`/`WsConnection`) captures OM1's
   WS caption/transcript RESULT stream: each inbound frame is a `SENSOR_TO_CAPTION`
   event, faithfully replayable in seq order, relayed zero-touch, binary via the blob
   path (no pickle). The ASGI websocket-scope server (`make_ws_asgi_app` /
   `make_ws_replay_asgi_app`) + a concrete `WebsocketsTransport` ship in `proxy/server.py`,
   tested with fakes AND validated against a real remote WS server
   (`examples/modal_ws_validate.py` vs `modal/ws_captions.py` — byte-identical replay
   PASS; the real run caught a re-serialization bug the fakes missed).
   **Residual (still open):** the RTSP video *upload* (`VLMGeminiRTSP` media ingest — a
   separate media transport, not a text-result stream).
2. **Tick source — CLOSED.** `proxy/tick.py::BoundaryTickPolicy` auto-advances
   `logical_tick` on the perception-boundary seam, so an out-of-process runtime needs no
   header; the header stays as an explicit override. Wired into `AsyncHTTPProxy` and the
   `record` CLI.
3. **Integrated record → counterfactual → gate — CLOSED.** `plumbline/recording.py::
   RecordingCoordinator` reconstructs `CAPTION_TO_FUSE` + `DECIDE_TO_ACT` around each
   Cortex call into a full four-seam episode; `tests/test_integrated_recording.py` runs
   `counterfactual` + `gate` on the recorder's OWN output (no hand-built fixtures).
4. **Decision-anchored gate — CLOSED (opt-in).** `regression.DecisionGate` scores drift
   as decision-distribution divergence corrected by the noise floor σ (reusing the
   reviewed fidelity math), failing iff excess > k·σ — CATCHING a low-surface decision
   flip the surface gate misses (flagship test) and NOT flagging a benign rephrasing.
   `recommended_behavior_matcher` (typed, numeric-tolerant, reorder-insensitive) is the
   recommended `behavior_matcher`. **Scope:** it runs the *supplied* decider on the
   counterfactual caption — it does NOT re-run a stateful fuser / the recorded Cortex
   (that still needs a runtime re-drive); the surface/structural path stays the default
   when no decider is supplied.

## Still open

- **Gap 1 residual:** the RTSP video upload and the ASGI websocket server wrapper (above).
- **Fidelity not wired to *recorded* seams.** `caption_loss`/`decision_drift` take a
  live decider; the decision gate runs a supplied decider on the counterfactual caption
  rather than replaying recorded `FUSE_TO_DECIDE` seams. A bridge from replayed decision
  seams into the metrics is not built.
- **Physical-action capture is lossy.** The Zenoh tap stores the binary CDR `Twist` via
  `utf-8`-`replace`, not a content-addressed blob; the `DECIDE_TO_ACT` comparison rests
  on the reconstructed tool call, not the bus bytes.
- **The OM1 adapter is now run-verified via a SIL episode** — the real OM1 Go
  binary + real cloud LLM + real Zenoh, no sim (`examples/record_om1_sil.py`):
  faithful replay byte-identical over 1,542 events, action sequence recovered,
  and the three previously-`UNVERIFIED` interface facts pinned (see
  [om1-integration.md](om1-integration.md)). Still open for full WS5: a
  **Gazebo** episode (needs Ubuntu+ROS2+Gazebo) for sim-grounded scenes and the
  ros2dds-bridged key naming.

## Testing without a robot

A cloud GPU account (e.g. Modal) validates most of Plumbline against **real,
nondeterministic models** — no robot, no sim. See [modal/README.md](../modal/README.md).
Tiers 1 and 2 have been **run and pass**: Tier 1 serves an OpenAI-compatible LLM + VLM
(vLLM on A10G) and `examples/modal_validate.py` proves faithful replay byte-identical
against real temperature-0.7 models — and Experiment C replicated on the same endpoints
with exact wide/narrow separation ([results](results-experiment-c.md)). Tier 2
(`examples/modal_ws_validate.py`) proves byte-identical WS caption replay against a real
remote WS server. Tier 3 (stretch, not yet run) runs OM1 + Gazebo headless for the
run-verified episode.

## Net

Faithful replay of HTTP model calls and comparative fidelity ranking are real and
usable today. The four red gaps are now addressed: the integrated **record →
counterfactual → gate** journey flows on the recorder's own output (auto-ticked,
four-seam episodes), the gate can score **decision divergence** anchored to the noise
floor (catching the low-surface flips the surface path missed), and the WS caption
stream is capturable. What remains is genuinely external or thin-glue: a **real OM1 +
Gazebo recording** (needs Ubuntu+ROS2+Gazebo — the one thing that upgrades the OM1
adapter from source-verified to run-verified), the ASGI websocket server wrapper +
RTSP upload, faithful-CDR bus capture, and wiring fidelity onto replayed seams. The
system is now demonstrable end-to-end in-process; the last mile is a real robot run.
