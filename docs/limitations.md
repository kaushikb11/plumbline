# Scope & limitations (honest audit)

This project's ethos is that *a tool built to detect overclaiming should not overclaim.*
This is the honest map — from a deep soundness audit of every pillar — of what works
today, what is true only within a scoped regime, and what is not yet built. Read it
before assuming a headline capability.

## What genuinely works today

- **Faithful record → replay of HTTP model calls** — byte-identical model I/O,
  zero-touch, wired through the `plumbline record` / `replay` CLI. If your runtime's
  model calls are HTTP (OpenAI-compatible), this delivers.
- **Comparative fidelity measurement** — ranking perception/captioner front-ends by
  downstream *decision* divergence, credited only beyond a correctly-scaled (2N
  split-half) decision-stability noise floor. Demonstrated live on Ollama in
  Experiment C. The math is sound and the rankings/deltas are meaningful.
- **A real OM1 + Gazebo physics recording, committed and gating CI** — see
  [Verified end to end](#verified-end-to-end) below.
- **Honesty discipline** — the model-I/O-only determinism envelope is stated
  correctly everywhere; no pickle; typed, dependency-free core.

## True only within a regime

These work, but narrower than a headline read suggests.

- **Reproducibility holds under one condition:** every nondeterminism source must cross
  a *captured seam*, and all other per-tick state must be a pure function of captured
  model outputs.
  - **Reproduced** (a real strength): deterministic memory — conversation history built
    from captured captions/decisions.
  - **Not reproduced:** uncaptured state — RAG retrieved inside the fuser, a
    timestamp/nonce in the prompt, or async sensor-arrival order deciding *which* frame
    is fused. This surfaces as a loud `KeyError` on re-drive, never as silent
    fabrication.
  - The `None` clock hook means wall-clock timing — and, in an async loop, *which* model
    calls happen — is uncontrolled.
- **Fidelity numbers are relative, not absolute.** `caption_loss` depends on the
  caller-supplied `render(G)`, the decider, and the action binning (both open judgment
  calls). A ranking or a regression delta *within one fixed harness* is meaningful; a
  single absolute `caption_loss` (or `decision_fidelity = 1 − loss`) is not portable
  across users/tasks. It also conflates caption *phrasing* sensitivity with
  *information* loss (no phrasing guard yet, unlike the fusion `salient_artifact`), and
  degenerates for continuous action spaces under the default label (inject a tolerance
  label from the ActionSchema).
- **The counterfactual is a single-seam linear-chain stand-in.** It compares the swapped
  seam's live-vs-recorded output with a matcher; it does NOT re-run the fuser or decider.
  Multi-seam frontiers raise `NotImplementedError`.

## Recently closed

The four gaps flagged in the original audit are now addressed.

- **WebSocket caption capture.** `proxy/ws.py` (`AsyncWSProxy` + injected
  `WsTransport`/`WsConnection`) captures OM1's WS caption/transcript RESULT stream: each
  inbound frame is a `SENSOR_TO_CAPTION` event, faithfully replayable in seq order,
  relayed zero-touch, binary via the blob path (no pickle). The ASGI websocket-scope
  server (`make_ws_asgi_app` / `make_ws_replay_asgi_app`) plus a concrete
  `WebsocketsTransport` ship in `proxy/server.py`, tested with fakes and validated
  against a real remote WS server (`examples/modal_ws_validate.py` vs
  `modal/ws_captions.py` — byte-identical replay PASS; the real run caught a
  re-serialization bug the fakes missed). *Residual: the RTSP video upload (see
  [Still open](#still-open-or-not-yet-built)).*
- **Tick source.** `proxy/tick.py::BoundaryTickPolicy` auto-advances `logical_tick` on
  the perception-boundary seam, so an out-of-process runtime needs no header; the header
  stays as an explicit override. Wired into `AsyncHTTPProxy` and the `record` CLI.
- **Integrated record → counterfactual → gate.** `plumbline/recording.py::
  RecordingCoordinator` reconstructs `CAPTION_TO_FUSE` + `DECIDE_TO_ACT` around each
  Cortex call into a full four-seam episode; `tests/test_integrated_recording.py` runs
  `counterfactual` + `gate` on the recorder's OWN output (no hand-built fixtures).
- **Decision-anchored gate (opt-in).** `regression.DecisionGate` scores drift as
  decision-distribution divergence corrected by the noise floor σ (reusing the reviewed
  fidelity math), failing iff excess > k·σ — catching a low-surface decision flip the
  surface gate misses (flagship test) and not flagging a benign rephrasing.
  `recommended_behavior_matcher` (typed, numeric-tolerant, reorder-insensitive) is the
  recommended `behavior_matcher`. **Scope:** it runs the *supplied* decider on the
  counterfactual caption — it does NOT re-run a stateful fuser or the recorded Cortex
  (that still needs a runtime re-drive); the surface/structural path stays the default
  when no decider is supplied.
- **Fidelity on recorded seams.** `fidelity/bridge.py` adds an opt-in post-record pass
  that re-samples each recorded `FUSE_TO_DECIDE` request N times against the same
  endpoint into a sibling `*.samples` episode (original trace byte-immutable, hot path
  untouched), giving recorded decision distributions, a measured σ, and
  `recorded_decision_drift` (excess over σ). Run on the real OM1 episode: σ = 0.000 at
  temperature 0.7, bad-rule divergence 1.000 — fully attributable
  ([results](results-experiment-b-om1.md)). *The sampling design and
  `default_decision_label` binning are judgment calls flagged for human review — see
  [Still open](#still-open-or-not-yet-built).*

### Verified end to end

The OM1 adapter is run-verified against a real runtime, not mocks — first via a
software-in-the-loop episode, then via the full Gazebo closed loop.

- **SIL** (`examples/record_om1_sil.py`): 1,542 events, byte-identical replay, the three
  previously-`UNVERIFIED` adapter facts pinned.
- **Tier-3 Gazebo closed loop on Modal** (`modal/gazebo_om1.py`), committed as the CI
  golden `om1-gazebo-maze-003` at `bench/golden/`: real physics (go2_sim + champ), real
  bridged odometry, 153 live-LLM Cortex decisions, 3,789 real `cmd_vel` Twist frames
  captured, **the simulated Go2 walked 8.374 m** through `maze_world`, faithful replay
  **byte-identical over all 4,095 events** — and reproduced byte-identically on a
  different machine/arch. Every pull request gates on it (`bench/om1_gazebo_gate.py`,
  [results](results-om1-gazebo.md)). Bridged key naming is pinned (bare topic names);
  two sim gaps are shimmed zero-touch and documented ([om1-integration.md](om1-integration.md)).

This is the OM1 + Gazebo *physics-simulation* integration, done and committed. A
**physical-robot** run (real Go2 hardware, not the simulator) is the one thing not yet
exercised — see below.

## Still open (or not yet built)

- **A physical-robot run.** The sim closed loop is proven; only real-robot actuation on
  physical Go2 hardware is untried. Also still open in sim: ground-truth extraction for
  caption fidelity (Experiment A).
- **RTSP video upload + the ASGI websocket server wrapper.** The WS *caption* path is
  covered (above); the `VLMGeminiRTSP` media *ingest* is a separate media transport, not
  a text-result stream, and is not yet captured.
- **Physical-action capture is lossy.** The Zenoh tap stores the binary CDR `Twist` via
  `utf-8`-`replace`, not a content-addressed blob; the `DECIDE_TO_ACT` comparison rests
  on the reconstructed tool call, not the bus bytes.
- **No OM1 runtime version guard.** The adapter is pinned to the `v1.0.0-beta.1` wire
  format (config-redirect shape, Zenoh key names, tool-call JSON — see
  [om1-integration.md](om1-integration.md)). A different OM1 build whose wire shape has
  drifted may be mis-recorded (wrong seam reconstruction) rather than rejected. Pin your
  OM1 commit and re-verify the adapter facts against a fresh recording when you upgrade.
- **Fidelity math judgment calls.** `render(G)`, the `salient`/`weights` fusion
  operation, and the decision-label binning are open decisions flagged for human review
  before fidelity numbers are published (see the `HUMAN REVIEW` banners in
  `fidelity/metrics.py` and [math-review-section7.md](math-review-section7.md)).

## Precise meanings (not limitations)

Two phrases are easy to over-read; here is exactly what they claim.

- **SSE streaming is passed through incrementally** (`proxy/server.py`). The record
  proxy forwards each SSE chunk to the runtime the moment it arrives — so
  **time-to-first-token is preserved** for a robot consuming a streamed decision — and
  records the assembled stream byte-identically *after* the client is fully served (so
  recording is off the hot path, strengthening zero-touch). This requires a
  streaming-capable transport (`AsyncStreamingTransport`, which `HttpxTransport`
  implements); a transport without a `stream()` capability falls back to buffering.
  Recording and replay are unaffected.
- **"Byte-identical" replay means canonical-payload identity, not raw provider wire
  bytes.** What `tests/test_determinism.py` asserts is that the normalized JSON `Payload`
  a runtime parses and acts on round-trips bit-for-bit; a provider's incidental key
  ordering or whitespace on the HTTP wire is not reproduced and does not affect any
  decision. Binary content is byte-exact via content-addressed blobs. Full detail:
  [determinism-envelope.md](determinism-envelope.md).

## Testing without a robot

A cloud GPU account (e.g. Modal) validates most of Plumbline against **real,
nondeterministic models** — no robot, no sim ([modal/README.md](../modal/README.md)).

- **Tier 1 (run, passes):** serves an OpenAI-compatible LLM + VLM (vLLM on A10G);
  `examples/modal_validate.py` proves faithful replay byte-identical against real
  temperature-0.7 models, and Experiment C replicated on the same endpoints with exact
  wide/narrow separation ([results](results-experiment-c.md)).
- **Tier 2 (run, passes):** `examples/modal_ws_validate.py` proves byte-identical WS
  caption replay against a real remote WS server.
- **Tier 3 (run):** OM1 + Gazebo headless produced the committed golden
  `om1-gazebo-maze-003` (4,095 events, byte-identical replay —
  [results](results-om1-gazebo.md)).

## Net

Faithful replay of HTTP model calls and comparative fidelity ranking are real and usable
today. The integrated **record → counterfactual → gate** journey now flows on the
recorder's own output (auto-ticked, four-seam episodes); the gate can score **decision
divergence** anchored to the noise floor (catching low-surface flips the surface path
missed); and the WS caption stream is capturable. The **real OM1 + Gazebo physics
recording is done and committed** (`om1-gazebo-maze-003`, gating every PR), upgrading the
OM1 adapter from source-verified to run-verified.

What remains is genuinely external or thin-glue: a run on **physical Go2 hardware** (the
sim closed loop is proven; only real-robot actuation is untried), the ASGI websocket
server wrapper + RTSP upload, faithful-CDR bus capture, and wiring fidelity onto replayed
seams. The system is demonstrable end-to-end both in-process and in Gazebo physics; the
last mile is a physical robot.
