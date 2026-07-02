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

## Known gaps — would NOT work today, and what closing each needs

1. **OM1 reference-config perception capture (RTSP/WebSocket).** OM1's default VLM/ASR
   (`VLMGeminiRTSP`, `GoogleASRInput`) move data over RTSP + WebSocket
   (`wss://api.openmind.com`), which the HTTP-only proxy cannot see — so
   `SENSOR_TO_CAPTION`, the caption bottleneck this project exists to measure, is not
   capturable for the reference config. `FUSE_TO_DECIDE` (the Cortex OpenMind-portal
   call) IS HTTP-capturable. **Needs:** a WebSocket-aware tap + RTSP handling.
   HTTP-based perception endpoints (OpenAI-compatible vision) work today.
2. **Tick structure for an out-of-process runtime.** `logical_tick` is read from the
   `x-plumbline-tick` request header, which no provider SDK / the OM1 Go binary sends
   → every event lands at tick 0, collapsing the per-tick grouping the
   counterfactual/gate rely on. The in-process `RecordingSession.set_tick` works for a
   Python driver loop; bridging an external runtime's loop index has no shipped
   mechanism. **Needs:** a tick source (session driver, a per-request header shim, or
   heuristic tick boundaries).
3. **The integrated record → counterfactual → gate journey.** The zero-touch HTTP
   recorder captures only the model seams it sees per call; it does not reconstruct
   `CAPTION_TO_FUSE`, capture `DECIDE_TO_ACT`, or tap Zenoh. The counterfactual/gate
   are validated on hand-built in-process fixtures, not on episodes the zero-touch
   flow itself produces. **Needs:** `RecordingSession` + the Zenoh tap + `reconstruct_*`
   wired into one recording path with a tick source.
4. **The gate scores reproducibility + surface divergence, not full behavior.** Its
   divergence detection uses surface caption similarity (a matcher + halt), which can
   (a) cry wolf on benign rephrasings that halt-truncate and (b) MISS a
   low-surface-distance decision flip (drop the one token that matters → served old
   decision → green) — the dangerous direction, and exactly the flagship scenario. The
   decision-divergence machinery (the fidelity metric) is not wired into the gate, and
   `drift_threshold` is a free float, not anchored to the noise floor σ. **Needs:**
   scoring drift via decision divergence (a decider re-drive) with a σ-anchored
   threshold, and making a tolerant `ActionSchemaMatcher` (with reorder tolerance) the
   default instead of `ExactMatcher`.
5. **Fidelity is not wired to recorded seams.** `caption_loss`/`fusion_loss` take a
   live decider callable, not replayed `SeamEvent`s; "measure fidelity on the recorded
   seams" is not yet a code path. **Needs:** a bridge from replayed `FUSE_TO_DECIDE`
   seams into the metrics.
6. **Physical-action capture is lossy.** The Zenoh tap stores the binary CDR `Twist`
   via `utf-8`-`replace`, not a content-addressed blob; the `DECIDE_TO_ACT` comparison
   rests on the reconstructed tool call, not the bus bytes.
7. **The OM1 adapter is verified against OM1's source, not a running episode** (WS5
   definition-of-done unmet); three interface facts stay `UNVERIFIED` pending a real
   recording (see [om1-integration.md](om1-integration.md)).

## Net

Faithful replay of HTTP model calls and comparative fidelity ranking are real and
usable today. The differentiated integrated claim — record a real robot run and gate
its *behavior* in CI — is a validated design on synthetic fixtures, not yet a working
end-to-end system against a real out-of-process runtime. The single biggest gap:
nothing has yet flowed record → counterfactual → gate from the zero-touch recorder's
own output. Closing gaps 1–4 (WebSocket capture, a tick source, wired recording, a
decision-anchored gate) is what turns the design into a system.
