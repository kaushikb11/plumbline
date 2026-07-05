# API stability policy

Plumbline is **0.x / alpha** (`Development Status :: 3 - Alpha`). This page states
exactly what you may build against today and what may still move under you.

## What is stable (treated as stable-within-0.x)

Two surfaces are the contract that lets you depend on Plumbline:

1. **The frozen `plumbline.core` interfaces.** The Protocols and frozen dataclasses in
   `core/` — `Seam`, `SeamEvent`, `Context`, `Recorder`, `Replayer`, `Matcher` (and the
   `Exact` / `NumericTolerance` / `Embedding` matchers), `VirtualClock`, `TraceStore`, the
   trace types, and the typed exceptions (`EpisodeExists`, `EpisodeNotFound`,
   `DigestMismatch`, `UnsafeTraceRef`, …) — are the parallelization contract for the whole
   project and are **frozen** (CLAUDE.md invariant 1). A signature, field, or type here does
   not change to make a local problem easier; changing one is a deliberate, human-approved,
   CHANGELOG-called-out event.
2. **The flat top-level re-exports** — `from plumbline import Seam, SeamEvent,
   make_seam_event, Recorder, Replayer, TraceStore, canonicalize, RecordingSession, …`
   (the full set is `plumbline.__all__`). These are the discoverable import surface and
   track the frozen core; we treat them as stable within 0.x.

The **on-disk trace format** (OTel-GenAI-flavored `events.jsonl` + content-addressed
blobs + manifests, JSON + safetensors, never pickle) is part of this contract: a trace
recorded by one 0.x release replays under the next. A format change that would break an
existing golden is a called-out breaking change, not a silent one.

## What is experimental (may change without the stable-surface guarantee)

- **The fidelity math** (`plumbline.fidelity`) — `caption_loss` / fusion loss, the
  decision-distribution sampling, and the label/binning choices are **§14.5 / §14.6 open
  decisions** flagged for human review. Numbers are *relative within one fixed harness*,
  not portable absolutes (see [limitations.md](limitations.md)). The metric definitions
  may change as those decisions are settled.
- **The regression gate math** (`plumbline.regression`) — the `DecisionGate` noise-floor
  correction reuses that same fidelity math and carries the same experimental status. The
  gate *plumbing* (the `GateSpec` / `gate()` call shape) is stable; the *scoring* may move.
- Submodule internals outside `core/` — `proxy/`, `transport/`, `observability/`,
  `adapters/` implementation details — are public-but-evolving. Prefer the documented
  entry points (`plumbline.proxy` façade, the `Adapter` contract, the CLI) over reaching
  into internal modules.

## 0.x change policy

- **No hard SemVer guarantee before 1.0.** Pre-1.0, minor versions (`0.N`) may carry
  breaking changes. That said, the two stable surfaces above are treated as
  stable-within-0.x and are not broken casually.
- **Breaking changes are called out in a CHANGELOG**, with a **one-minor-version
  deprecation window where feasible** — a deprecation warning ships in `0.N` before the
  removal lands in `0.N+1`. Some pre-1.0 changes (a security fix, an unavoidable core
  correction) may not get a full window; those are still called out explicitly, never
  silent.
- When the core interfaces and trace format have proven out on real deployments, the
  project cuts **1.0** and adopts standard SemVer.

## In one line

Build against `plumbline.core` and the flat `from plumbline import …` re-exports and the
trace format — those we treat as stable within 0.x and break only with a CHANGELOG note
and, where feasible, a deprecation window. Treat the `fidelity` / `regression` *math* as
experimental and pin your version if you depend on its exact numbers.
