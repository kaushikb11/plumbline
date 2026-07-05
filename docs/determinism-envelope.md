# The determinism envelope

This is the most important honesty boundary in the project. It is stated precisely so that no code comment, log line, docstring, or README sentence drifts into claiming more than the mechanism delivers.

## What Plumbline guarantees

On replay, every model call receives the **recorded request** and returns the **recorded response**. Because the loop's non-model transforms are deterministic, the **sequence of decisions and actions is reproduced**. This is *deterministic model-I/O replay*.

Concretely, this is what the substrate proves:

- **Faithful replay** serves every seam from the trace; `tests/test_determinism.py` asserts the recorded model I/O round-trips byte-identically through the store. "Byte-identical" is at the **canonical-payload layer** — the normalized JSON `Payload` a runtime parses and acts on, which is what determinism requires — not the raw upstream HTTP wire bytes (a provider's incidental key ordering or whitespace is not reproduced, and does not affect any decision). Binary content is byte-exact via content-addressed blobs.
- **Re-execution** (`tests/test_reexecution.py`) re-drives a runtime loop while the proxy serves each recorded response *by request digest*; the recomputed decision/action sequence matches the recording, while a fresh live run with a different seed diverges — proving the serving, not loop determinism, is doing the work.

## What Plumbline does NOT guarantee

Plumbline does **not** control the runtime's internal wall-clock scheduler — *unless* an adapter exposes a clock hook.

- The OM1 adapter's `clock_hook()` returns `None` today.
- Absent that hook, loop **timing** may vary across replays (a faster or slower served response can shift when the next tick fires in wall-clock terms), while model **I/O** does not vary.
- **Given a fixed sequence of model calls**, a diverged wall clock does not change the reproduced decision/action sequence, because that sequence is driven by served model I/O, not by wall time. The virtual clock records logical ticks so the loop's *logical* ordering is reconstructed from the trace rather than from wall time. CAVEAT: in an *async* loop, timing can change *which* model calls happen (e.g. which camera frame sits in the fuser's buffer when a tick fires) — that selection is upstream of every captured seam and is not controlled here (see [limitations.md](limitations.md)).

## The claim, stated atomically

> We claim deterministic model-I/O replay, not deterministic wall-clock scheduling.

If OM1 (or any runtime) later exposes or accepts a hook on its hertz loop, `clock_hook()` can return it and the envelope upgrades to full scheduler determinism. Until then, do not write or imply otherwise anywhere in the codebase.

## Known replay limitation (sampling distributions)

Faithful replay serves recorded responses keyed by `request_digest`, so *N identical requests collapse to one recorded response*. Reproducing a **distribution** over repeated identical calls (e.g. an N-sample `decision_distribution`, or the judge's own noise floor) under replay would require sequence-aware serving rather than by-digest serving. The noise-floor estimators are therefore record/live-mode measurements today. This is flagged at the call sites (`fidelity/judge.py`) and is a candidate proxy enhancement, not a silent gap.
