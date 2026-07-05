# Plumbline docs

Start with the [project README](../README.md) and the [quickstart](quickstart.md).
The two specs in [`../spec/`](../spec/) are the source of truth for the design.

## Map

**Start here (reading order)**
- [concepts.md](concepts.md) — **step 0**: the one-page mental model — the four seams and the record → faithful-replay → counterfactual → gate lifecycle as a single arc. Read this first.
- [quickstart.md](quickstart.md) — run it: one green command, then point base URLs at the proxy → record → replay → measure, with runnable snippets.
- [api.md](api.md) — reference for the frozen `core/` contract plus the public `fidelity` / `proxy` / `regression` / `adapters` surfaces (real signatures).
- [writing-an-adapter.md](writing-an-adapter.md) — teach Plumbline a new runtime: the 7-method `Adapter` contract, classify-vs-reconstruct seams, and an annotated minimal adapter.
- [pytest-plugin.md](pytest-plugin.md) — record/replay and the behavior gate as native pytest: the `recorded_proxy` fixture, record modes (`none`/`once`/`all`), `plumbline_gate`, and loading third-party adapters.
- [faq.md](faq.md) — which extra to install, `ModuleNotFoundError` / `ConnectionRefused` fixes, console script vs `python -m`.

**Scope & guarantees**
- [limitations.md](limitations.md) — the honest scope audit: what works, what's scoped to a regime, what isn't built. Read before assuming a headline capability.
- [determinism-envelope.md](determinism-envelope.md) — exactly what is guaranteed bit-identical (model I/O) and what is not (the wall-clock scheduler).

**Results (real-model / real-runtime)**
- [results-om1-gazebo.md](results-om1-gazebo.md) — the showcase episode: real OM1 + Gazebo physics, lidar-conditioned decisions, byte-identical replay.
- [results-experiment-b-om1.md](results-experiment-b-om1.md) — silent-regression detection on a real OM1 episode; baselines stay green, Plumbline goes red.
- [results-experiment-c.md](results-experiment-c.md) — captioner-for-decisions fidelity on real models (Ollama + Modal).
- [experiment-c.md](experiment-c.md) — Experiment C runbook.

**OM1 integration**
- [om1-integration.md](om1-integration.md) — verified interface facts for the OM1 adapter (config redirect, Zenoh keys, tool-call wire shape), pinned by the SIL and Gazebo runs.
- [record-om1-gazebo.md](record-om1-gazebo.md) — wiring the proxy + tap into a real OM1 + Gazebo run.

**Observability & review**
- [observability.md](observability.md) — Grafana panels, OTLP export, trace-diff.
- [math-review-section7.md](math-review-section7.md) — the fidelity-layer (§7) human-review packet: formula-to-code map, judgment-call register, sign-off questions.
- [related-work-audit-2026-07.md](related-work-audit-2026-07.md) — pre-release novelty re-check and the atomic-claim positioning.

## For contributors

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for the dev loop and the hard invariants. In short: `mypy --strict` and `ruff` must be clean, the determinism/divergence property tests are CI gate zero, `core/` interfaces are frozen, and nothing uses pickle.
