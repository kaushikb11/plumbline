# Contributing to Plumbline

Thanks for looking. Plumbline is a substrate other work builds on, so a few
invariants are non-negotiable — they are what let the layers parallelize and what
make the guarantees trustworthy. `CLAUDE.md` and `spec/plumbline-engineering-spec.md`
are the full detail; this is the short version.

## Dev loop

```bash
pip install -e ".[proxy,zenoh,embeddings,dev]"   # everything, editable
python -m pytest -q                              # the full suite (fast; no network)
mypy --strict plumbline tests                    # must be clean
ruff check plumbline tests examples modal        # must be clean
ruff format --check plumbline tests              # must be clean
```

`pre-commit install` wires the lint/format/type checks to every commit.

## Hard invariants (do not violate without a deliberate human decision)

1. **`plumbline/core/` interfaces are frozen.** The Protocols and dataclasses in
   `core/` are the contract that lets workstreams parallelize. Don't change a
   signature, field, or type in `core/` to make a local problem easier — if a
   change seems necessary, stop and raise it.
2. **The determinism property tests are CI gate zero.** `tests/test_determinism.py`
   and `tests/test_divergence.py` (and `test_golden_gazebo.py`, which runs them on a
   real robot episode) must stay green. Never skip, xfail, or weaken them.
3. **No pickle. Anywhere.** Serialization is JSON for metadata and safetensors for
   tensors; content-addressed blobs for binary. `pickle`/`dill`/`cloudpickle`/
   `torch.save` of arbitrary objects and any `eval`/`exec` on stored data are
   forbidden (this is the LeRobot CVE-2026-25874 lesson, not a style preference).
4. **Determinism is model-I/O only.** No comment, docstring, or log may claim or
   imply full wall-clock / scheduler determinism. See `docs/determinism-envelope.md`.
5. **Halt-on-divergence is the default; divergence is a result, not an error.**
   Never silently serve a stale recorded response past a divergence.
6. **The fidelity layer (`fidelity/`) is short-leash.** The §7 metric math is a
   research surface with real judgment calls (see `docs/math-review-section7.md`).
   "It typechecks and the test passes" is not "it measures the right thing" — twice
   in this project's history a passing test hid a wrong metric. Get metric changes
   reviewed on the math, not just the diff.

## Style

- Fully typed; `mypy --strict` clean, no `# type: ignore`/`cast` escape hatches.
- Protocols for interfaces, frozen dataclasses for data.
- Small, reviewable commits; reference the spec section (e.g. `§6`, `§8.3`).
- Tests for behavior and failure modes, not just the happy path.

## Adding a runtime adapter

Implement the `Adapter` protocol in `plumbline/adapters/base.py` (five methods:
`configure_proxy`, `bus_tap`, `seam_of`, `action_schema`, `clock_hook`).
`adapters/om1.py` is the worked reference — it is grounded in OM1's real source,
not assumptions, and run-verified end to end. Build a new adapter against a real
recorded episode where possible, not a mock.
