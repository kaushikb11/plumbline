# CLAUDE.md

Operating rules for building **Plumbline** with Claude Code. Read this fully before any task. The two specs in `spec/` are the source of truth; this file is the set of invariants that must hold across every session.

---

## What this project is

Plumbline is a standalone, runtime-agnostic, open-source framework that makes a language-bus robot runtime (OpenMind's OM1 is the reference) reproducible, regression-testable, and fidelity-measurable. It records and replays the nondeterministic model calls at the four seams of the perception-to-action loop, scores how much task-relevant information survives the caption/fuse bottleneck, and gates CI when a model/prompt/rule change drifts robot behavior.

Full detail:
- `spec/plumbline-project-spec.md` — the why, scope, workstreams, roadmap.
- `spec/plumbline-engineering-spec.md` — the how: interfaces, trace schema, metric math, adapter contract, repo layout, tests. **Cite section numbers from this file in commits and PRs.**

---

## Hard invariants (do not violate without an explicit human decision)

These are the places an agent will drift. Each is non-negotiable.

1. **Interfaces in `plumbline/core/` are frozen.** The Protocols and dataclasses in `core/` (Seam, SeamEvent, Interceptor, VirtualClock, Recorder, Replayer, Matcher, and the trace types) are the contract that lets workstreams parallelize. **Do not change a signature, field, or type in `core/` to make a local problem easier.** If a change seems necessary, STOP, explain why in plain language, and ask the human. Changing a frozen interface is a deliberate decision, never a convenience.

2. **The determinism property test is CI gate zero.** `tests/test_determinism.py` (record a toy two-model loop, faithful-replay, assert byte-identical model I/O) must pass before any substrate work is considered done and must stay green on every commit. If it goes red, fixing it is the only priority. Never skip, xfail, or weaken it to get a build green.

3. **No pickle. Anywhere. Ever.** Serialization is JSON for metadata and safetensors for tensors. `pickle`, `dill`, `cloudpickle`, `torch.save` of arbitrary objects, and any `eval`/`exec` on stored data are forbidden. This is not stylistic: unauthenticated pickle deserialization is the exact mechanism of CVE-2026-25874 (the LeRobot RCE). If you find yourself reaching for pickle to serialize "arbitrary Python," redesign the payload to be JSON+safetensors. See engineering spec Section 5.1.

4. **The determinism envelope is model-I/O only.** Plumbline guarantees that on replay every model call receives the recorded request and returns the recorded response, so the decision/action sequence is reproduced. It does NOT control the runtime's wall-clock scheduler unless an adapter exposes a clock hook. **No code comment, docstring, README line, or log message may claim or imply full wall-clock / scheduler determinism.** See engineering spec Sections 3.4 and 14.4.

5. **Halt-on-divergence is the default and divergence is a result, not an error.** In counterfactual replay, when a downstream seam's live request does not match the recorded one, the default is to HALT, record the divergence seam and distance, and return that as part of the result. Never silently serve a stale recorded response past a divergence. See engineering spec Section 6.

6. **Build vertical slices, not breadth-first.** Each piece validates the next: toy loop validates the substrate, OM1 recording validates the proxy, counterfactual replay validates divergence handling, the gate depends on all of it. Do not implement multiple layers at once. One slice, green and tested, before the next starts.

---

## How to work

- **Test-first wherever the spec gives a property.** Section 15 of the engineering spec defines the tests. Write the failing test, then implement to green. The substrate has hard oracles; use them.
- **Read the relevant spec section before coding.** Prompts will reference sections (e.g. "implement the Replayer per Section 6"). Open that section first.
- **Small, reviewable commits.** One logical unit per commit, message referencing the spec section.
- **Typed Python throughout.** Full type hints; `mypy --strict` must pass. Protocols for interfaces, frozen dataclasses for data.
- **Ask when the spec is silent on a judgment call.** The metric layer (Section 7) and the open decisions (Section 14) contain real judgment. If implementing one requires a choice the spec does not make, surface it rather than guessing.

## How NOT to work

- Do not refactor across the `core/` boundary to simplify an implementation.
- Do not add a dependency without noting why; keep the substrate light. The proxy is I/O-bound, so no heavy async frameworks unless justified.
- Do not implement the fidelity metrics "to pass the test" without confirming the math matches Section 7. A passing test on a wrong metric is worse than no metric.
- Do not build the OM1 adapter against assumptions; build it against a real recorded OM1 episode (available from week 3).

---

## Per-area leash settings

The right amount of autonomy differs by layer.

- **Substrate (`core/`), proxy (`proxy/`), store, gate plumbing, observability:** clear interfaces, hard tests. Long leash. Implement to green and the test is the judge.
- **Fidelity metrics (`fidelity/`):** research-flavored, judgment calls in the math. Short leash. Implement, but the human reviews the math, not just the test. "It typechecks and passes" does not mean "it measures the right thing."
- **OM1 adapter (`adapters/om1.py`):** plumbing, but against a real runtime. Medium leash. Verify against a recorded episode, not a mock.

---

## Definition of done (per workstream)

- **WS1 Substrate:** `test_determinism.py` and `test_divergence.py` green; all `core/` interfaces implemented; `mypy --strict` clean.
- **WS2 Trace + proxy:** proxy records and faithful-replays the toy loop; OTel-GenAI schema validated; zero-touch invariant test green (proxy does not alter the response the runtime receives in record mode).
- **WS3 Fidelity:** noise-floor calibration test green; caption/fusion loss implemented per Section 7 and math-reviewed by a human; golden-episode dataset loader working.
- **WS4 Gate + observability:** gate fails on an injected regression and passes on an unchanged config; GitHub Action wraps it; trace-diff view renders two episodes.
- **WS5 OM1 adapter + sim:** record + faithful-replay a real Gazebo episode with matching action sequence; counterfactual captioner swap runs end-to-end.

---

## Repository layout (target)

```
plumbline/
  core/         # FROZEN interfaces: seam, trace, clock, recorder, replayer, matcher, store
  proxy/        # recording/replaying HTTP proxy, provider normalizers
  transport/    # zenoh tap
  fidelity/     # metrics, judge, decision sampling   (short leash)
  regression/   # gate, drift, golden episodes
  adapters/     # base contract, om1
  bench/        # golden-episode dataset, episode definitions
  observability/# grafana dashboards, trace-diff backend
  cli.py
tests/          # determinism, divergence, noise-floor, matchers, proxy fidelity, e2e
spec/           # the two specs (source of truth)
CLAUDE.md       # this file
```