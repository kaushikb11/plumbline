# Developer-experience assessment — 2026-07

A four-surface DX audit (onboarding/CLI, Python-API ergonomics, adapter authoring,
docs/examples), each **empirically exercised** — commands run, snippets executed,
a minimal adapter written from the contract — and benchmarked against the tools
developers will mentally compare Plumbline to. Evidence is cited as `file:line`.

## The through-line

**The engineering is strong; the on-ramp is not.** The frozen contract is clean and
fully typed (`py.typed`, `mypy --strict` clean, no `Any` in public signatures), the
docs are accurate (near-zero snippet rot), and the honesty discipline is meticulous
(no wall-clock-determinism overclaim anywhere — invariant 4 respected). But **every
"first N minutes" surface has friction**, and two of them are outright blockers on the
command the README leads with. A framework whose thesis is "make robot runtimes
testable" is currently hard to test-drive. None of this is deep — it's namespaces,
guard-rails, one guide, and a handful of documented-path bugs. The bones are good.

### Scorecard

| Surface | Grade | One-line |
|---|---|---|
| Honesty / no-overclaim | **A** | Determinism scoping meticulous; the tool that detects overclaiming does not overclaim. |
| Docs accuracy | **A−** | Every snippet checked against the live API; near-zero rot. One wrong "# the default" comment. |
| Docs completeness | **C+** | Great orientation + concept docs; **no API reference**, no concepts/lifecycle page, no FAQ. |
| Python-API ergonomics | **C** | `fidelity` is a model of curation; `core` + top-level namespaces are **0 bytes**; the easy path is unexported. |
| Onboarding / CLI | **C−** | Bare install is clean, but the README's headline `record`/`replay` **crash** on it; the one command that works is buried. |
| Adapter authoring | **C−** | Contract is small and clean, but mis-documented ("5 methods" → really 7), no guide, no conformance test, CLI can't run third-party adapters. |

### The competitive bar

DX is relative to what developers already know. Plumbline is a **record-replay**
framework, so the reference is [**VCR.py**](https://github.com/vcr/vcr): one decorator,
first run records a cassette, later runs replay, and **record modes**
(`once`/`new_episodes`/`none`/`all`) are a first-class concept — `none` in CI guarantees
no live calls. For the **eval/gate** role the references are
[**promptfoo**](https://www.promptfoo.dev/docs/intro/) (~15 min to first eval, single
version-controlled config, CLI-first) and [**deepeval**](https://scrolltest.com/deepeval-vs-promptfoo-2026-llm-evaluation-framework/)
("Pytest for LLMs," pytest-native CI metric gate). Takeaways that shape the recommendations:
**TTHW of 15–30 min is the bar**, **meet developers inside pytest**, **name the
record-mode concept**, and **lead with the command that works**.

---

## Resolution status (2026-07) — RESOLVED

All of P0–P3 and the P4 doc/example items were implemented (five parallel workstreams
+ an integration pass). **238 tests green, `mypy --strict` + `ruff` clean.** Highlights:

- **P0 (blockers/bugs):** record/replay now raise the friendly `pip install 'plumbline[proxy]'`
  message instead of a raw `httpx` traceback (guard reached); the quickstart/README
  counterfactual snippets run on copy-paste (stale episode id fixed); the wrong
  "# the default" comment corrected; the PyPI claim softened.
- **P1 (discoverability):** `plumbline/__init__.py` and `core/__init__.py` are now
  curated (`from plumbline import Seam, SeamEvent, make_seam_event, Recorder, Replayer,
  TraceStore, RecordingSession, …`); `RecordingSession` exported; docs lead with the
  one-command green (`plumbline gate bench/om1_gazebo_gate.py`); a zero-dep
  `examples/toy_loop.py` runs on a bare install; `plumbline list` + `--version` added;
  episode-not-found humanized; `python -m plumbline` works (`__main__.py`).
- **P2 (footguns):** `make_seam_event(...)` auto-computes the digest (kills the
  silent-unreplayable footgun); the digest is validated at record time
  (`DigestMismatch`); raw `KeyError`s replaced by typed `EpisodeNotOpen` / `EpisodeNotFound`
  / `ReplayMiss` (backward-compatible subclasses); the `gate` module renamed `gating`
  to end the name collision.
- **P3 (extensibility):** the adapter contract is unified into ONE `@runtime_checkable`
  `Adapter` Protocol (7 methods, reconstruct hooks folded in); `assert_conforms` +
  `conformance_checks` + `adapters/_template.py`; `docs/writing-an-adapter.md`;
  CONTRIBUTING's "five methods" corrected. (CLI entry-point loading for third-party
  adapters is the one P3 item deferred — see below.)
- **P4 (docs):** `docs/api.md`, `docs/concepts.md`, `docs/faq.md` added; examples fail
  with one-line actionable messages + an `examples/README.md`.

**Deliberately deferred (need your input, not blind implementation):**
- **The `pytest-plumbline` plugin + named record-mode concept (P4.20)** — a net-new
  product surface whose semantics (record-mode names, fixture shape) deserve a design
  decision. Sketch below.
- **CLI entry-point loading of third-party adapters (P3.16)** — a `--adapter-class
  module:Class` escape hatch and an `importlib.metadata` group; small, but it's an API
  surface worth deciding deliberately.

---

## Prioritized action plan

### P0 — Documented-path bugs (break the first 5 minutes; ~1–2 hrs total)

These falsify the docs' own "every snippet runs" promise. Highest ROI in the audit.

1. **`plumbline record`/`replay` crash with a raw `ModuleNotFoundError: httpx` on the
   base install the README tells users to do.** README leads with `pip install -e .`
   (README.md:73) then `plumbline record …` (README.md:103); `run_record` does
   `import httpx` (cli.py:164) and `run_replay` hits `proxy/server.py:23` — both
   **before** the friendly guard at cli.py:139, which raises the correct
   `"… pip install 'plumbline[proxy]'"` message but is **unreachable dead code**.
   *Fix:* guard the httpx/uvicorn imports so the friendly message actually fires.
2. **Quickstart's flagship counterfactual snippet crashes on copy-paste.**
   quickstart.md:98 (and README.md:120) call `.counterfactual("go2-gazebo-001", …)`,
   an episode no earlier step records (step 2 records `"demo"`, quickstart.md:62) →
   `FileNotFoundError`. *Fix:* change the id to `"demo"`.
3. **The `# the default` comment on `counterfactual(..., on_divergence=…)` is wrong**
   (README.md:124, quickstart.md:103) — the parameter has **no default** (omitting it
   raises `TypeError`). *Fix:* give it a real default (`DivergencePolicy.HALT`, which
   matches invariant 5) or reword the comment.
4. **`pip install plumbline` (README.md:70) implies PyPI availability** the project
   may not have yet (every working command uses `-e .`). *Fix:* verify or soften.

### P1 — Discoverability & the easy path (make the API explorable; ~half a day)

5. **Populate `plumbline/core/__init__.py` and `plumbline/__init__.py`** (both **0
   bytes**) with `__all__` + re-exports (Seam, SeamEvent, Payload, Context, Recorder,
   Replayer, TraceStore, VirtualClock, Matcher + builtins, DivergencePolicy, …). Today
   `import plumbline` exposes `[]` and the most-used frozen types force 5-module-deep
   imports — asymmetric with the well-curated `proxy`/`fidelity`/`regression`/`adapters`
   (`fidelity/__init__.py:55`). **Zero risk to the frozen contract: re-exports don't
   change signatures.**
6. **Export and document `RecordingSession`/`RecordingCoordinator`** (session.py:57,
   recording.py:71) — the intended "coordinate all four seams" easy path that is
   currently unexported and absent from the README, so users hand-roll seq/tick
   coordination.
7. **Lead the docs with the command that works.** `plumbline gate bench/om1_gazebo_gate.py`
   is a genuine one-command green result (4,095-event golden, 0% drift, **verified**) on
   a bare install — promote it (or add a `plumbline demo`) instead of steering operators
   into `plumbline record`. Add a zero-dependency `examples/toy_loop.py` mirroring the
   quickstart.
8. **Humanize episode-not-found + add discovery.** Missing episodes dump internal
   `episodes/<id>/manifest.json` paths (store.py:138); there is no `plumbline list`.
   Add one, and raise `episode '<id>' not found (recorded: …)`.

### P2 — Reduce boilerplate & footguns in the frozen types (~half a day)

9. **A `SeamEvent` capture helper.** `SeamEvent` has **11 required fields, zero
   defaults** (core/trace.py:73), including a `request_digest` the user computes via
   `canonicalize(request).digest`. A wrong/empty digest **records fine but is silently
   unreplayable** (surfaces later as raw `KeyError(<hash>)`). Add
   `make_seam_event(...)` (additive — like the existing `derived_seam_event` in
   `adapters/base.py:35`, so it does **not** touch the frozen dataclass) that
   auto-computes the digest and defaults `wall_ts`/`latency_ms`/`params`/`model_id`.
10. **Validate the digest at record time** — validate-and-raise (invariant: the recorder
    must not *alter* captured I/O, so recompute is out), turning the silent footgun into
    a loud error.
11. **Replace raw `KeyError`s with typed, message-rich exceptions**: record-before-open
    (store.py:82) → "episode 'e' not open; call open_episode() first"; `faithful()` miss
    (proxy/recording.py:156) → a `ProxyDivergence`-style error echoing the request. The
    good news: `ProxyDivergence` and `verify_fully_consumed` already show the target
    quality bar (proxy/recording.py:53,200).
12. **End the `gate` name collision** — the `gate` function shadows the `gate` submodule
    (regression/__init__.py:3,10), so the module is unreachable by that import path.

### P3 — Make "runtime-agnostic" actually self-service (the extensibility promise; ~1–2 days)

13. **Correct the contract's documented size.** CONTRIBUTING.md:52 and base.py:4 say
    "five methods," but a valid five-method adapter is **not** a working four-seam
    producer — `RecordingCoordinator` also requires the two `reconstruct_*` hooks in a
    *separate* Protocol (recording.py:25-48). This "five methods" claim is the single
    most misleading DX artifact. Fold the hooks into the `Adapter` Protocol (all three
    adapters already satisfy them) or fix + cross-link the docs.
14. **Ship `docs/writing-an-adapter.md`** — the biggest single gap. Assemble in one place
    the 4-seam taxonomy with the classify-vs-reconstruct table, the `env` vs
    `config_fields` decision, when `bus_tap` is `None`, and an annotated ~70-line minimal
    adapter from `generic.py` (which is an honest, non-OM1 template — generic.py:1-14).
15. **A conformance harness authors run.** Make `Adapter`/`ReconstructingAdapter`
    `@runtime_checkable` (they aren't — `isinstance` currently raises `TypeError`), and
    ship `plumbline.adapters.conformance.assert_conforms(adapter)` + a parametrizable
    pytest suite. Today a wrong `seam_of`/`action_schema`/tap surfaces only as a late
    counterfactual-replay divergence, not a clear "your `seam_of` never returns
    `DECIDE_TO_ACT`."
16. **Open the CLI to third-party adapters.** `cli.py:143` hardcodes
    `{"om1","g1","generic"}` and rejects anything else, so a correct external adapter is
    unusable via the shipped CLI — directly undercutting the runtime-agnostic pitch. Use
    an `importlib.metadata` entry-point group (`plumbline.adapters`) or a
    `--adapter-class module:Class` escape hatch. Add an `adapters/_template.py` skeleton.

### P4 — Docs completeness & strategic DX

17. **`docs/api.md`** — a reference for the frozen `core/` Protocols/dataclasses + the
    public `fidelity`/`proxy` surfaces. The library's whole value is a *frozen contract*
    yet there is nowhere to see it without reading source. Biggest completeness gap.
18. **`docs/concepts.md`** — draw the `record → faithful-replay → counterfactual → gate`
    lifecycle as one arc (today it's only implicit in sequential quickstart steps); make
    it reading-path step 0.
19. **`examples/README.md` + friendly example errors.** All 8 examples need external
    infra (Ollama/Modal/ROS) and crash with raw tracebacks when it's absent; wrap the
    `os.environ[...]`/httpx-connect calls in one-line actionable messages ("start Ollama
    at :11434" / "set PLUMBLINE_LLM_URL"). Add a troubleshooting/FAQ.
20. **Strategic (post-1.0):** a **named record-mode** concept (VCR-style
    `once`/`none`/`all`, with `none` = "CI, no live calls") and a **`pytest-plumbline`**
    plugin that turns the gate into a pytest assertion — meeting robotics devs in the
    ecosystem they already live in (deepeval's "Pytest for LLMs" playbook).

---

## What is genuinely good (keep)

- **Honesty.** No overclaim survived the audit; `determinism-envelope.md` is a strong
  standalone concept doc; `limitations.md` bridges spec→user.
- **Docs accuracy.** Every README/quickstart snippet was checked against the live API;
  the earlier `cfg.env→cfg.config_fields` fix left no analogous rot.
- **The `fidelity` public API** — one curated import, task-oriented docstrings — is the
  model the rest of the framework should match (caption_loss friction: 3/10).
- **`ProxyDivergence` / `verify_fully_consumed`** are exemplary error messages; the fix
  in P2 is to make the rest of the surface as good.
- **`generic.py`** is an honest, deliberately-not-OM1 adapter template; `derived_seam_event`
  already kills the worst SeamEvent boilerplate — precedent for fix #9.

---

*Method: four `general-purpose` subagents each exercised one surface on a clean venv
(commands run, snippets executed, a minimal adapter written from the contract);
competitive frame from VCR.py, promptfoo, deepeval, pytest. Findings are grounded in
`file:line` and real command output.*
