# Production-readiness review — 2026-07

Four adversarial reviewers, each on a disjoint dimension, each instructed to *break*
the guarantees (not confirm them) and to ground findings in `file:line` + real runs
(wheel builds, repro scripts, PoCs). This is a production-readiness review — "would a
real OM1 team survive depending on this in a live robot + CI" — not a DX review.

> **UPDATE (2026-07): all four blockers + both security majors below are now FIXED**
> (266 tests, `mypy --strict` + `ruff` clean). Blocker 1/2 — the HTTP replay path now
> delegates to the tested `ReplayingProxy` (per-digest cursor, `ReplayMiss`, indexed
> once) — no more false-green, no more O(n²). Blocker 3 — `_ensure_open` moved inside
> the zero-touch guard; a store fault on the first call is logged-and-dropped and the
> upstream response is still returned. Blocker 4 — torn *trailing* lines recover the
> good events (interior corruption still raises), re-opening a non-empty episode raises
> `EpisodeExists` instead of wiping, and appends fsync / manifests write atomically.
> Security 5 — a `redactor` hook on the recording proxy (`redactor_for({...})`) scrubs
> named JSON fields before write, plus a "traces are sensitive" warning in README/FAQ.
> Security 6 — `TraceStore` validates path components (`UnsafeTraceRef`) so a hostile
> shared trace can't traverse outside the store. **The deployed verdict moves to
> Ready-with-caveats**; the remaining *majors* (streaming reads, semver policy,
> `limitations.md` reconciliation, OTel status, SSE buffering, global embedder) are the
> follow-up set. See the "Majors" section for what's left.

## Verdict: NOT YET production-ready for the deployed/live use case — but the gap is narrow, well-defined, and fixable
## (original assessment below; blockers + security majors since resolved — see banner)

The **substrate is genuinely production-grade**: the frozen `core/`, the `ReplayingProxy`,
the determinism/divergence guarantees, the fidelity gate math, the OM1 integration
(really proven on a 4,095-event Gazebo golden, not mocked), and the packaging (a real
wheel installs clean, `py.typed` ships, the pytest plugin and adapter entry points load).
No pickle, no RCE, no credential leak, no overclaim of wall-clock determinism.

But **the two surfaces a real OM1 team actually deploys** — the `plumbline replay`
HTTP server and the live recording hot path — carry reproduced blockers, and **CI is
green (249 passed) because those deployed paths lack tests.** The most damning single
fact: the tool exists to prevent false-greens, and a reviewer reproduced a **false-green
in the deployed CI gate**.

### Per-dimension verdicts

| Dimension | Verdict | One-line |
|---|---|---|
| Correctness & guarantees | **Not ready** | The deployed HTTP replay path is a *second, divergent* implementation of the tested one — no cursor (false-green) and O(n²). |
| Operational robustness | **Not ready** | Zero-touch has a hole on episode-open (breaks the live robot); crash recovery loses the mission trace. |
| Security & supply chain | Ready-with-caveats | No RCE/pickle/key-leak, but traces store prompts verbatim with no redaction/warning, and loading a shared trace has a path traversal. |
| Release integrity & OM1 fit | Ready-with-caveats | Wheel installs clean and OM1 fit is genuinely proven; missing a stability policy, a self-contradicting honesty doc, real gate not packaged. |

## The through-line

**Solid core, unfinished edges — and the edges are the deployed surfaces.** Every
blocker sits where the substrate's correct implementation was *not carried into the
path users actually run*, and no test guards that path:

- `ReplayingProxy` (tested) serves recurring requests by a per-digest cursor; the
  deployed `AsyncHTTPProxy.replay` (untested for this) serves response #1 every time.
- The recorder's content zero-touch is correctly guarded; the *episode-open* step
  that precedes it is not.
- The store is correct for well-formed traces; it has no defense for the *untrusted/
  crashed* traces that are its actual production input.

## Blockers (fix before any live-robot or deployed-gate use)

1. **HTTP faithful-replay has no cursor → false-green in the CI gate.**
   `proxy/http.py:186-190` returns the *first* event matching a `request_digest` on
   every call. A robot re-issuing a recurring request (idle/hover/static-scene at
   temp>0, where distinct sampled responses were recorded) gets response #1 every time;
   the replayed action trace silently diverges from the recording with **no `ReplayMiss`,
   no divergence flag**. PoC: recorded `[LEFT, RIGHT]` → replayed `[LEFT, LEFT]`, while
   `ReplayingProxy` on the same store correctly gives `[LEFT, RIGHT]`. **Fix:** make
   `make_replay_asgi_app` delegate to `ReplayingProxy` (per-digest cursor, `ReplayMiss`
   on over-consumption, `verify_fully_consumed` on under-consumption) — collapse the two
   replay implementations into the tested one. Add repeated-digest + scale tests to
   `tests/test_http_replay.py`.
2. **HTTP replay is O(n²).** Same path calls `store.load_episode()` (full reparse) per
   request — ~65 ms/request on the 4,095-event golden, minutes for a real episode. The
   `ReplayingProxy` delegation (blocker 1) also fixes this (it pre-indexes once).
3. **Zero-touch hole on episode-open (invariant 4 violation against a live robot).**
   `_ensure_open` runs *outside* the guarded `try` (`proxy/recording.py:109`,
   `proxy/http.py:92`). If the store can't be opened (read-only/full log partition — the
   normal failure of a long mission), the runtime's model call **throws** though the
   model is healthy. Verified: `ZERO-TOUCH VIOLATION on open failure: OSError`. **Fix:**
   open lazily/best-effort inside the guard; a store fault must log-and-drop, never reach
   the runtime.
4. **Crash recovery loses the mission trace.** A robot killed mid-episode leaves a torn
   trailing line that makes `load_episode` reject the *entire* episode
   (`core/store.py:154`), and re-opening the same episode id `write_text("")`s the file
   (`store.py:100`) — the recording becomes zero. No fsync/atomic write. **Fix:** tolerate
   a torn *trailing* line on read (recover the good events; only a corrupt interior line
   is fatal), stop truncating on re-open (append or refuse), `fsync` on append +
   atomic-rename for manifest/config, and add a `repair`/`resume` path.

## Majors (fix before telling users to share artifacts or depend on the API)

5. **No trace redaction + no sensitivity warning.** Bodies (OM1 system prompts,
   governance rules, prompt IP, possibly PII) are stored verbatim; committing a golden
   episode to a public repo leaks them. Add a redaction hook (`Callable[[Payload],
   Payload]` applied before `recorder.record`) + a loud "traces are sensitive" notice.
6. **Path traversal loading a shared trace.** `episode_id`/`sha256`/`config_hash` aren't
   validated before filesystem joins (`core/store.py`); "download a trace and replay it"
   is the product's core flow. Reject `/`, `..`, absolute paths; enforce `^[0-9a-f]{64}$`
   for hashes.
7. **Store slurps whole episodes into RAM** (~478 MB projected for a 100k-event mission →
   CI OOM). Stream `events.jsonl` line-by-line for load/gate/replay.
8. **No API stability / semver / deprecation policy.** The "frozen core" is sold as the
   contract but nothing states 0.x semantics or what's stable vs experimental. Publish one.
9. **`docs/limitations.md` contradicts itself** — cites `om1-gazebo-004` as done, then
   "Tier 3 not yet run," while the actually-committed golden is a *third* episode
   (`om1-gazebo-maze-003`). A self-contradicting honesty doc is worse than none; reconcile
   to the one real episode.
10. **Global embedder is last-writer-wins** (`core/matcher.py:156`) — breaks reproducible
    pinned matching under concurrent episodes and cross-contaminates parallel CI
    (pytest-xdist). Thread it through / bind per-context. (This is the deferred C2.)
11. **OTel export drops `plumbline.http_status`** — error-saturated episodes render all
    green in Tempo/Grafana; export is post-hoc only (no live mission view).
12. **SSE is fully buffered** (`proxy/server.py:77`) — destroys time-to-first-token for a
    robot consuming a streamed decision. Fix (pass through incrementally) or document loudly.
13. **Real gate not in the wheel + sdist leaks dev files.** `bench/` isn't packaged (only
    `plumbline/bench/`), so `plumbline gate bench/…` is repo-only; the sdist bundles
    `.claude/settings.local.json` and `CLAUDE.md`. Package the golden as data (or scope the
    real gate as repo-only) and add sdist excludes — required before PyPI.

## Minors

- Gate-config `exec` trust boundary is only a code comment — surface it in CLI help/docs.
- `--adapter module:Class` / entry-point loading is standard plugin trust — one doc line.
- Default `EmbeddingMatcher` is bag-of-tokens unless `set_embedder` is called; gate output
  should print which embedder actually ran.
- OM1 adapter is pinned to `v1.0.0-beta.1` wire format with no version-compat guard — record
  the OM1 commit into the manifest and warn if assumptions can't be confirmed.
- Replay re-serializes canonical JSON (not raw provider wire bytes); invariant 2's
  "byte-identical" should specify canonical-payload identity.

## What is already production-grade (verified, keep)

- **Substrate soundness:** no pickle/eval/exec anywhere; the determinism test is real
  (nondeterministic stubs, byte-identity, plus a re-drive that proves a fresh seed
  diverges); halt-on-divergence is the default; concurrency is correct on the documented
  `RecordingSession` path (8×40 real threads, gap-free seq).
- **Fidelity gate math:** no false-green in the math — endpoint-stationarity enforced, σ
  floored at 1/n, distribution-free permutation p-value default, empty golden set *fails*.
- **Packaging:** a real wheel installs clean in a fresh venv (core + extras); `py.typed`
  ships; console script, pytest11 plugin, and adapter entry points all load; extras resolve.
- **OM1 fit is genuinely proven:** the committed `om1-gazebo-maze-003` golden is a real
  4,095-event episode; `plumbline gate bench/om1_gazebo_gate.py` PASSES byte-identically.
- **Honesty discipline:** no wall-clock/scheduler-determinism overclaim survived the audit.

## Bottom line

For an **offline library + CI record-replay substrate**, Plumbline is close to ready —
close blockers 4 and majors 5–9 and it is dependable. For a **deployed replay server or a
live-robot recorder**, it is **not ready** until blockers 1–4 are fixed — the deployed
replay path can go green on a regression, and the live recorder can break the robot or lose
the trace. All blockers are contained fixes (chiefly: route the HTTP paths through the
already-correct `ReplayingProxy`, and harden the store's open/write/read for the crashed-
and-untrusted inputs that are its real production diet).
