# Math review: fidelity layer (§7) — human sign-off document

**Purpose.** CLAUDE.md puts `fidelity/` on a short leash: "the human reviews the math, not just the test." This document is that review packet. It maps every §7 definition to its implementation, registers every judgment call the spec left open, and lists the edge cases. Each register entry ends in a YES/NO question; Section 4 collects them. Target: sign-off in ~1 hour.

**Source of truth:** `spec/plumbline-engineering-spec.md` §7 (lines 302–359), §8.3 (line 383), §14.5 (line 511), §14.6 (line 514).

**Code under review:**
- `plumbline/fidelity/decision.py` (D(x), divergences, σ)
- `plumbline/fidelity/metrics.py` (caption_loss, fusion_loss, decision_drift, salient_artifact)
- `plumbline/fidelity/judge.py` (structural/semantic equivalence, judge noise floor)
- `plumbline/fidelity/bridge.py` (recorded-seam sampling, recorded_decision_drift)
- `plumbline/regression/gate.py` (DecisionGate)
- Tests: `tests/test_fidelity.py`, `tests/test_noise_floor.py`, `tests/test_judge.py`, `tests/test_bridge.py`, `tests/test_gate.py`

---

## Findings (read first)

Two real inconsistencies found by this review — **both fixed in code the same day**
(see the Resolution notes below). The reviewer's job on F1/F2 is to confirm the
fixes, not to choose among the original options.

### F1. `bridge.recorded_decision_drift` violates decision.py's own √2 argument (σ inflated ≈ √2, gate becomes more lenient)

`decision.py:144-164` (`decision_stability`) deliberately draws **2N** samples and splits into two **N**-halves, so σ is measured at the same sample size as the numerator's two full-N distributions. The docstring states the rationale explicitly: "estimating sigma from N/2 halves of a single N sample would make the floor ~sqrt(2) too large and under-report small real losses."

`bridge.py:162-184` (`recorded_decision_drift`) does exactly the thing that docstring warns against: it computes σ as the split-half self-divergence of the **same** golden label sample used for the numerator (`bridge.py:178-179` — `golden_labels` of size N+1 with `include_original`, split into halves of size ~N/2). The numerator `div` at `bridge.py:180-183` compares a full-(N+1) golden histogram against the candidate. So the floor is measured at roughly **half** the numerator's golden sample size and is ~√2 too large by the module's own argument.

Direction of the error: an inflated σ shrinks `excess = max(0, div − σ)`, so the recorded-seam drift path **under-reports real drift and passes candidates it should flag**. This flatters the candidate, the worst direction for a regression gate. (It is *not* the direction the decision.py docstring frames — under-reporting losses here means missing regressions.)

The structural cause: the bridge has only one recorded sample set per tick; drawing "2N" would mean `sample_recorded_decisions(..., n=2N)` and using half for the distribution — nothing in the code or docs says this, and `test_bridge.py:108-122` only checks a full flip (div=1.0), which clears even an inflated floor. A small real drift (say div ≈ 1.5σ_true) would be swallowed.

**Fix options:** (a) document that callers must sample 2N and have `recorded_decision_drift` use disjoint halves for distribution vs floor; (b) apply a √2 correction to the split-half σ; (c) accept and document the leniency. → **Q1.**

**RESOLUTION (applied, then corrected — F1-redux):** the fix treats the recorded
pool of M labels as the 2N draw and measures the numerator's golden distribution at
size M//2 to match σ (also size M//2). **The first fix used a SINGLE seeded M//2
half for the numerator — which a later review round showed was seed-noise: `excess`
flipped sign between seed=0 and seed=1, and the original pinning test passed only by
a seed=0 coincidence (and used a candidate whose divergence was genuinely below the
floor, so `excess` should have been 0).** Corrected: the numerator is now
`E[div(M//2 golden half, candidate)]` **averaged over `trials` half-draws**, exactly
mirroring how σ averages its split-half self-divergence — so both sides are size
M//2 and both averaged, and the estimate is stable. Pinned by
`tests/test_bridge.py::test_drift_is_seed_stable_and_size_matched`: a real flip
candidate scores excess > 0.6 with |e(seed 0) − e(seed 1)| < 0.05 (seed-stability),
a same-distribution candidate scores < 0.1. Callers wanting size-N semantics record
n = 2N. The candidate-side sample size remains uncorrected (matching
`decision_stability`'s convention); that residual asymmetry is Q1's remaining
question. **Lesson for the reviewer: this is exactly the short-leash §7 math where
"it typechecks and the test passes" did not mean "it measures the right thing" —
twice.**

### F2. `bridge.default_decision_label` claims "lossless" but is lossy

`bridge.py:50-84` docstring: "Lossless by default like `canonical_label` (§14.6): distinct decisions never collapse." Not true as stated. The tool-call path (`bridge.py:63-80`) keeps only `(name, arguments)` and drops the tool-call `id`, `type`, `finish_reason`, and every other response field; the text path (`bridge.py:81-83`) keeps only `message.content`. Two responses differing only in tool-call `id` (which OpenAI randomizes per call) **do** collapse.

Collapsing on `id` is almost certainly the *right* behavior — a random per-call id is not a decision, and treating it as one would make every sample its own class and push σ toward its maximum. But then it is deliberate lossy canonicalization and the docstring's "lossless / distinct decisions never collapse" claim is false and should say what is actually discarded. `test_bridge.py:125-133` pins the shapes but not the losslessness claim. → **Q2.**

**RESOLUTION (applied):** the docstring now states the truth — deliberately lossy on
provider noise (ids, finish_reason, envelope fields) with the rationale, lossless
within the decision content (`(name, arguments)` pairs never collapse). Behavior
unchanged. Q2's remaining question is whether that binning boundary is the right one.

### Near-findings (not bugs, but sharp edges — details in Section 3)

- **σ=0 → n_sigma=∞ in the DecisionGate** (`gate.py:266-270`): any positive excess over a zero *sample* floor fails the gate, even excess ≈ 1/N from a single stray sample. Correct for a truly deterministic decider; brittle for a low-temperature one whose 2N samples happened to agree. → **Q12.**
- **`gate(..., drift_threshold=...)` is silently ignored in decision mode** (`gate.py:181-182` returns before `drift_threshold` is used; the threshold becomes `decision.k`). The caller in `test_gate.py:291` passes `0.1` and gets a 3.0-σ gate. Surprising API, documented only via `threshold_units`. → **Q13.**

---

## Research-grounded upgrades (implemented — the sign-off is now partly pre-answered)

A focused literature pass (LLM-eval thresholding, split-half reliability, permutation
testing, LLM-API drift) resolved five of the register questions in code. The reviewer
now confirms these rather than deciding them from scratch.

- **Q15/Q16 — the k·σ threshold is the wrong tool; added a permutation p-value mode.**
  The one paper on this exact question ([*How to Choose a Threshold for an LLM
  Evaluation Metric*](https://arxiv.org/html/2412.12148v1)) **explicitly rejects fixed
  z-score/σ multiples** because they assume normality — which fails for a metric that
  clusters near 0/1 (exactly Plumbline's TV divergence on real data). Permutation tests
  are the standard assumption-free two-distribution test ([data8](https://data8.org/fa15/text/3_inference.html)).
  `decision.py` now exposes `null_divergence_samples` (the split-half null the σ
  estimator already draws) and `permutation_pvalue`; `DecisionGate(alpha=…)` gates on
  `p < alpha` — calibrated, normality-free — with `k` kept as a documented legacy
  placeholder. The "σ-units" label was also imprecise (σ is the null *mean*, not the
  null *SD*, so `excess/σ` is a relative excess, not a z-score); the p-value sidesteps
  this. **Q15/Q16 → RESOLVED: `alpha=0.05` (permutation) is now the DEFAULT gate mode;
  `k` is opt-in via `alpha=None`, labeled legacy.**
- **Q12 — σ floored at 1/N.** The σ=0 → n_sigma=∞ hair-trigger is the zero-frequency
  problem; the standard fix is additive smoothing / a minimum-detectable-effect floor
  ([additive smoothing](https://en.wikipedia.org/wiki/Additive_smoothing), [MDE](https://www.mdrc.org/work/publications/why-estimates-below-minimum-detectable-effect-can-be-statistically-significant)):
  you can't resolve noise below the estimator resolution ~1/N. The gate's σ-mode now
  divides by `max(σ, 1/n)` — a genuine flip still fails, a single stray sample no
  longer does. **Q12 → resolved (floored).**
- **Q11 — the endpoint-stationarity assumption is now enforced.** Silent LLM-API drift
  is a documented, named problem ([*Who Drifted?*](https://arxiv.org/abs/2606.15474),
  [*Test Before You Deploy*](https://arxiv.org/html/2604.27789v1)); the recommended
  mitigation is exactly what Plumbline records — the served model. `recorded_decision_
  drift` now raises if the sibling samples report a different served model than the
  on-path decision, turning the buried assumption into a checked precondition.
  **Q11 → resolved (guarded).**
- **Q9/Q1 — the 2N-draw σ is validated by the psychometric literature.** The noise
  floor is a "noise ceiling from split-half reliability"; split-half *underestimates*
  reliability (the √2 concern), corrected by [Spearman–Brown](https://en.wikipedia.org/wiki/Spearman%E2%80%93Brown_prediction_formula)
  — and the 2N-draw is the "just measure at full size" alternative to that correction.
  A principled choice, not ad-hoc. **Q9 → recommend YES.** (Q1's F1-redux stands; the
  candidate-side size remains the only residual.)
- **Q19 — resolved.** `recorded_decision_drift` now errors on a <2-label pool instead
  of silently using σ=0.
- **Q5 — resolved (understatement positive control added).** `salient_artifact` guarded
  only the *over*-statement direction; `salient_sensitivity` (metrics.py) is the mirror
  — a prompt that OMITS a decision-critical fact plus the caption carrying it must
  produce loss > 0, else the salient is too weak and fusion_loss under-reports. Run
  BOTH per (salient, decider) pair. Pinned by `test_fidelity.py`.
- **Q20 — resolved (unscored episodes flagged).** A decision-mode episode with no
  frontier-seam events now reports `EpisodeDrift.scored = False` (and logs), so a
  "nothing to gate" pass can't masquerade as "scored and clean". Pinned by `test_gate.py`.

Full memo and sources: this section + the Findings above; the code changes are pinned
by `tests/test_fidelity.py` (permutation), `tests/test_gate.py` (alpha mode + σ floor),
and `tests/test_bridge.py` (model-drift guard, no-samples error).

---

## 1. Formula-to-code map

| Spec definition | Formula (spec text) | Implementation | Pinning test(s) | Deviation from literal spec & why |
|---|---|---|---|---|
| **§7.1 decision distribution** (spec:306-314) | `D(x)` estimated by drawing N samples from the decision-maker at temperature; `decision_distribution(decider, context, n)` | `decision.py:109-113` `decision_distribution` = `histogram(sample_labels(...))`; `histogram` `decision.py:92-100`; binning via injectable `label`, default `canonical_label` `decision.py:54-60` | `test_fidelity.py:47-67` (binning pluggable, canonical default lossless); `test_bridge.py:99-105` (recorded variant) | Spec does not define how a sampled action plan becomes a distribution bin; code adds the `label: DecisionLabel` hook (the §14.6 question). Deviation is an *extension point*, not a math change. See register J4. |
| **§7.1 divergence** (spec:314) | "total variation for discrete typed action plans; Jensen-Shannon for soft distributions; task-success-rate gap when defined" | `total_variation` `decision.py:66-69` (½·L1, range [0,1]); `jensen_shannon` `decision.py:72-86` (log2, range [0,1]) | `test_fidelity.py:30-40` (TV endpoints and half-overlap; JS endpoints and interior) | Task-success-rate-gap divergence is **not implemented** anywhere. TV/JS math is textbook-correct (checked by hand: ½Σ\|p−q\|; JSD in bits with 0·log0=0 convention at `decision.py:79-86`). |
| **§7.2 noise floor σ** (spec:316-324) | `sigma(x) = E[div(D_half1(x), D_half2(x))]`, "splitting the N samples of D(x) into two halves" | `decision_stability` `decision.py:144-164`: draws **2N** samples, then `self_divergence` `decision.py:119-141` averages the split-half divergence over 32 seeded random partitions | `test_noise_floor.py:42-60` (σ → 0 as N grows, ratio ≈ √(N_max/N_min) within rel=0.5 — the §15 calibration test); `test_fidelity.py:43-44` (deterministic decider → σ=0) | **Two deliberate deviations, both documented in-code.** (1) 2N draw instead of splitting the N samples: keeps the floor at the numerator's sample size; splitting N would inflate σ by ~√2 (`decision.py:154-161`). (2) The spec's `E[...]` is estimated by averaging 32 seeded random partitions rather than one arbitrary split (`decision.py:34-36`). Both make the estimator *tighter/cleaner*, neither changes what σ means. See register J6; contrast with Finding F1 where the bridge breaks deviation (1). |
| **§7.3 caption_loss** (spec:326-334) | `caption_loss(C) = max(0, div(D(C), D(render(G))) − sigma)`, "sigma here is computed on render(G)" | `metrics.py:57-78` `caption_loss`: `d_caption`, `d_oracle` each from N samples; `sigma = decision_stability(decider, oracle_context, ...)` at `metrics.py:77` — floor at the oracle input, per spec | `test_fidelity.py:70-77` (decision-preserving caption → 0; decision-flipping caption → 1.0, the LiDAR-dog case) | Faithful to the formula. `render(G)` itself is **caller-supplied** (`oracle_context` param) — the §14.5 open decision is deliberately pushed out of the metric (`metrics.py:13-20` HUMAN REVIEW banner). See register J1. |
| §7.3 gate-shaped variant | (not in spec — packaging of §7.3) | `decision_drift` `metrics.py:91-109` returns `DecisionDrift(divergence, sigma, excess)`; `excess` is definitionally identical to caption_loss with golden as oracle | `test_fidelity.py:110-120` | Additive: exposes raw div and σ so the gate can threshold in σ-units instead of only on the clipped excess. No math change. |
| **§7.4 fusion_loss** (spec:336-342) | `fusion_loss = sum_i weight_i · max(0, div(D(F), D(F + salient(C_i))) − sigma)`, "weight_i … uniform by default" | `metrics.py:112-150` `fusion_loss`: per-caption term at `metrics.py:144-149`; augmented context is `f"{fused_prompt} {salient(caption)}"`; σ computed **once at F** (`metrics.py:142`) | `test_fidelity.py:100-107` (dropped obstacle caption → 1.0; irrelevant caption → 0) | **Two resolutions of spec silence.** (1) "uniform" resolved to *normalized* 1/k, so the result is the mean per-caption loss, bounded [0,1]; the raw unbounded sum needs explicit `weights` (`metrics.py:127-130`). (2) σ is computed at F, not per-augmented-context; spec does not say. Register J2, J3. `salient` defaults to `default_salient` `metrics.py:46-54` — register J2. |
| §7.4 salient guard | (not in spec — self-consistency check for the §7.4 apparatus) | `salient_artifact` `metrics.py:153-188`: run against a fused prompt that already contains the caption; must be ~0, else `salient` flips decisions via phrasing artifacts and fusion_loss overstates | `test_fidelity.py:80-97` (faithful re-emphasis → 0; phrasing-sensitive decider caught → >0) | Additive guard, in the metric's favor: it detects the *overstating* failure mode of J2 before fusion_loss is trusted. |
| **§7.5 structural judge** (spec:346-350) + §8.3 alignment (spec:383-391) | typed plans compared field-wise (Exact/NumericTolerance matcher); "alignment then per-step distance, so insertions/deletions … are penalized" | `structural_equivalence` `judge.py:62-85`: index-wise alignment; `distance = (mismatches + length_gap) / max(len)` | `test_judge.py:43-61` (mismatch → 0.5, shorter → 0.5, longer → >0); `test_gate.py:185-214` (ActionSchemaMatcher tolerance path) | Spec leaves the alignment open (§14.6). Code picks **index-wise**, not edit-distance: a single insertion at the front misaligns everything after it and drives distance toward 1. Conservative (can only over-report drift), flagged at `judge.py:17-22`. Register J8. |
| **§7.5 semantic judge** (spec:346-351) | LLM-as-judge through the proxy, recorded, "the judge's own noise floor is measured the same way as sigma" | `semantic_equivalence` `judge.py:103-125` (majority vote of n samples, distance = not-equivalent fraction, tie → NOT equivalent); `judge_noise_floor` `judge.py:128-144` (2N draws, `self_divergence` — same convention as `decision_stability`); verdict parser `_parse_equivalent` `judge.py:147-173` | `test_judge.py:63-88` (recorded and byte-identical on replay through the proxy); `test_judge.py:91-105` (noisy judge → floor > 0, steady judge → 0) | Faithful. Judge floor correctly uses the 2N convention (`judge.py:136-138`), unlike the bridge (F1). Note the floor cannot be reproduced under by-digest faithful replay — the REPLAY CAVEAT at `judge.py:28-32`; same caveat for all of §7 at `decision.py:19-23`. Register J7, J9. |
| **§8.2/§8.3 gate drift** (spec:373-391) | counterfactual-replay per golden episode; drift = `div_behavior(B_c, B_g)` from §7.5; fail per policy | Structural path: `gate.py:157-208` (Replayer counterfactual → `action_sequence` → `structural_equivalence`). Decision path: `DecisionGate` `gate.py:96-114` + `_decision_gate` `gate.py:233-294` (drift in σ-units, fail iff excess/σ > k) | `test_gate.py:145-182` (pass on compatible, fail on injected regression with seam attribution, all three policies); `test_gate.py:277-307` (decision gate catches low-surface flip the surface gate misses; benign rephrase not flagged) | Decision mode is an extension beyond §8.3's literal drift (it scores decision divergence at the frontier seam instead of replayed action distance, and skips the Replayer entirely — no §6.4 per-seam attribution beyond the frontier seam label, `gate.py:280-284`). Register J10–J13. |

**Sample-cost note (not a correctness issue):** `caption_loss` costs 4N decider calls per invocation (N + N + 2N for σ); `fusion_loss` costs (3+k)·N; `_decision_gate` costs 4N per matching event per episode (`gate.py:258-265` calls `decision_drift`, which internally re-draws σ's 2N each time).

---

## 2. Judgment-call register

Each entry: the choice, the alternative, how it could flatter the metric, and the sign-off question (collected in Section 4).

### J1. `render(G)` is caller-supplied (§14.5)
- **Choice:** `caption_loss` takes `oracle_context: str`; the metric never constructs render(G) (`metrics.py:57-78`, banner at `metrics.py:13-20`).
- **Alternative:** ship a default renderer (e.g. pose-table-to-text) in `fidelity/`.
- **Flattery risk:** a render(G) phrased in caption-like prose makes the caption "agree" with the oracle stylistically, shrinking div(D(C), D(render(G))) for reasons unrelated to information content → caption_loss too small. The module cannot detect this; it is an adapter-review gate. The honest form is a caption-agnostic structured dump identical across captioners under test.
- **Sign-off:** Q3.

### J2. `default_salient` restates the caption verbatim (§7.4)
- **Choice:** `default_salient(c) = f"Additionally, note this observation: {c}"` (`metrics.py:46-54`), appended to F with a space (`metrics.py:147`).
- **Alternatives:** extract-and-rephrase key facts; structured re-injection; prompt-position-controlled insertion.
- **Flattery risk (both directions):** a salient that flips decisions via repetition/position/vocabulary artifacts **overstates** fusion loss (looks bad for the Fuser, good for Plumbline's demo — the flattering direction for *us*); a salient too weak to resurface dropped content **understates** it. The `salient_artifact` guard (`metrics.py:153-188`) detects the overstatement direction only — there is no guard for the understatement direction.
- **Sign-off:** Q4, Q5.

### J3. fusion_loss weights: "uniform" resolved to normalized 1/k; σ computed once at F
- **Choice:** `weight_i = 1/k` default → result is the mean per-caption loss, bounded [0,1], comparable across episodes with different caption counts (`metrics.py:127-130,145`). Explicit `weights` gives the raw spec sum. σ is `decision_stability` at F (`metrics.py:142`) and the same floor is subtracted from every per-caption term.
- **Alternatives:** weight_i = 1 (raw sum — arguably the literal spec reading, unbounded); per-term σ computed at each `F + salient(C_i)`.
- **Flattery risk:** 1/k normalization shrinks the headline number as k grows (10 captions each with loss 0.1 reads as 0.1, not 1.0) — flattering to the Fuser if the reader expects a sum. Single-σ-at-F is fine when the decider's noise is context-independent; if noise is higher at the longer augmented contexts, the floor is too low there and per-caption terms **overstate**.
- **Sign-off:** Q6, Q7.

### J4. `canonical_label` lossless binning and its §14.6 degeneracy
- **Choice:** default decision bin = the full canonical JSON of the action plan (`decision.py:54-60`). Lossless: the core never bins lossily on its own; continuous action spaces must inject a coarser label (type + tolerance via ActionSchema).
- **Alternative:** default to a typed/tolerance binning in core.
- **Flattery risk:** the degeneracy runs *against* the metric's user, not for it — for continuous plans (move(x,y,yaw) with float jitter) every sample is its own class, so both div and σ saturate near their maxima and `excess = div − σ` becomes noise-dominated garbage (not systematically flattering, but meaningless). The test pins the degeneracy explicitly (`test_fidelity.py:47-67`). Nothing *forces* a caller to notice; a caller who runs the defaults on continuous actions gets numbers that look plausible.
- **Sign-off:** Q8.

### J5. `bridge.default_decision_label`: tool-call canonicalization as binning
- **Choice:** recorded Cortex responses bin by the `(name, parsed-arguments)` list of tool calls, else message text, else canonical inline (`bridge.py:50-84`). Arguments JSON-parsed so formatting differences don't split bins; unparseable arguments kept raw.
- **Alternative:** full-response canonical binning (would split on random tool-call ids — clearly worse); or tolerance-bucketed arguments (coarser).
- **Flattery risk:** minimal — dropping id/finish_reason merges samples that differ only in non-decision noise, which *lowers* σ and makes the gate *stricter*. But the "lossless" docstring claim is false as written (Finding F2). Numeric arguments still split exactly (J4's degeneracy applies to tool-call args too).
- **Sign-off:** Q2, Q8.

### J6. σ estimated from 2N draws with 32-trial averaged split-half (§7.2)
- **Choice:** `decision_stability` draws 2N and splits into N-halves; the expectation is estimated over 32 seeded shuffles (`decision.py:119-164`).
- **Alternative (literal spec):** split the existing N samples into two N/2 halves once.
- **Flattery risk:** this deviation is the *anti*-flattering one — the literal reading inflates σ by ~√2 and hides small real losses. Drawing 2N keeps the floor honest at the numerator's scale. The 32-trial average with a fixed seed (0) makes σ deterministic given the samples; a single arbitrary split would be high-variance. Both choices are defensible; both are documented in the module docstring (`decision.py:33-36`).
- **The inconsistency:** `bridge.recorded_decision_drift` does not follow this convention — Finding F1.
- **Sign-off:** Q1, Q9.

### J7. The REPLAY CAVEAT: by-digest replay collapses D(x) to a point mass
- **Choice:** documented prominently (`decision.py:19-23`, `judge.py:28-32`, `metrics.py:102-104`, `gate.py:105`): all §7 metrics require a live/temperature-sampling decider; under faithful replay N identical prompts hit one recorded response → σ=0 and every distribution is a point mass. Nothing *enforces* it — a caller who wires a ReplayingProxy-backed decider into `caption_loss` gets silently degenerate numbers (σ=0, div ∈ {0,1}).
- **Alternative:** detect repeated-digest point-mass collapse and raise/warn (e.g. if 2N samples of a temperature>0 decider produce exactly 1 label, warn that the decider may be replay-backed).
- **Flattery risk:** a replay-backed decider makes σ=0 and div ∈ {0,1}: hair-trigger, not flattering — but a *mixed* setup (some digests recorded, some live) could produce arbitrary garbage.
- **Sign-off:** Q10.

### J8. Bridge design: post-record sibling-episode sampling
- **Choice:** `sample_recorded_decisions` (`bridge.py:87-125`) re-executes each recorded FUSE_TO_DECIDE request N more times against the same endpoint *after* the episode, storing responses in `<episode>.samples`; original trace stays byte-immutable (`test_bridge.py:80-96`).
- **The buried assumption:** "same endpoint" ≠ "same distribution." The samples are drawn *later in wall-clock time* than the on-path call: a provider-side model update, server-side sampling change, or any session/state dependence between the recording and the sampling pass makes the sibling samples come from a *different* D(x) than the on-path decision. The design implicitly assumes the endpoint is stationary and the request is stateless (the full context is in the request payload). For OpenAI-style stateless chat completions this is mostly fine modulo silent model updates; for anything session-stateful it is wrong.
- **`include_original=True` (default, `bridge.py:128-147`):** mixes the on-path recorded decision into the sample. Pro: it is the one sample known to be from the *true* recording-time distribution, and the decision actually taken. Con: if the endpoint drifted between record and sampling, the distribution is a mixture; and it produces the N+1 sample-size wrinkle in F1. With N=16+ the single extra sample is negligible; the mixture concern is the real one.
- **N+1 vs 2N:** see Finding F1 — the drift path measures σ from split-halves of the very sample used as the numerator, which decision.py's own math says is ~√2 too large.
- **Alternatives:** sample at record time on the hot path (rejected — perturbs the runtime, spec §14.1 latency concern); an analysis-time stand-in decider (rejected — not the recorded model).
- **Flattery risk:** endpoint drift between record and sampling inflates σ (the sample mixes two distributions), which shrinks excess → **misses regressions**. Same direction as F1, compounding it.
- **Sign-off:** Q1, Q11.

### J9. Semantic judge: tie-break, vote distance, verdict parsing
- **Choice:** strict-majority equivalence with ties → NOT equivalent (`judge.py:121`); `distance` = fraction voting not-equivalent (`judge.py:118`); `_parse_equivalent` is word- and clause-aware with per-clause negation, any surviving difference-signal wins, unparseable → NOT equivalent (`judge.py:147-173`).
- **Alternatives:** ties → equivalent; log-prob-based verdicts; structured JSON output enforced on the judge.
- **Flattery risk:** every tie/parse-failure default is conservative (biases toward flagging), so this cannot flatter a candidate. Residual risk is parser over-cleverness misreading an exotic hedge ("not un-identical"), but misreads land on the conservative side by construction.
- **Sign-off:** Q14.

### J10. DecisionGate k=3.0 σ-units default
- **Choice:** `DecisionGate.k = 3.0` (`gate.py:110`), flagged in-code as §14.6 HUMAN REVIEW. Fail iff excess/σ > 3.
- **Alternatives:** k=2 (tighter, more false positives), k derived from a desired false-positive rate given the σ estimator's own sampling distribution (unmodeled).
- **Flattery risk:** k is the gate's entire leniency dial. There is no empirical calibration anywhere in the repo tying k=3 to a false-positive/false-negative rate on real episodes; 3 is a physics-convention placeholder. Note the σ-unit conversion at `gate.py:266-270` uses `excess/σ`, not `div/σ` — the floor is subtracted *and then* the result is divided by σ, so the effective raw threshold is div > (k+1)·σ. That is stricter bookkeeping than "fail beyond k·σ of raw divergence" — reviewer should confirm which was intended.
- **Sign-off:** Q12, Q15, Q16.

### J11. DecisionGate scope: decider on captions, no fuser, no replay
- **Choice:** decision mode runs the supplied probe `decider` directly on `context_of(recorded caption)` vs `context_of(override(request))` at the frontier seam (`gate.py:253-265`); it never invokes the Replayer, never re-runs the fuser or the recorded Cortex (`gate.py:103-106` docstring is honest about this). `context_of` defaults to `_default_context` = concatenated string leaves of the payload (`gate.py:47-61`).
- **Alternative:** full runtime re-drive (needs a live runtime; out of scope for pure-trace gating).
- **Flattery risk:** the probe decider is not the production Cortex; a probe blind to the dropped token reports zero divergence → gate passes a real regression. The probe choice is entirely the caller's, and nothing measures probe-vs-Cortex agreement. Also, decision-mode results carry no §6.4 per-seam attribution (the `divergence_seam` is just the frontier seam label when it fails, `gate.py:281`).
- **Sign-off:** Q13, Q17.

---

## 3. Edge cases and failure modes

| # | Case | Behavior | Assessment |
|---|---|---|---|
| E1 | **Empty distributions / n=0.** `histogram([])` → `{}` (`decision.py:94-96`); `total_variation({}, {})` = 0. So `caption_loss(..., n=0)` = 0.0 silently. | Silent zero loss on a zero-sample call. | No guard; a misconfigured n=0 reads as "perfect fidelity." Cheap to add `n >= 1` validation. → Q18 |
| E2 | **Single-label samples (σ=0).** Deterministic decider → σ=0 exactly (`test_fidelity.py:43-44`); then any div passes through undamped. In the DecisionGate, σ=0 with excess>0 → n_sigma=∞ (`gate.py:266-270`) → episode fails regardless of k. | Correct for truly deterministic deciders. Brittle for finite samples: a temperature decider whose 2N draws happened to agree gives sample-σ=0; one stray candidate sample (div ≈ 1/N) then fails the gate at ∞ σ. | Known sharp edge. Mitigation would be a σ lower bound (e.g. the resolution 1/N of an N-sample TV estimate). → Q12 |
| E3 | **`len(labels) < 2` in `self_divergence`** → `half == 0` → returns 0.0 (`decision.py:132-134`). | A 1-sample floor is 0, so excess = full div. Conservative (over-flags), but means `recorded_decision_drift` on an unsampled episode (only the on-path event, no sibling) silently uses σ=0 rather than erroring "you forgot to run sample_recorded_decisions." | → Q19 |
| E4 | **Differing sample sizes between numerator and floor.** decision.py keeps them equal by construction (2N draw). bridge.py does not: golden = N+1 labels, candidate = arbitrary `len(candidate_responses)`, σ from ~N/2 halves (`bridge.py:178-183`). | Finding F1. Additionally, if the candidate sample is much smaller than the golden one, its histogram is noisier than σ accounts for → spurious excess (over-flagging this time). | Both directions broken in the bridge; fix together with F1. → Q1 |
| E5 | **Judge tie-breaking.** Even n with a split vote: `equivalent_votes * 2 > n` false → NOT equivalent (`judge.py:121`). Unparseable judge text → NOT equivalent (`judge.py:171-173`). | Conservative in both cases. | OK. → Q14 |
| E6 | **Non-finite values.** No NaN/inf can arise from the divergences themselves (TV bounded, `_kl` skips zero-mass terms `decision.py:79-86`). `weights` are unvalidated beyond length (`metrics.py:136-137`): negative, NaN, or non-normalized weights pass straight through into fusion_loss. n_sigma=∞ is deliberately produced at `gate.py:269` and compared against k (∞ > k always fails; `max()` at `gate.py:274` handles it). | fusion_loss with NaN weight silently yields NaN, which then fails any `<= threshold` comparison — accidentally conservative, but ugly. | Cheap guard: reject non-finite weights. → Q18 |
| E7 | **Empty golden set.** `gate()` fails rather than passing vacuously (`gate.py:203-205`, same guard in `_decision_gate` at `gate.py:287`). | A mis-loaded corpus cannot green CI. | Correct. |
| E8 | **Episode with zero frontier-seam events in decision mode.** `n_sigmas` empty → `episode_n_sigma = max(..., default=0.0)` (`gate.py:274`) → drift 0, episode passes. | An episode that never hit the frontier seam silently contributes a pass. Defensible (nothing to gate) but indistinguishable from "gated and clean" in the result. | → Q20 |
| E9 | **`_parse_equivalent` on compound clauses.** Negation state accumulates within a clause and never resets after use (`judge.py:160-170`): in "not equivalent but the same", "same" is also read negated → difference. Misreads land conservative. | OK by construction (difference-signal wins). | OK. |
| E10 | **JS divergence with disjoint support** → exactly 1.0 in bits (`test_fidelity.py:38`). TV of identical dicts → 0.0 exactly. | Endpoints exact, no float drama at the boundaries. | OK. |
| E11 | **QUANTILE policy off-by-one.** Nearest-rank `ceil(q·n)−1` (`gate.py:221-224`); the previous `int(q·n)` collapsed P95 into ANY for n≳20 — already fixed and commented in-code. | Fixed. | OK. |

---

## 4. Open questions for the reviewer

Answer YES/NO (or the one-line decision indicated). Q1–Q2 correspond to the Findings; the rest to the register/edge cases.

1. **[F1] `recorded_decision_drift` σ convention:** accept that the recorded-path floor is ~√2 inflated (documented leniency), or require a fix (disjoint 2N sampling, or √2 correction)? **Decision: accept / fix-disjoint / fix-correct.**
2. **[F2] Fix the `default_decision_label` docstring** to state it deliberately discards tool-call id/type and non-decision response fields (behavior unchanged)? **YES/NO.**
3. **[J1]** Is caller-supplied `render(G)` with a documented review gate (no default renderer in core) acceptable until §14.5 is settled with WS3? **YES/NO.**
4. **[J2]** Is `default_salient`'s verbatim restatement acceptable as the default re-emphasis, given the `salient_artifact` guard must be run per (salient, decider) pair? **YES/NO.**
5. **[J2]** Is it acceptable that there is no guard for the *understatement* direction (a salient too weak to resurface dropped content)? **YES/NO** (NO ⇒ open an issue for a positive-control check: a known-dropped caption must produce loss > 0).
6. **[J3]** fusion_loss "uniform" = normalized 1/k (mean, bounded [0,1]) rather than the raw sum — correct reading of §7.4? **YES/NO.**
7. **[J3]** Single σ computed at F (not per augmented context F + salient(C_i)) — acceptable? **YES/NO.**
8. **[J4/J5]** Is lossless-by-default binning with documented degeneracy for continuous action spaces the right core default (coarsening lives in the adapter's ActionSchema), per §14.6? **YES/NO.**
9. **[J6]** 2N-draw split-half with 32-trial seeded averaging as the σ estimator (the two documented deviations from §7.2's literal text) — approved? **YES/NO.**
10. **[J7]** Is documentation-only enforcement of the REPLAY CAVEAT acceptable, or should point-mass collapse under a temperature>0 decider raise a warning? **YES/NO (NO ⇒ add the warning).**
11. **[J8]** Is post-record same-endpoint sampling (with its endpoint-stationarity assumption and `include_original=True` mixture) acceptable for the bridge, documented as a limitation? **YES/NO.**
12. **[J10/E2]** Is n_sigma=∞ on sample-σ=0 acceptable, or should σ be floored at the estimator resolution (~1/N)? **YES/NO (NO ⇒ floor it).**
13. **[J11]** In decision mode, `gate()`'s `drift_threshold` parameter is ignored (threshold is `decision.k`). Acceptable API, or should passing both raise? **YES/NO.**
14. **[J9]** Judge tie → NOT equivalent, unparseable → NOT equivalent, difference-signal-wins parsing — approved as the conservative stack? **YES/NO.**
15. **[J10]** k=3.0 as the DecisionGate default pending empirical calibration on real episodes — approved as a placeholder? **YES/NO.**
16. **[J10]** The gate thresholds `excess/σ = (div−σ)/σ > k`, i.e. effectively `div > (k+1)·σ` — is subtract-then-divide the intended definition of "k σ-units" (vs `div/σ > k`)? **YES/NO.**
17. **[J11]** Is probe-decider-on-captions (no fuser, no Cortex re-run) an honest enough decision gate given the in-code scope statement, until a runtime re-drive exists? **YES/NO.**
18. **[E1/E6]** Add cheap input validation: `n >= 1` in the distribution/σ functions, finite weights in fusion_loss? **YES/NO.**
19. **[E3]** Should `recorded_decision_drift` error (rather than use σ=0) when no sibling samples exist for the tick? **YES/NO.**
20. **[E8]** Should a decision-mode episode with zero frontier-seam events be reported distinctly (e.g. a flag) instead of drift 0.0? **YES/NO.**

---

## Sign-off

- Reviewer: ____________  Date: ____________
- Findings F1/F2 resolution recorded in: ____________ (issue/commit)
- Blanket approval of unlisted defaults is **not** implied; anything not covered by Q1–Q20 that the reviewer flags goes in below.

Notes:
