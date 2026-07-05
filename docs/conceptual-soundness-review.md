# Conceptual-soundness review — 2026-07

Do Plumbline's core *ideas* hold up scientifically? Four adversarial reviewers, each
with web access, took one concept, grounded it in the literature, and tried to
**falsify** it (not confirm it — the project's ethos is that a tool detecting
overclaiming must not overclaim, so a rubber-stamp is worthless). This is a review of
the concepts, not the code (that's `production-readiness-review.md`) and not the
novelty (that's `related-work.md`).

## Bottom line: all four concepts are SOUND-WITH-CAVEATS — none is unsound

The underlying methodology is well-precedented and correct across the board. Every
sharp critique converged on **one theme**: the *math and mechanisms are careful and
correct, but the headline prose occasionally overclaims relative to the precise
quantity the code computes.* These are framing-honesty gaps fixable with a few
sentences — not conceptual flaws, not code bugs. Notably, in three of four the
reviewers found Plumbline had **already self-identified the sharpest attack on itself**
(in `limitations.md` / `determinism-envelope.md`); the gaps below are the residual
un-flagged sub-points, now closed.

| Concept | Verdict | Grounded in | The sharpest critique |
|---|---|---|---|
| Fidelity metric (decision-divergence, noise-floor, permutation) | sound-with-caveats | ViSIL, CheckList behavioral testing, split-half noise ceiling, arXiv 2412.12148 | Measures *agreement with the oracle-fed decider*, not task *success* |
| Language-bottleneck thesis | sound-with-caveats | Tishby IB, CaBM (2607.00578), "Reading Is Believing" (2406.15816) | "Bottleneck" oversells the fuse seam (which *injects* rules/RAG); Exp B is *injection*, not *loss* |
| Record-replay + single-seam counterfactual | sound-with-caveats | rr / deterministic simulation testing, activation/path patching (2404.15255) | Halt measures a *direct effect* (divergence onset), not the *total* downstream magnitude (2606.27510) |
| Four seams + determinism envelope | sound-with-caveats | rr / DST record-replay taxonomy, OM1 architecture | Reproducibility is conditional on "all nondeterminism crosses a captured seam" — real loops violate it, but *loudly* (digest miss → error, never silent fabrication) |

## The four concepts, in depth

### 1. Fidelity metric — sound, well-precedented; the "success" framing overclaims

Measuring caption/fuse information loss by *downstream decision divergence* (behavioral,
not surface-text) is the CheckList paradigm and has a near-exact precedent in ViSIL
(info loss validated against downstream VQA). The split-half σ is the standard
psychometric noise-ceiling estimator; the permutation-p-value gate is the statistically
correct move (σ is the null *mean*, not its SD, so the old `excess/σ` was never a
z-score — a fix the project already made).

**The gap:** the metric scores agreement with `D(render(G))` — ground truth *run through
the same decider* — not task correctness. So it conflates decision *change* with decision
*degradation*: a caption that steers a systematically-wrong decider toward the *correct*
action scores as "loss," and one that faithfully reproduces the decider's *mistake*
scores zero. The spec's own honest alternative (a task-success-rate-gap divergence,
§7.1) is not implemented, and the word "success" in the framing overclaims versus the
computed "agreement." (Secondary, minor: `max(0, div − σ)` has the standard rectified-
estimator positive bias at the floor — conservative for the gate, moot under the p-value.)

### 2. Language-bottleneck thesis — real research direction, not metaphor; the umbrella oversells the fuse seam

"Natural language as a lossy information bottleneck between perception and decision" is
an active, converging direction (Tishby's IB defines relevance *by the downstream task* —
exactly Plumbline's framing; CaBM and "Reading Is Believing" move the bottleneck into
language). Scoring surviving task-relevant information via decisions is IB-faithful.

**The gap:** "bottleneck" is precise for `SENSOR_TO_CAPTION` (genuine pixels→text
compression) but *loose* for `CAPTION_TO_FUSE`, where the Fuser **injects** rules + RAG —
information can *increase* there. And the flagship demo (Experiment B: append a governance
rule → every decision flips) is an information-*injection* effect presented under a
"how much survives the bottleneck" banner. The §7 metrics keep the mechanisms separate
(`caption_loss` = compression-loss; `fusion_loss` = dropped-caption content; Exp B framed
as drift-detection) and the seam table discloses "captions + rules + RAG" — so it's honest
in the machinery, imprecise in the prose.

### 3. Record-replay + counterfactual — established primitives; halt attributes *onset*, not *magnitude*

Digest-keyed faithful replay is orthodox record-replay (rr, DST). Single-seam-swap-freeze-
the-rest is a *named* causal-isolation primitive: the do-operator on one node / path
patching, which freezes non-targeted components precisely to isolate a direct effect.

**The gap:** by pinning downstream to the trace and **halting** at the first divergence,
the counterfactual measures a *controlled direct effect* — divergence *onset* (which seam
broke) plus a scalar distance there — **not** the *total* behavioral consequence a live,
feedback-closed re-run would show (the "Curse of Multiple Mediators," 2606.27510, proves a
single-node/fixed-context effect is confounded by sign-ambiguous interaction terms).
Halting is the *correct* reproducibility rule (continuing would fabricate a decision the
model never made) — so this is not an error — but as a *drift-attribution* signal the
distance is onset, not consequence. The full consequence is available via the live
re-drive and the `DecisionGate` (one de-truncated downstream step). `limitations.md` flags
the fixed-context/no-re-run half; it did not flag the truncation-of-magnitude half.

### 4. Four seams + determinism envelope — faithful map; reproducibility is conditional but fails *loudly*

The seam taxonomy tracks OM1's real component boundaries one-for-one. "Model-I/O
determinism, not wall-clock" is the standard record-replay stance narrowed to the boundary
that matters for decisions, honestly scoped.

**The gap:** reproducibility holds only if *every* nondeterminism source crosses a captured
seam and all other per-tick state is a pure function of captured outputs — the exact DST
invariant. Real loops violate it (async sensor-arrival order deciding *which* frame is
fused; RAG/timestamps inside the fuser; uncaptured memory), and Plumbline does **not**
control the scheduler (`clock_hook()` → `None`), so it does not meet the full DST bar.
**Why it stays honest:** the violation is architecturally forced to be *loud* — an
uncaptured input changes a request digest → no recorded response → a `ReplayMiss`/`KeyError`
on re-drive, never a silent fabrication. Combined with halt-on-divergence, Plumbline
degrades to *refusing to lie* rather than *lying quietly*. Two smaller points: the "four
seams" are really *two* captured model seams + *two* reconstructed boundaries (asymmetric),
and `determinism-envelope.md` stated reproduction as an unconditional property one line
before the conditions that qualify it (now leads with the conditional).

## What was fixed as a result (framing-honesty)

All small, precise prose changes — the concepts were already sound; these make the
headline claims match the computed quantities:

1. **"agreement", not "success"** — `limitations.md` now states the fidelity metric scores
   agreement with the oracle-fed decider (task fidelity only insofar as the decider is
   itself competent on ground truth), and the unimplemented task-success-rate-gap
   alternative is named.
2. **bottleneck vs injection** — `concepts.md` / `limitations.md` note that "bottleneck"
   is precise for the caption seam and looser for the fuse seam (a bottleneck *and* an
   injector), and that Experiment B is a drift/injection result, not a fidelity/loss one.
3. **onset, not magnitude** — the docs state the divergence distance is onset/location
   attribution, not the downstream action-sequence consequence (which needs the live
   re-drive), resolving the `concepts.md` "action sequence" tension.
4. **conditional reproduction** — `determinism-envelope.md` leads with the "every
   nondeterminism source must cross a captured seam" condition before asserting the
   reproduction property.

## Literature anchors

- Fidelity: [ViSIL 2601.09851](https://arxiv.org/abs/2601.09851), [CheckList 2005.04118](https://arxiv.org/abs/2005.04118), [threshold choice 2412.12148](https://arxiv.org/abs/2412.12148), split-half noise ceiling ([PLOS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006397)).
- Bottleneck: [CaBM 2607.00578](https://arxiv.org/abs/2607.00578), [Reading Is Believing 2406.15816](https://arxiv.org/abs/2406.15816), [Language in a Bottle 2211.11158](https://arxiv.org/abs/2211.11158), Tishby Information Bottleneck.
- Replay/counterfactual: [rr](https://rr-project.org/), [ACM Queue record-replay](https://queue.acm.org/detail.cfm?id=3391621), [Antithesis DST](https://antithesis.com/docs/resources/deterministic_simulation_testing/), [activation/path patching 2404.15255](https://arxiv.org/abs/2404.15255), [Curse of Multiple Mediators 2606.27510](https://arxiv.org/html/2606.27510).
- Seams/determinism: [OM1](https://github.com/OpenMind/OM1), [Sources of Irreproducibility in ML 2204.07610](https://arxiv.org/pdf/2204.07610).

**Verdict:** the concepts make sense. All four are scientifically defensible and
well-anchored; the only issues were headline-prose imprecisions, now tightened to match
what the code actually computes.
