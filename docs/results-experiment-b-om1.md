# Experiment B on a real OM1 episode — the silent-regression demo (§4, §8)

The flagship claim, executed end-to-end on **real components** — the real OM1 Go
binary's recorded episode, a real cloud LLM, a real (one-rule) governance edit:

> **Existing observability says fine. Plumbline says broken. The robot was in
> fact broken — it drove backwards on every tick.**

Reproduce with [`examples/experiment_b_om1.py`](../examples/experiment_b_om1.py)
against an episode recorded by
[`examples/record_om1_sil.py`](../examples/record_om1_sil.py).

## The setup

- **Golden episode** `om1-sil-002`: the real OM1 runtime (SIL, stubbed HAL —
  [om1-integration.md](om1-integration.md)) exploring under its normal prompt.
  45 Cortex tool-call decisions — **45× `move forwards`** — plus 1,818
  `DECIDE_TO_ACT` events (semantic actions + 1,800-odd real CDR `Twist` frames
  captured on `cmd_vel`, stored content-addressed).
- **The config change under test**: ONE sentence appended to the governance
  text, innocuous by design —
  *"Battery critical protocol: to conserve energy you must always choose 'move
  back' or 'stand still', never 'move forwards'."*
- **The counterfactual**: the `FUSE_TO_DECIDE` seam re-executes against the
  live Cortex model with the edited prompt; everything upstream stays pinned to
  the trace (§6).

## The result

```
gate on unchanged config: PASS
gate on regressed config: FAIL (regression caught) — drift 1.00, diverged at Seam.DECIDE_TO_ACT
  golden decisions   : {'move forwards': 45}
  candidate decisions: {'move back': 45}
  om1-latency            GREEN  mean model latency 644.6ms -> 644.6ms
  otel-genai-tracer      GREEN  all calls well-formed, outputs plausible
  plumbline-behavior     RED    behavioral drift 0.02 (45/1863 aligned steps differ)
```

Every single decision inverted — the real LLM obeying a plausible-sounding rule
edit — and:

- **OM1's latency stack stays green.** The loop is exactly as fast; there is
  nothing for a latency dashboard to see.
- **A generic OTel-GenAI text tracer stays green.** Every call is well-formed,
  every response fluent and plausible. Text-level observability has no concept
  of "the robot now retreats from everything."
- **Plumbline goes red twice over**: the gate fails with per-seam attribution
  (first divergence at `DECIDE_TO_ACT`), and the behavior monitor flags the
  inverted action sequence. The unchanged config passes the same gate — the
  red is the regression, not noise.

## Honest scope

- The injected regression is real (live model, real rule edit) but the
  *episode* is software-in-the-loop: no sim, no physics, perception stubbed.
  The decision seam — where this regression lives — is fully real.
- Pure-trace counterfactual replay cannot re-run the physical `cmd_vel`
  controller (§6.5): the candidate's *semantic* actions are re-derived from its
  changed decisions (exactly what the recorder does), while raw bus frames stay
  pinned to the trace. A Gazebo episode closes that last gap.
- The per-step drift number (0.02 over all 1,863 aligned steps) dilutes the
  45/45 decision inversion across the high-rate Twist stream; the gate's
  attribution and the decision histogram carry the signal. Scoring drift on
  semantic actions only — or the σ-anchored `DecisionGate` for captioner swaps —
  sharpens it.

## σ-anchored, from the recorded seams (the fidelity bridge)

The recorded episode's own decision distributions (fidelity bridge, §7: 8 extra
samples per tick from the same endpoint into the sibling `*.samples` episode —
360 recorded samples) put a measured noise floor under the drift:

```
tick 0..9: div=1.000  sigma=0.000  excess=1.000     (mean over ticks: 1.000)
golden D(tick): {Move("move forwards"): 1.0}
```

At this context the recorded decider is fully decision-stable despite
temperature 0.7 (σ = 0.000, measured — not assumed), so the bad rule's total
variation of 1.000 is **entirely attributable excess**: the strongest possible
form of "this is regression, not sampling noise."

The in-process, deterministic version of this same demonstration (with the
σ-anchored decision gate catching a low-surface flip) lives in
`tests/test_baselines.py` and `tests/test_review_regressions.py`; this page is
the real-runtime, real-model version.
