# Related-work audit — 2026-07 (pre-release novelty re-check)

Engineering spec §7 requires re-verifying the novelty position immediately before
release. This is the Jan–Jul 2026 pass (a deep multi-source search with 3-vote
adversarial verification). It updates the README related-work section.

## Verified (survived 3-0 adversarial verification) — acted on in the README

| Work | Link | Overlap | Verdict |
|---|---|---|---|
| **AgenTracer** | in survey arXiv 2606.04990 | Already does **counterfactual replay + fault injection** for LLM multi-agent failure attribution | Counterfactual replay *per se* is no longer novel — README reworded to claim the *conjunction across the modality boundary*, not the swap alone. Cited + differentiated. |
| **CTA (Counterfactual Trace Auditing)** | arXiv 2605.11946 | Single-component (skill) swap; "divergence records" (terminology overlap) | Runs live twice, **not** deterministic replay; divergence records are post-hoc, not halt-on-divergence. Text agents only. Cited + differentiated. |
| **langchain-replay** | github.com/sixty-north/langchain-replay | Records/replays LLM agent decisions to JSONL (no pickle); pytest CI plugin | Closest OSS record-replay precedent. No counterfactual/divergence/fidelity/robotics. Cited + differentiated. |
| **Agent-tracing survey** | arXiv 2606.04990 (Jun 2026) | Taxonomy of LLM-agent tracing | Software agents only; no embodied/replay/reproducibility. Cited for framing. |
| **Trace-Based Assurance Framework** (already cited) | arXiv 2603.18096 | Confirmed: MAT with deterministic replay **and** a replay-based *regression-gating* mechanism analogous to Plumbline's gate | Still text/service-only, no perception/modality boundary. Differentiation holds. |

**Net effect on the claim:** the atomic novel claim's individual verbs
(counterfactual swap, decision record-replay, replay-based CI gating) are each now
prior art for *text* agents. The README claim was reworded to foreground the
defensible novelty: the **conjunction carried across the perception→language→
decision modality boundary** into an embodied loop, which no verified prior work does.

## UNVERIFIED — flagged candidates (verification aborted mid-run on a credit
limit; **re-check by hand before public release**)

These surfaced in search but did **not** complete 3-vote verification, so they are
NOT cited and NOT trusted. Each needs a manual primary-source read:

- `arXiv 2605.20774` — "VLA-REPLICA" (claimed record-replay for VLA models?)
- `arXiv 2606.17929` — "PreAct" (Pine AI; per-step check-then-act guardrail?)
- `arXiv 2601.15322` — claimed "deterministic LLM agents" / DiverseArm (Jan 2026)
- `arXiv 2604.07833` — claimed governance framework for embodied agents
- "DEMM-Bench" — claimed multimodal metric (no arXiv id captured)
- "LIFE-HARNESS" (Peking Univ.) — claimed embodied-agent harness
- "Caption Bottleneck Models (CaBM)" — claimed language-bottleneck metric — **most
  relevant to Experiment A's thesis if real; verify first**
- "ROSE" — claimed to exclude embodied/robot scope
- Also seen: `2509.03312`, `2606.20634`, `2606.08275`, `github.com/OpenMind/OM1-ros2-sdk`
  (a new OM1 repo — worth confirming for the ecosystem framing).

## Method note

Search fanned out over 5 angles; 15 sources fetched; each falsifiable claim went to
3 adversarial skeptic voters (≥2 refutations kills a claim). The synthesis and ~46
of the verification votes failed on a mid-run usage-credit exhaustion, so this audit
reports only the claims that reached a full 3-0 verdict plus an explicit
re-check list — deliberately under-claiming rather than citing unverified work.
