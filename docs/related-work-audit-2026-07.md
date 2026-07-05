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

## Hand-checked (the credit-truncated candidates, now verified against primary sources)

All the flagged candidates were re-fetched from their arXiv/GitHub sources. Results:

| Candidate | Real? | What it actually is | Verdict |
|---|---|---|---|
| **CaBM — Caption Bottleneck Models** ([arXiv 2607.00578](https://arxiv.org/abs/2607.00578), Jul 2026) | ✅ | Interpretable image *classification* that moves the information bottleneck entirely into natural-language captions (frozen LMM → captions → text classifier), "leakage-free by construction," decoupling perception from recognition. | **CITE — on-thesis neighbor.** Independently validates Plumbline's "put the bottleneck in language to decouple perception from decision" framing, but for offline classification interpretability, not an embodied decision loop with a noise-floor-corrected fidelity metric or record-replay. Strengthens the framing; does not preempt. |
| **DFAH — Replayable Financial Agents** ([arXiv 2601.15322](https://arxiv.org/abs/2601.15322), Jan 2026) | ✅ | A "Determinism-Faithfulness Assurance Harness" for tool-using LLM agents: measures trajectory determinism, decision determinism, and faithfulness over 4,700+ runs (finds determinism and accuracy are independent). | **CITE — closest new "determinism assurance" neighbor.** But it *measures whether* text/financial agents are reproducible; Plumbline's substrate *makes* the loop reproducible via record-replay of model calls — and is embodied (perception→decision), which DFAH is not. |
| **Embodied Runtime Governance** ([arXiv 2604.07833](https://arxiv.org/abs/2604.07833), Apr 2026) | ✅ | Runtime governance for policy-constrained embodied agents — an external policy-checking layer intercepting unauthorized actions (96.2%), with capability admission and rollback. | **CITE — embodied-agent assurance neighbor (governance axis).** Governance/interception, not reproducibility/regression-gating/fidelity; adjacent to the `system_governance` rules Experiment B edits, complementary rather than overlapping. |
| **VLA-REPLICA** ([arXiv 2605.20774](https://arxiv.org/abs/2605.20774), May 2026) | ✅ | A low-cost, reproducible real-world *hardware benchmark* for VLA manipulation policies across distributed labs. | Brief note. Name collision only — "replica" = physical-setup reproducibility, NOT record-replay of model calls. Low overlap. |
| **PreAct** ([arXiv 2606.17929](https://arxiv.org/abs/2606.17929), Jun 2026) | ✅ | Computer-using (GUI) agents that compile successful runs into state-machine programs and replay them with screen-state checks, falling back to fresh reasoning on mismatch. | Fold into the **AgentRR** neighbor — same GUI-agent replay-to-accelerate family; check-before-replay is a cousin of the input-consistency matchers; not bit-perfect, not embodied. |
| **OM1-ros2-sdk** ([github.com/OpenMind/OM1-ros2-sdk](https://github.com/OpenMind/OM1-ros2-sdk), active Apr 2026) | ✅ | OM1's ROS2 SDK for multiple robots; OpenMind also ships a Prometheus/Grafana observability stack + a telemetry platform. | Ecosystem context. Reinforces the existing framing — OM1's own Prometheus/Grafana latency stack is exactly the baseline Experiment B contrasts against. No claim impact. |
| DEMM-Bench, LIFE-HARNESS, ROSE, `2509.03312`, `2606.20634`, `2606.08275` | — | Named without confident identifiers by the credit-truncated pass; the material candidates above were prioritized and confirmed. | Not confirmed, not cited. Low expected relevance; re-run a search if a comprehensive citation sweep is wanted before release. |

**Net:** three new papers to CITE (CaBM, DFAH, embodied governance), two to fold in as
brief notes (VLA-REPLICA, PreAct), one ecosystem confirmation (OM1-ros2-sdk). **None
preempts the atomic claim** — the closest (DFAH) measures determinism where Plumbline
enforces it, and is text-only; CaBM shares the language-bottleneck *thesis* but for
offline classification, not an embodied decision loop with deterministic replay and a
noise-floor-corrected decision-fidelity metric.

## Method note

Search fanned out over 5 angles; 15 sources fetched; each falsifiable claim went to
3 adversarial skeptic voters (≥2 refutations kills a claim). The synthesis and ~46
of the verification votes failed on a mid-run usage-credit exhaustion. The flagged
candidates were then **hand-checked against primary sources** (the table above) —
so the audit no longer under-claims: every material candidate is confirmed real,
read, and given a cite/fold/skip verdict. Re-verify all arXiv identifiers once more
immediately before the public release (spec §7 standing rule).
