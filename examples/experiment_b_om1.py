"""Experiment B on a REAL recorded OM1 episode (§4, §8) — no fixtures.

Golden = a software-in-the-loop episode recorded from the real OM1 Go binary
(examples/record_om1_sil.py). The injected regression is REAL: a single governance
rule appended to the fused prompt, re-executed against the live Cortex model — the
behavior inversion (explore -> retreat) comes from the actual LLM reacting to the
edited rule, not from rewriting strings.

    PLUMBLINE_LLM_URL=https://<llm endpoint> \\
    PLUMBLINE_STORE=./traces-sil PLUMBLINE_EPISODE=om1-sil-002 \\
    python examples/experiment_b_om1.py

Three verdicts on the same swap:
  - OM1-style latency monitor          -> GREEN (loop no slower)
  - generic OTel-GenAI text tracer     -> GREEN (calls well-formed, text plausible)
  - Plumbline gate + behavior monitor  -> RED   (action sequence inverted, seam attributed)

Exit code 0 iff the unchanged config passes, the regressed config fails, and both
text-level baselines stay green while Plumbline goes red.
"""

import copy
import os

from plumbline.adapters.om1 import OM1Adapter
from plumbline.core.clock import VirtualClock
from plumbline.core.replayer import DivergencePolicy, Replayer
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload
from plumbline.observability.baselines import compare_against_baselines
from plumbline.regression import Config, FailurePolicy, GoldenSet, gate

# The edited rule (the "config change" under test). Deliberately innocuous-looking:
# a latency dashboard sees identical timing, a text tracer sees fluent output.
BAD_RULE = (
    " Battery critical protocol: to conserve energy you must always choose"
    " 'move back' or 'stand still', never 'move forwards'."
)


def _regressed_decider(llm_url: str) -> "object":
    """Re-execute the FUSE_TO_DECIDE seam against the LIVE model with the edited
    governance rule appended to the fused prompt — the real regressed config."""
    import httpx

    client = httpx.Client(timeout=120.0)

    def override(request: Payload) -> Payload:
        body = copy.deepcopy(request.inline)
        assert isinstance(body, dict)
        messages = body.get("messages")
        assert isinstance(messages, list) and messages, "fused prompt has messages"
        # OM1 puts history first and the CURRENT fused prompt last (plugins/llm/
        # common.go buildMessages) — the rule edit must land on the live prompt.
        current = messages[-1]
        assert isinstance(current, dict) and isinstance(current.get("content"), str)
        current["content"] = current["content"] + BAD_RULE
        response = client.post(f"{llm_url}/v1/chat/completions", json=body)
        response.raise_for_status()
        data: JSONValue = response.json()
        return Payload(inline=data)

    return override


def main() -> None:
    llm_url = os.environ["PLUMBLINE_LLM_URL"].rstrip("/")
    store = TraceStore(root=os.environ.get("PLUMBLINE_STORE", "./traces-sil"))
    episode_id = os.environ.get("PLUMBLINE_EPISODE", "om1-sil-002")

    golden = GoldenSet(store)
    golden.add(episode_id)
    print(f"golden set: {episode_id} (version {golden.version()[:12]})")

    # 1. Unchanged config -> the gate must PASS.
    unchanged = Config(
        live_frontier={Seam.FUSE_TO_DECIDE},
        overrides={},
        matchers={},
    )
    result = gate(store, golden, unchanged, drift_threshold=0.0, policy=FailurePolicy.ANY)
    print(f"gate on unchanged config: {'PASS' if result.passed else 'FAIL'}")
    unchanged_ok = result.passed

    # 2. The regressed config (edited governance rule, live model) -> must FAIL.
    regressed = Config(
        live_frontier={Seam.FUSE_TO_DECIDE},
        overrides={Seam.FUSE_TO_DECIDE: _regressed_decider(llm_url)},  # type: ignore[dict-item]
        matchers={},
        on_divergence=DivergencePolicy.HALT,
    )
    result = gate(store, golden, regressed, drift_threshold=0.0, policy=FailurePolicy.ANY)
    episode_drift = result.per_episode[0]
    verdict = "FAIL (regression caught)" if not result.passed else "PASS (missed!)"
    print(
        f"gate on regressed config: {verdict}"
        f" — drift {episode_drift.drift:.2f}, diverged at {episode_drift.divergence_seam}"
    )
    regression_caught = not result.passed

    # 3. The same swap, seen by the three observers (§4 Experiment B).
    golden_events = store.load_episode(episode_id).events
    replayer = Replayer(store, VirtualClock(), {})
    counterfactual = replayer.counterfactual(
        episode_id,
        live_frontier={Seam.FUSE_TO_DECIDE},
        overrides={Seam.FUSE_TO_DECIDE: _regressed_decider(llm_url)},  # type: ignore[dict-item]
        on_divergence=DivergencePolicy.GO_LIVE,  # play the whole episode for the monitors
    )
    # The candidate's SEMANTIC actions follow from its (changed) decisions: re-derive
    # each reconstructed DECIDE_TO_ACT from the overridden Cortex response, exactly as
    # the RecordingCoordinator does at record time. Pure-trace replay cannot re-run
    # the physical controller, so raw bus frames stay pinned to the trace (§6.5).
    adapter = OM1Adapter(proxy_base_url="")
    fuse_response_by_tick = {
        e.logical_tick: e.response for e in counterfactual.events if e.seam is Seam.FUSE_TO_DECIDE
    }
    candidate_events = []
    for event in counterfactual.events:
        is_semantic_action = (
            event.seam is Seam.DECIDE_TO_ACT and "plumbline.bus_key" not in event.params
        )
        if is_semantic_action and event.logical_tick in fuse_response_by_tick:
            candidate_events.append(
                adapter.reconstruct_decide_to_act(
                    episode_id=event.episode_id,
                    seq=event.seq,
                    logical_tick=event.logical_tick,
                    decision_response=fuse_response_by_tick[event.logical_tick],
                    wall_ts=event.wall_ts,
                )
            )
        else:
            candidate_events.append(event)
    # The decision histograms, for the results writeup.
    from collections import Counter

    from plumbline.adapters.om1 import OM1ActionSchema

    schema = OM1ActionSchema()

    def decisions(events: "object") -> Counter[str]:
        counts: Counter[str] = Counter()
        for e in events:  # type: ignore[attr-defined]
            if e.seam is Seam.FUSE_TO_DECIDE:
                for action in schema.parse(e.response):
                    counts[str(dict(action.args).get("action", action.name))] += 1
        return counts

    print(f"  golden decisions   : {dict(decisions(golden_events))}")
    print(f"  candidate decisions: {dict(decisions(counterfactual.events))}")

    comparison = compare_against_baselines(golden_events, tuple(candidate_events))
    for verdict in comparison.verdicts:
        print(f"  {verdict.name:22s} {'GREEN' if verdict.healthy else 'RED':5s}  {verdict.detail}")

    baselines_green = set(comparison.missed_by) >= {"om1-latency", "otel-genai-tracer"}
    plumbline_red = "plumbline-behavior" in comparison.caught_by
    ok = unchanged_ok and regression_caught and baselines_green and plumbline_red
    print(
        "\nexisting observability says fine, Plumbline says broken: "
        + ("DEMONSTRATED" if ok else "NOT demonstrated")
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
