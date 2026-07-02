"""Regression locks for the review fixes across all three tranches.

Each test pins a bug the deep review found and this pass fixed, so a re-regression
is caught. Grouped by area; see the review for the originating findings.
"""

from collections.abc import Mapping

import pytest
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, NumericToleranceMatcher
from plumbline.core.trace import JSONValue, Payload, canonical_dumps
from plumbline.fidelity import fusion_loss
from plumbline.fidelity.judge import _parse_equivalent

# --- matchers (core) --------------------------------------------------------


def test_numeric_matcher_compares_nested_and_list_fields() -> None:
    matcher = NumericToleranceMatcher(rtol=1e-3, atol=1e-3)
    assert matcher.matches(Payload({"pose": {"x": 1.0}}), Payload({"pose": {"x": 1.0}})).is_match
    # Nested / list divergences must be caught, not vacuously matched.
    assert not matcher.matches(
        Payload({"pose": {"x": 1.0}}), Payload({"pose": {"x": 9.0}})
    ).is_match
    assert not matcher.matches(Payload([1.0, 2.0, 3.0]), Payload([1.0, 2.0, 9.0])).is_match


def test_numeric_matcher_missing_field_has_nonzero_distance() -> None:
    matcher = NumericToleranceMatcher(rtol=1e-3, atol=1e-3)
    verdict = matcher.matches(Payload({"x": 1.0, "y": 2.0}), Payload({"x": 1.0}))
    assert not verdict.is_match
    assert verdict.distance >= 1.0  # structural mismatch is not distance 0.0


def test_numeric_matcher_no_numeric_fields_falls_back_to_exact() -> None:
    matcher = NumericToleranceMatcher(rtol=1e-3, atol=1e-3)
    assert matcher.matches(Payload({"s": "a"}), Payload({"s": "a"})).is_match
    assert not matcher.matches(Payload({"s": "a"}), Payload({"s": "b"})).is_match


def test_embedding_matcher_identical_empty_text_is_a_match() -> None:
    payload = Payload({"x": 1})  # no string leaves -> empty vector
    verdict = EmbeddingMatcher(threshold=0.2).matches(payload, payload)
    assert verdict.is_match
    assert verdict.distance == 0.0


def test_exact_matcher_distinguishes_int_float_bool() -> None:
    matcher = ExactMatcher()
    assert not matcher.matches(Payload({"a": 1}), Payload({"a": 1.0})).is_match
    assert not matcher.matches(Payload({"a": True}), Payload({"a": 1})).is_match
    assert matcher.matches(Payload({"a": 1}), Payload({"a": 1})).is_match


def test_canonical_dumps_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        canonical_dumps({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_dumps({"x": float("inf")})


# --- judge (fidelity) -------------------------------------------------------


def _judge_reply(text: str) -> Payload:
    return Payload({"choices": [{"message": {"content": text}}]})


def test_judge_parses_negated_diverge_as_equivalent() -> None:
    assert _parse_equivalent(_judge_reply("They do not diverge; behavior is identical.")) is True
    assert (
        _parse_equivalent(_judge_reply("NOT EQUIVALENT — the robot turns the other way")) is False
    )
    assert _parse_equivalent(_judge_reply("EQUIVALENT, same plan")) is True


# --- fusion_loss (fidelity) -------------------------------------------------


def _obstacle_decider(context: str) -> Mapping[str, JSONValue]:
    return {"action": "stop" if "obstacle" in context else "advance"}


def test_fusion_loss_is_bounded_mean_not_unbounded_sum() -> None:
    loss = fusion_loss(
        _obstacle_decider,
        "the path is clear",
        ["obstacle A", "obstacle B", "obstacle C"],
        8,
        salient=lambda caption: caption,
    )
    assert 0.0 <= loss <= 1.0  # mean over captions; the raw sum would be 3.0


def test_fusion_loss_rejects_mismatched_weights_length() -> None:
    with pytest.raises(ValueError):
        fusion_loss(_obstacle_decider, "F", ["a", "b"], 4, weights=[1.0])


# --- gate policies (regression) ---------------------------------------------


def test_gate_policies_are_differentiated() -> None:
    from plumbline.regression.gate import FailurePolicy, _passes

    # mean passes, max fails -> ANY != AGGREGATE
    spread = [0.0, 0.0, 0.0, 0.0, 0.4]
    assert _passes(spread, 0.1, FailurePolicy.ANY, 0.95) is False
    assert _passes(spread, 0.1, FailurePolicy.AGGREGATE, 0.95) is True
    # P95 tolerates the single worst outlier -> QUANTILE != ANY
    outlier = [0.0] * 19 + [1.0]
    assert _passes(outlier, 0.1, FailurePolicy.ANY, 0.95) is False
    assert _passes(outlier, 0.1, FailurePolicy.QUANTILE, 0.95) is True


def test_quantile_nearest_rank_is_not_off_by_one() -> None:
    from plumbline.regression.gate import FailurePolicy, _passes

    # Exactly the top 5% are bad: nearest-rank P95 (index 94 of 100) is 0.0 -> pass.
    # The old int(0.95*100)=95 wrongly read index 95 (=1.0) and failed.
    drifts = [0.0] * 95 + [1.0] * 5
    assert _passes(drifts, 0.1, FailurePolicy.QUANTILE, 0.95) is True


# --- streaming (proxy) ------------------------------------------------------


def test_split_sse_handles_crlf_framing() -> None:
    from plumbline.proxy.streaming import split_sse

    raw = "data: a\r\n\r\ndata: b\r\n\r\n"
    chunks = split_sse(raw)
    assert "".join(chunks) == raw  # exact round-trip
    assert len(chunks) == 2  # framed per-event, not collapsed into one


# --- session thread-safety (higher layers) ----------------------------------


def test_session_record_after_close_is_dropped_not_crash() -> None:
    from plumbline.adapters.base import BusSample
    from plumbline.core.store import TraceStore
    from plumbline.session import RecordingSession

    store = TraceStore()
    session = RecordingSession(store, episode_id="ep", metadata={})
    session.open()
    session.record_bus_sample(
        BusSample(key_expr="om1/agent/actions/go2", payload={"n": 1}, wall_ts=0.0)
    )
    session.close()
    # A late bus sample after close() must be dropped, not crash the tap thread.
    session.record_bus_sample(
        BusSample(key_expr="om1/agent/actions/go2", payload={"n": 2}, wall_ts=1.0)
    )
    assert len(store.load_episode("ep").events) == 1


# --- observability (higher layers) ------------------------------------------


def test_latency_monitor_zero_baseline_is_not_a_free_pass() -> None:
    from plumbline.core.seam import Seam
    from plumbline.core.trace import SeamEvent, canonicalize
    from plumbline.observability.baselines import latency_monitor

    def model_event(latency_ms: float) -> SeamEvent:
        request = Payload({"model": "m", "messages": []})
        return SeamEvent(
            episode_id="e",
            seq=0,
            seam=Seam.FUSE_TO_DECIDE,
            logical_tick=0,
            wall_ts=0.0,
            request=request,
            response=Payload({"choices": [{"message": {"content": "x"}}]}),
            model_id="openai/m",
            params={},
            request_digest=canonicalize(request).digest,
            latency_ms=latency_ms,
        )

    verdict = latency_monitor([model_event(0.0)], [model_event(100.0)])
    assert verdict.healthy is False  # zero baseline must not pass an arbitrarily slow candidate


# --- bench openai client (higher layers) ------------------------------------


def test_openai_client_raises_on_malformed_response() -> None:
    from plumbline.bench.openai_client import MalformedResponse, _message_content, _normalize_action

    with pytest.raises(MalformedResponse):
        _message_content({"error": {"message": "boom"}})
    with pytest.raises(MalformedResponse):
        _message_content({"choices": []})
    # earliest-mentioned action wins
    assert _normalize_action("move_forward, not stop", ("move_forward", "stop")) == "move_forward"
