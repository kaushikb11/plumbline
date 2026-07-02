"""OTLP/JSON span export (engineering spec §5.4, §11)."""

import json
from typing import Any

from plumbline.core.seam import Seam
from plumbline.core.trace import Episode, JSONValue, Payload, SeamEvent, canonicalize
from plumbline.observability.otlp import _any_value, episode_to_otlp, event_to_otlp_span
from plumbline.proxy.otel import to_span


def _event(seq: int, seam: Seam, response: JSONValue, *, latency_ms: float = 12.0) -> SeamEvent:
    request = Payload(inline={"messages": [{"role": "user", "content": "hi"}]})
    return SeamEvent(
        episode_id="ep",
        seq=seq,
        seam=seam,
        logical_tick=0,
        wall_ts=1.0,
        request=request,
        response=Payload(inline=response),
        model_id="openai/gpt-4o",
        params={"temperature": 0.2},
        request_digest=canonicalize(request).digest,
        latency_ms=latency_ms,
    )


def _episode() -> Episode:
    return Episode(
        episode_id="ep",
        events=(
            _event(
                0,
                Seam.SENSOR_TO_CAPTION,
                {
                    "id": "r1",
                    "choices": [{"message": {"content": "a cat"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                },
            ),
            _event(1, Seam.FUSE_TO_DECIDE, {"choices": [{"message": {"content": "move"}}]}),
        ),
        metadata={},
    )


def test_span_ids_name_and_attributes() -> None:
    event = _episode().events[0]
    span: Any = event_to_otlp_span(event)
    assert len(span["traceId"]) == 32 and all(c in "0123456789abcdef" for c in span["traceId"])
    assert len(span["spanId"]) == 16
    assert span["name"] == to_span(event).name
    keys = {attr["key"] for attr in span["attributes"]}
    assert "plumbline.seam" in keys and "gen_ai.operation.name" in keys


def test_any_value_typing() -> None:
    assert _any_value(True) == {"boolValue": True}  # bool before int
    assert _any_value(5) == {"intValue": "5"}  # 64-bit ints are strings in proto3 JSON
    assert _any_value(1.5) == {"doubleValue": 1.5}
    assert _any_value("x") == {"stringValue": "x"}


def test_export_is_deterministic() -> None:
    assert json.dumps(episode_to_otlp(_episode()), sort_keys=True) == json.dumps(
        episode_to_otlp(_episode()), sort_keys=True
    )


def test_span_duration_matches_latency() -> None:
    event = _episode().events[0]
    span: Any = event_to_otlp_span(event)
    duration = int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])
    assert duration == round(event.latency_ms * 1e6)


def test_token_attributes_only_when_usage_recorded() -> None:
    document: Any = episode_to_otlp(_episode())
    spans = document["resourceSpans"][0]["scopeSpans"][0]["spans"]
    with_usage = {attr["key"] for attr in spans[0]["attributes"]}
    without_usage = {attr["key"] for attr in spans[1]["attributes"]}
    assert "gen_ai.usage.input_tokens" in with_usage
    assert "gen_ai.usage.input_tokens" not in without_usage


def test_no_raw_payload_leak() -> None:
    document = json.dumps(episode_to_otlp(_episode()))
    assert "plumbline.request_digest" in document  # the digest is carried...
    assert "a cat" not in document  # ...but never the response content
    assert '"role"' not in document  # ...nor the raw request messages


def test_valid_json_roundtrip() -> None:
    document = episode_to_otlp(_episode())
    assert json.loads(json.dumps(document)) == document
