"""OTLP export surfaces the recorded upstream HTTP status (engineering spec §5.4, §11).

Regression guard: an error-saturated episode (upstream 429/5xx) must export spans that
carry the status code AND an ERROR span status, so operators see failures in
Tempo/Grafana/Phoenix instead of an all-green trace.
"""

from typing import Any

from plumbline.core.seam import Seam
from plumbline.core.trace import Episode, JSONValue, Payload, SeamEvent, canonicalize
from plumbline.observability.otlp import (
    HTTP_RESPONSE_STATUS_CODE,
    episode_to_otlp,
    event_to_otlp_span,
)

_STATUS_ATTR = HTTP_RESPONSE_STATUS_CODE


def _event(seq: int, params: dict[str, JSONValue]) -> SeamEvent:
    request = Payload(inline={"messages": [{"role": "user", "content": "hi"}]})
    return SeamEvent(
        episode_id="ep",
        seq=seq,
        seam=Seam.SENSOR_TO_CAPTION,
        logical_tick=0,
        wall_ts=1.0,
        request=request,
        response=Payload(inline={"choices": [{"message": {"content": "x"}}]}),
        model_id="openai/gpt-4o",
        params=params,
        request_digest=canonicalize(request).digest,
        latency_ms=12.0,
    )


def _attrs(span: Any) -> dict[str, Any]:
    """Flatten OTLP attribute list back to {key: scalar}."""
    out: dict[str, Any] = {}
    for attr in span["attributes"]:
        value = attr["value"]
        out[attr["key"]] = next(iter(value.values()))
    return out


def test_5xx_span_carries_status_code_and_error_status() -> None:
    span: Any = event_to_otlp_span(_event(0, {"plumbline.http_status": 503}))
    # proto3 JSON: 64-bit ints are strings.
    assert _attrs(span)[_STATUS_ATTR] == "503"
    assert span["status"] == {"code": 2, "message": "HTTP 503"}


def test_200_span_has_status_code_but_no_error_status() -> None:
    span: Any = event_to_otlp_span(_event(0, {"plumbline.http_status": 200}))
    assert _attrs(span)[_STATUS_ATTR] == "200"
    assert "status" not in span  # OK/unset stays green


def test_429_is_an_error() -> None:
    span: Any = event_to_otlp_span(_event(0, {"plumbline.http_status": 429}))
    assert span["status"]["code"] == 2


def test_absent_status_is_unchanged() -> None:
    span: Any = event_to_otlp_span(_event(0, {"temperature": 0.2}))
    assert _STATUS_ATTR not in _attrs(span)
    assert "status" not in span


def test_mixed_episode_only_5xx_spans_are_errors() -> None:
    episode = Episode(
        episode_id="ep",
        events=(
            _event(0, {"plumbline.http_status": 200}),
            _event(1, {"plumbline.http_status": 503}),
        ),
        metadata={},
    )
    document: Any = episode_to_otlp(episode)
    spans = document["resourceSpans"][0]["scopeSpans"][0]["spans"]
    ok_span, err_span = spans[0], spans[1]

    assert _attrs(ok_span)[_STATUS_ATTR] == "200"
    assert "status" not in ok_span

    assert _attrs(err_span)[_STATUS_ATTR] == "503"
    assert err_span["status"] == {"code": 2, "message": "HTTP 503"}
