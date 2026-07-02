"""Pure-stdlib OTLP/JSON span export for recorded episodes (engineering spec §5.4, §11).

Turns a stored `Episode` into a standard OTLP `resourceSpans` document (JSON), built
on `proxy.otel.to_span` — POST it to any OTel-GenAI backend (Grafana Tempo, Phoenix,
Langfuse) and the trace loads unchanged. This is the "existing observability stays
green" story made concrete.

Deliberately dependency-free: the SeamEvent -> attribute mapping already lives in
`otel.py`; the only missing step is serializing a known nested dict, which is stdlib
`json`. Pulling `opentelemetry-sdk` (grpc + protobuf, built for live in-process
export) to emit a static file from already-recorded episodes is unjustified weight.

`traceId`/`spanId` are derived deterministically (sha256), so re-exporting an episode
is byte-identical. No pickle (invariant 3). Spans carry the request DIGEST, never the
raw request/response payload.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from plumbline.core.trace import Episode, JSONValue, SeamEvent
from plumbline.proxy.otel import to_span


def episode_to_otlp(episode: Episode, *, service_name: str = "plumbline") -> dict[str, JSONValue]:
    """A full OTLP resourceSpans -> scopeSpans -> spans document for one episode."""
    spans: list[JSONValue] = [event_to_otlp_span(event) for event in episode.events]
    scope_spans: list[JSONValue] = [{"scope": {"name": "plumbline"}, "spans": spans}]
    resource_attrs: list[JSONValue] = [_key_value("service.name", service_name)]
    resource_spans: list[JSONValue] = [
        {"resource": {"attributes": resource_attrs}, "scopeSpans": scope_spans}
    ]
    return {"resourceSpans": resource_spans}


def event_to_otlp_span(event: SeamEvent) -> dict[str, JSONValue]:
    """One OTLP span; name and attributes come from `to_span(event)`."""
    span = to_span(event)
    return {
        "traceId": _trace_id(event.episode_id),
        "spanId": _span_id(event.episode_id, event.seq),
        "name": span.name,
        "kind": 3,  # SPAN_KIND_CLIENT
        "startTimeUnixNano": str(_start_nanos(event)),
        "endTimeUnixNano": str(_end_nanos(event)),
        "attributes": _otlp_attributes(span.attributes),
    }


def write_otlp(episode: Episode, path: str | Path, *, service_name: str = "plumbline") -> None:
    document = episode_to_otlp(episode, service_name=service_name)
    # allow_nan=False (matching canonical_dumps): a non-finite param must not emit the
    # non-standard NaN/Infinity tokens that strict parsers and OTLP collectors reject.
    Path(path).write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _otlp_attributes(attrs: Mapping[str, JSONValue]) -> list[JSONValue]:
    return [_key_value(key, value) for key, value in attrs.items()]


def _key_value(key: str, value: JSONValue) -> dict[str, JSONValue]:
    return {"key": key, "value": _any_value(value)}


def _any_value(value: JSONValue) -> dict[str, JSONValue]:
    # bool BEFORE int: bool is an int subclass. 64-bit ints are strings in proto3 JSON.
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value)}  # list/dict/None -> stringified


def _trace_id(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:32]


def _span_id(episode_id: str, seq: int) -> str:
    return hashlib.sha256(f"{episode_id}:{seq}".encode()).hexdigest()[:16]


def _start_nanos(event: SeamEvent) -> int:
    return int(event.wall_ts * 1e9)


def _end_nanos(event: SeamEvent) -> int:
    return _start_nanos(event) + int(event.latency_ms * 1e6)
