# Observability & Grafana (WS4, spec §11)

Plumbline exports recorded episodes and analysis results into JSON artifacts that
Grafana renders. There is **no new Python dependency** — the exporter is pure
stdlib (`json` + `hashlib`), building on the OTel-GenAI attribute mapping already in
`plumbline/proxy/otel.py`. Grafana, the Infinity datasource, and Grafana Tempo are
external tools, not Python packages.

## Two data families

Spans alone can't drive the regression view: drift and divergence are *analysis
outputs*, not span attributes. So there are two artifact families:

1. **OTLP/JSON spans** — the standard interchange. `plumbline export EP --format otlp`
   writes an OTLP `resourceSpans` document (from `to_span`). POST it to a Grafana
   Tempo OTLP/HTTP endpoint (or any OTel-GenAI backend — Tempo, Phoenix, Langfuse)
   and the trace loads unchanged. `traceId`/`spanId` are sha256-derived, so
   re-exporting an episode is byte-identical.
2. **Flattened dashboard feeds** — what the panels bind to via the **Infinity**
   datasource (reads JSON files/HTTP, no backend, no collector). Built by
   `plumbline.observability.feed`: `episode_telemetry` (per-seam / per-tick rollups),
   `gate_feed` (drift / divergence), `baseline_feed` (Experiment-B verdicts).

## Default path (zero backend)

```bash
plumbline export go2-001 --store ./traces -o telemetry.json --format telemetry
plumbline gate gate_config.py --emit-feed gate.json
```

1. Install the [Infinity datasource](https://grafana.com/grafana/plugins/yesoreyeram-infinity-datasource/).
2. Import the dashboards in `plumbline/observability/grafana/`
   (`plumbline-telemetry.json`, `plumbline-regression.json`).
3. Point each panel's Infinity target at the exported JSON (URL or inline). The
   `${datasource}` template variable selects your Infinity instance.

The telemetry dashboard shows per-seam latency (mean + p95), token usage, and
seams-per-tick. The regression dashboard shows per-episode behavioral drift,
diverged fraction, the divergence seam, and the Experiment-B green/red contrast
(feed it a `baseline_feed(...)` JSON, e.g. from `examples/experiment_b.py`).

## Backend path (if you already run a collector)

POST the OTLP export to Tempo's OTLP/HTTP traces endpoint:

```bash
plumbline export go2-001 --store ./traces -o spans.json --format otlp
curl -X POST http://localhost:4318/v1/traces -H 'Content-Type: application/json' -d @spans.json
```

This needs a **running collector** (Tempo/OTel Collector) — the default path above
does not.

## Honesty box

- **No `opentelemetry-sdk` dependency.** The SDK is for live in-process export
  (grpc/protobuf, batch processors). Plumbline exports from *already-recorded*
  episodes in batch; the attribute map already exists in `otel.py`, so this is a
  stdlib `json.dumps` of a known shape. Keeping the substrate light is a project
  rule.
- **Latency panels reflect the recorded `latency_ms`, not scheduler wall-clock.**
  Plumbline's determinism envelope is model-I/O only (invariant 4, §3.4/§14.4); no
  panel implies scheduler timing.
- **Token panels appear only when the recording carried a `usage` block.** The feed
  omits token fields otherwise rather than reporting zero.
- **Spans carry the request *digest*, never the raw request/response payload.**
- Dashboards are covered by `tests/test_grafana_dashboards.py`, which fails CI if a
  panel references a field the feeds don't actually emit.
