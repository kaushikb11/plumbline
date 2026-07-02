"""Proxy behaviour beyond the zero-touch invariant (eng spec §4.2, §5.4, §5.5).

Covers replay (faithful by digest, counterfactual halt-on-divergence), provider
normalizers (seam tagging, model/params, data-URL blob extraction), SSE capture
and exact reframing, the OTel-GenAI span mapping, and the async HTTP proxy's
zero-touch record path (driven via asyncio.run — no pytest-asyncio dependency).
"""

import asyncio
import json
from collections.abc import Mapping

import pytest
from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.matcher import EmbeddingMatcher
from plumbline.core.recorder import Recorder
from plumbline.core.replayer import DivergencePolicy
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent
from plumbline.proxy import (
    AnthropicMessagesNormalizer,
    AsyncHTTPProxy,
    CapturedStream,
    GeminiNormalizer,
    HTTPRequest,
    HTTPResponse,
    OpenAIChatNormalizer,
    ProxyDivergence,
    RecordingProxy,
    ReplayingProxy,
    assemble_openai,
    seam_event_attributes,
    split_sse,
)


def _inline(payload: Payload) -> dict[str, JSONValue]:
    assert isinstance(payload.inline, dict)
    return payload.inline


def _echo_upstream(request: Payload) -> Payload:
    return Payload(inline={"answer": _inline(request)["q"]})


def _record_two_calls() -> tuple[TraceStore, Payload, Payload, Payload, Payload]:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    proxy = RecordingProxy(_echo_upstream, recorder)
    ctx = Context(episode_id="ep", model_id="openai/gpt-4o", params={"temperature": 0.0})
    req_a = Payload(inline={"q": "alpha"})
    req_b = Payload(inline={"q": "beta"})
    resp_a = proxy.forward(req_a, ctx)
    resp_b = proxy.forward(req_b, ctx)
    proxy.close("ep")
    return store, req_a, req_b, resp_a, resp_b


def test_faithful_replay_serves_recorded_response_by_digest() -> None:
    store, req_a, req_b, resp_a, resp_b = _record_two_calls()
    ctx = Context(episode_id="ep", model_id=None, params={})
    replay = ReplayingProxy(store, "ep")

    assert replay.faithful(req_a, ctx) == resp_a
    assert replay.faithful(req_b, ctx) == resp_b


def test_counterfactual_halts_and_serves_no_stale_response() -> None:
    store, req_a, _req_b, _resp_a, _resp_b = _record_two_calls()
    ctx = Context(episode_id="ep", model_id=None, params={})
    replay = ReplayingProxy(
        store,
        "ep",
        matchers={Seam.FUSE_TO_DECIDE: EmbeddingMatcher(threshold=0.2)},
        on_divergence=DivergencePolicy.HALT,
    )

    diverged = Payload(inline={"q": "something entirely unrelated and far away"})
    with pytest.raises(ProxyDivergence) as excinfo:
        replay.counterfactual(diverged, ctx)
    assert excinfo.value.seam is Seam.FUSE_TO_DECIDE
    assert excinfo.value.distance > 0.2


def test_openai_normalizer_classifies_and_extracts_image_blob() -> None:
    normalizer = OpenAIChatNormalizer()
    assert normalizer.handles("https://api.openai.com/v1/chat/completions")

    text = normalizer.normalize_request(
        {"model": "gpt-4o", "temperature": 0.5, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert text.seam is Seam.FUSE_TO_DECIDE
    assert text.model_id == "openai/gpt-4o"
    assert text.params["temperature"] == 0.5
    assert not text.blobs

    vision = normalizer.normalize_request(
        {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                        },
                    ],
                }
            ],
        }
    )
    assert vision.seam is Seam.SENSOR_TO_CAPTION
    assert len(vision.blobs) == 1
    # The raw data URL is replaced by a blob marker; raw bytes go to the store.
    assert "data:image/png;base64" not in vision.digest_key
    assert next(iter(vision.blobs.values())) == b"hello"


def test_gemini_and_anthropic_normalizers() -> None:
    gemini = GeminiNormalizer()
    assert gemini.handles(
        "https://generativelanguage.googleapis.com/v1/models/gemini:generateContent"
    )
    g = gemini.normalize_request(
        {"model": "gemini-1.5", "generationConfig": {"temperature": 0.9, "maxOutputTokens": 100}}
    )
    assert g.model_id == "gemini/gemini-1.5"
    assert g.params["temperature"] == 0.9
    assert g.params["maxOutputTokens"] == 100

    anthropic = AnthropicMessagesNormalizer()
    assert anthropic.handles("https://api.anthropic.com/v1/messages")
    a = anthropic.normalize_request(
        {"model": "claude-3", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert a.model_id == "anthropic/claude-3"
    assert a.params["max_tokens"] == 64


def test_sse_split_reframes_exactly_and_assembles() -> None:
    raw = (
        'data: {"id":"x","model":"gpt-4o","choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    chunks = split_sse(raw)
    stream = CapturedStream(chunks)
    # Reframing reproduces the original byte stream exactly (§14.3).
    assert stream.raw == raw

    assembled = assemble_openai(stream)
    assert isinstance(assembled, dict)
    assert "Hello" in json.dumps(assembled)


def test_otel_genai_attributes_from_seam_event() -> None:
    event = SeamEvent(
        episode_id="ep",
        seq=3,
        seam=Seam.FUSE_TO_DECIDE,
        logical_tick=3,
        wall_ts=0.0,
        request=Payload(inline={"prompt": "decide"}),
        response=Payload(
            inline={"id": "resp_1", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        ),
        model_id="openai/gpt-4o",
        params={"temperature": 0.7},
        request_digest="abc123",
        latency_ms=1.0,
    )
    attrs = seam_event_attributes(event)
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["gen_ai.response.id"] == "resp_1"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["plumbline.seam"] == "fuse_to_decide"
    assert attrs["plumbline.request_digest"] == "abc123"


class _FakeTransport:
    def __init__(self, response: HTTPResponse) -> None:
        self.response = response
        self.sent: list[HTTPRequest] = []

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        self.sent.append(request)
        return self.response


def test_async_http_proxy_record_is_zero_touch() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    upstream_body: Mapping[str, JSONValue] = {
        "id": "r",
        "choices": [{"message": {"role": "assistant", "content": "avoid"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    upstream_response = HTTPResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps(upstream_body).encode("utf-8"),
    )
    transport = _FakeTransport(upstream_response)
    proxy = AsyncHTTPProxy(transport=transport, recorder=recorder, store=store)

    ctx = Context(episode_id="http-ep", model_id=None, params={})
    request = HTTPRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=json.dumps(
            {
                "model": "gpt-4o",
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "go?"}],
            }
        ).encode("utf-8"),
    )

    returned = asyncio.run(proxy.record(request, ctx))

    # Zero-touch: the runtime receives the exact upstream response, forwarded once.
    assert returned is upstream_response
    assert len(transport.sent) == 1
    assert transport.sent[0] is request

    events = store.load_episode("http-ep").events
    assert len(events) == 1
    assert events[0].seam is Seam.FUSE_TO_DECIDE
    assert events[0].model_id == "openai/gpt-4o"
    assert seam_event_attributes(events[0])["gen_ai.system"] == "openai"
