"""OpenTelemetry GenAI-aligned span schema (engineering spec §5.4).

Each `SeamEvent` maps to a span. Attributes follow the `gen_ai.*` semantic
conventions where defined and add a `plumbline.*` namespace for what GenAI
conventions do not cover (the seam, logical tick, request digest, episode id,
seq). A Plumbline trace is therefore viewable in any OTel-GenAI-aware backend
(Langfuse, Phoenix, Grafana Tempo) — the basis for the "existing observability
stays green" demonstration in §7.

The mapping is *derived* from a SeamEvent (no fields are stored redundantly): the
provider system is parsed from `model_id` (`"openai/gpt-4o"` -> system "openai",
model "gpt-4o"), request params supply the request attributes, and the response
inline supplies the response id / token usage.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from plumbline.core.trace import JSONValue, SeamEvent

# gen_ai.* keys (OTel GenAI semantic conventions).
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# plumbline.* keys (what GenAI conventions do not cover).
PLUMBLINE_SEAM = "plumbline.seam"
PLUMBLINE_EPISODE_ID = "plumbline.episode_id"
PLUMBLINE_SEQ = "plumbline.seq"
PLUMBLINE_LOGICAL_TICK = "plumbline.logical_tick"
PLUMBLINE_REQUEST_DIGEST = "plumbline.request_digest"


@dataclass(frozen=True)
class OTelSpan:
    """A minimal OTel-shaped span derived from a SeamEvent (§5.4)."""

    name: str
    attributes: dict[str, JSONValue]


def seam_event_attributes(event: SeamEvent) -> dict[str, JSONValue]:
    """Render a SeamEvent as a flat `gen_ai.*` + `plumbline.*` attribute map."""
    attrs: dict[str, JSONValue] = {
        PLUMBLINE_SEAM: event.seam.value,
        PLUMBLINE_EPISODE_ID: event.episode_id,
        PLUMBLINE_SEQ: event.seq,
        PLUMBLINE_LOGICAL_TICK: event.logical_tick,
        PLUMBLINE_REQUEST_DIGEST: event.request_digest,
        GEN_AI_OPERATION_NAME: "chat",
    }

    system, model = _split_model(event.model_id)
    if system is not None:
        attrs[GEN_AI_SYSTEM] = system
    if model is not None:
        attrs[GEN_AI_REQUEST_MODEL] = model

    # Accept both OpenAI/Anthropic (top_p, max_tokens) and Gemini (topP,
    # maxOutputTokens) param spellings so no provider silently loses attributes.
    for attr_key, param_keys in (
        (GEN_AI_REQUEST_TEMPERATURE, ("temperature",)),
        (GEN_AI_REQUEST_TOP_P, ("top_p", "topP")),
        (GEN_AI_REQUEST_MAX_TOKENS, ("max_tokens", "maxOutputTokens")),
    ):
        value = _first(event.params, param_keys)
        if value is not None:
            attrs[attr_key] = value

    response = event.response.inline
    if isinstance(response, dict):
        response_id = response.get("id") or response.get("responseId")
        if isinstance(response_id, str):
            attrs[GEN_AI_RESPONSE_ID] = response_id
        input_tokens, output_tokens = _usage_tokens(response)
        if input_tokens is not None:
            attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        if output_tokens is not None:
            attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens

    return attrs


def _usage_tokens(response: dict[str, JSONValue]) -> tuple[JSONValue, JSONValue]:
    """Input/output token counts from either the OpenAI/Anthropic `usage` object or
    Gemini's `usageMetadata`."""
    usage = response.get("usage")
    if isinstance(usage, dict):
        return (
            _first(usage, ("prompt_tokens", "input_tokens")),
            _first(usage, ("completion_tokens", "output_tokens")),
        )
    meta = response.get("usageMetadata")
    if isinstance(meta, dict):
        return meta.get("promptTokenCount"), meta.get("candidatesTokenCount")
    return None, None


def to_span(event: SeamEvent) -> OTelSpan:
    attrs = seam_event_attributes(event)
    model = attrs.get(GEN_AI_REQUEST_MODEL)
    name = f"chat {model}" if isinstance(model, str) else f"seam {event.seam.value}"
    return OTelSpan(name=name, attributes=attrs)


def _split_model(model_id: str | None) -> tuple[str | None, str | None]:
    if not model_id:
        return None, None
    if "/" in model_id:
        system, model = model_id.split("/", 1)
        return system, model
    return None, model_id


def _first(mapping: Mapping[str, JSONValue], keys: tuple[str, ...]) -> JSONValue:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None
