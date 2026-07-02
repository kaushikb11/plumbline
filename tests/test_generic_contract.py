"""Generic-adapter contract conformance (engineering spec §9.1).

Proves the frozen `Adapter` contract is not OM1-shaped: single-endpoint config,
no bus, two-seam classification, a flat action schema with two encodings.
"""

from plumbline.adapters.base import Action, Adapter
from plumbline.adapters.generic import DEFAULT_ACTIONS, GenericActionSchema, GenericAgentAdapter
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload

_PROXY = "http://localhost:8900"
_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def _vision_request() -> JSONValue:
    return {
        "model": "vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the scene."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
    }


def _decide_request() -> JSONValue:
    return {"model": "llm", "messages": [{"role": "user", "content": "Decide the next action."}]}


def test_conforms_to_adapter_protocol() -> None:
    adapter: Adapter = GenericAgentAdapter(proxy_base_url=_PROXY)  # structural conformance
    assert adapter.clock_hook() is None


def test_configure_proxy_is_a_single_endpoint() -> None:
    config = GenericAgentAdapter(proxy_base_url=_PROXY).configure_proxy()
    assert config.env == {"OPENAI_BASE_URL": f"{_PROXY}/v1"}
    assert config.config_fields == {}  # a neutral adapter declares no config-file schema
    assert config.proxy_base_url == _PROXY
    assert GenericAgentAdapter(proxy_base_url=_PROXY, append_v1=False).configure_proxy().env == {
        "OPENAI_BASE_URL": _PROXY
    }
    assert GenericAgentAdapter(
        proxy_base_url=_PROXY, base_url_env_var="LLM_BASE_URL"
    ).configure_proxy().env == {"LLM_BASE_URL": f"{_PROXY}/v1"}


def test_no_bus_tap_by_default() -> None:
    # The headline difference from OM1: the loop runs with no bus at all.
    assert GenericAgentAdapter(proxy_base_url=_PROXY).bus_tap() is None


def test_seam_of_classifies_only_perception_and_decide() -> None:
    adapter = GenericAgentAdapter(proxy_base_url=_PROXY)
    assert adapter.seam_of(Payload(inline=_vision_request()), _ENDPOINT) is Seam.SENSOR_TO_CAPTION
    assert adapter.seam_of(Payload(inline=_decide_request()), _ENDPOINT) is Seam.FUSE_TO_DECIDE
    # A text-only perception call is disambiguated by a caption marker.
    tagged = GenericAgentAdapter(proxy_base_url=_PROXY, caption_markers=("/transcriptions",))
    asr = "https://api.openai.com/v1/audio/transcriptions"
    assert tagged.seam_of(Payload(inline={"audio": "..."}), asr) is Seam.SENSOR_TO_CAPTION
    # It NEVER returns the reconstructed seams.
    for request in (_vision_request(), _decide_request()):
        assert adapter.seam_of(Payload(inline=request), _ENDPOINT) in (
            Seam.SENSOR_TO_CAPTION,
            Seam.FUSE_TO_DECIDE,
        )


def test_action_schema_parses_token_and_tool_call() -> None:
    schema = GenericActionSchema()
    assert schema.commands == DEFAULT_ACTIONS
    assert schema.parse(Payload(inline={"action": "turn_left"})) == (
        Action("act", "turn_left", {}),
    )
    tool_call: JSONValue = {
        "tool_calls": [{"function": {"name": "move_forward", "arguments": {"speed": 0.3}}}]
    }
    assert schema.parse(Payload(inline=tool_call)) == (
        Action("act", "move_forward", {"speed": 0.3}),
    )
    assert schema.parse(Payload(inline={"unrelated": 1})) == ()
