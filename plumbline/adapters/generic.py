"""A generic OpenAI-agent-loop adapter (engineering spec §9.1).

A second adapter whose whole purpose is to prove the `Adapter` contract is NOT
OM1-shaped. Unlike the OM1 adapter it has **no bus**: `bus_tap()` returns None, so
the record/replay/counterfactual loop must work when one of the two interception
mechanisms (§3.1: proxy + bus) is entirely absent. The action seam is *derived*
from the decision response rather than observed on a bus, `configure_proxy`
collapses to a single endpoint, and `seam_of` classifies only two of the four
seams (the other two are reconstructed). If the substrate survives all of that,
runtime-agnosticism is demonstrated, not asserted.

Targets any perception -> caption -> fuse -> decide loop that talks to an
OpenAI-compatible endpoint. No `core/` or `Adapter`-Protocol change is required.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from plumbline.adapters.base import (
    Action,
    ActionSchema,
    BusTap,
    ClockHook,
    ProxyConfig,
    derived_seam_event,
)
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent
from plumbline.proxy.normalizers import contains_image

# A neutral flat action vocabulary (distinct from OM1's typed move/skill/speak).
DEFAULT_ACTIONS: tuple[str, ...] = ("move_forward", "turn_left", "turn_right", "back_up", "stop")


class MalformedDecision(ValueError):
    """A decide response with no extractable action (missing content / error body)."""


@dataclass(frozen=True)
class GenericActionSchema:
    """A flat, runtime-neutral action plan (§9.1).

    Unlike OM1's typed `{"commands": [{"type": ...}]}`, a generic decision plan is
    either a single action token (`{"action": "turn_left"}`, the common
    free-text / Ollama path) or an OpenAI tool/function call. Every parsed Action
    uses `kind="act"` — demonstrating `Action.kind` is a free string, not OM1's
    vocabulary.
    """

    commands: tuple[str, ...] = DEFAULT_ACTIONS

    def parse(self, payload: Payload) -> tuple[Action, ...]:
        inline = payload.inline
        if not isinstance(inline, dict):
            return ()
        tool_calls = inline.get("tool_calls")
        if isinstance(tool_calls, list):
            actions: list[Action] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict):
                    actions.append(
                        Action(
                            "act",
                            _as_str(function.get("name")),
                            _tool_args(function.get("arguments")),
                        )
                    )
            return tuple(actions)
        action = inline.get("action")
        if isinstance(action, str):
            return (Action("act", action, {}),)
        return ()


@dataclass
class GenericAgentAdapter:
    """Wires any OpenAI-compatible perception->decide loop into the substrate (§9.1).

    Implements the `Adapter` Protocol. Differs from OM1 in every method that could
    be OM1-specific (see the module docstring): single-endpoint config, no bus,
    two-seam classification, derived action seam.
    """

    proxy_base_url: str
    # UNVERIFIED (per target runtime): which env var the runtime's OpenAI client
    # reads for its base URL, and whether it already appends `/v1`. Constructor
    # fields so a caller overrides without editing this file.
    base_url_env_var: str = "OPENAI_BASE_URL"
    append_v1: bool = True
    # Endpoint substrings that mark a TEXT perception (caption) call, to
    # disambiguate it from the (also text) decide call to the same endpoint.
    # UNVERIFIED heuristic: default is image -> caption, else -> decide.
    caption_markers: tuple[str, ...] = ()
    # Optional injected non-Zenoh action tap (mirrors OM1's session injection).
    # Default None: the loop runs with NO bus — the action seam is derived instead.
    action_tap: BusTap | None = field(default=None)

    def configure_proxy(self) -> ProxyConfig:
        base = self.proxy_base_url.rstrip("/")
        value = f"{base}/v1" if self.append_v1 else base
        # A runtime-neutral adapter knows one endpoint and no config-file schema.
        return ProxyConfig(
            proxy_base_url=base, env={self.base_url_env_var: value}, config_fields={}
        )

    def bus_tap(self) -> BusTap | None:
        return self.action_tap  # None by default: the loop works with no bus

    def seam_of(self, request: Payload, endpoint: str) -> Seam:
        """Classify a captured model call. Only two outcomes: perception vs decide.

        DECIDE_TO_ACT and CAPTION_TO_FUSE are never returned here — they are
        reconstructed (no live call), not classified. That 2-classified /
        2-reconstructed asymmetry is the concrete evidence the contract is general.
        """
        lowered = endpoint.lower()
        if contains_image(request.inline):
            return Seam.SENSOR_TO_CAPTION
        if any(marker.lower() in lowered for marker in self.caption_markers):
            return Seam.SENSOR_TO_CAPTION  # a text-only perception/caption call
        return Seam.FUSE_TO_DECIDE

    def action_schema(self) -> ActionSchema:
        return GenericActionSchema()

    def clock_hook(self) -> ClockHook | None:
        """No clock hook: model-I/O determinism only, NOT wall-clock scheduling
        (§3.4, §14.4)."""
        return None

    def reconstruct_caption_to_fuse(
        self,
        *,
        episode_id: str,
        seq: int,
        logical_tick: int,
        captions: Sequence[JSONValue],
        fused_prompt: JSONValue,
        wall_ts: float = 0.0,
    ) -> SeamEvent:
        """Derive CAPTION_TO_FUSE (no model call at this seam) — same mechanism as OM1,
        via the shared `derived_seam_event` helper (a helper inside adapters/, no
        cross-core-boundary refactor)."""
        return derived_seam_event(
            seam=Seam.CAPTION_TO_FUSE,
            episode_id=episode_id,
            seq=seq,
            logical_tick=logical_tick,
            request=Payload(inline={"captions": list(captions)}),
            response=Payload(inline={"fused_prompt": fused_prompt}),
            wall_ts=wall_ts,
        )

    def reconstruct_decide_to_act(
        self,
        *,
        episode_id: str,
        seq: int,
        logical_tick: int,
        decision_response: Payload,
        wall_ts: float = 0.0,
    ) -> SeamEvent:
        """Derive DECIDE_TO_ACT from the decision response — the no-bus answer.

        Called at record time. The action is a pure function of the recorded decide
        response, so on faithful replay the derived request is byte-identical. In a
        counterfactual captioner swap, a diverging decision halts at the first
        downstream served seam (CAPTION_TO_FUSE) — no stale derived action is served
        past a divergence (invariant 5, §6). Raises `MalformedDecision` if the decide
        response has no extractable action."""
        return derived_seam_event(
            seam=Seam.DECIDE_TO_ACT,
            episode_id=episode_id,
            seq=seq,
            logical_tick=logical_tick,
            request=Payload(inline={"action": _decided_action(decision_response)}),
            response=Payload(inline={"dispatched": True}),
            wall_ts=wall_ts,
        )


def _decided_action(response: Payload) -> str:
    """Extract the decided action from a chat-completion (or an already-normalized
    `{"action": ...}`) decide response. A genuinely empty decision (`content: ""`)
    returns `""`; a response with NO extractable action (missing content, an error
    body) raises `MalformedDecision` rather than silently minting `{"action": ""}`."""
    inline = response.inline
    if isinstance(inline, dict):
        choices = inline.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return str(message["content"]).strip()
        if isinstance(inline.get("action"), str):
            return str(inline["action"]).strip()
    raise MalformedDecision(f"no extractable action in decide response: {response.inline!r}")


def _tool_args(arguments: JSONValue) -> Mapping[str, JSONValue]:
    """OpenAI tool-call `arguments` is a JSON-encoded string on the wire; parse it
    (or accept a pre-parsed dict). Malformed/other -> empty args."""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return arguments if isinstance(arguments, dict) else {}


def _as_str(value: JSONValue) -> str:
    return value if isinstance(value, str) else ""
