"""The OM1 adapter (engineering spec §9.2) — the flagship reference integration.

Grounded in OM1's actual source (github.com/OpenMind/OM1, v1.0.0-beta.1) — see
docs/om1-integration.md for the verified facts and citations. Two mechanisms: the
recording HTTP proxy for the model seams (VLM/ASR and the Cortex LLM) and a Zenoh tap
for the action seam (the `cmd_vel` Twist).

CONFIRMED from source:
- Model endpoint is the JSON5 field `cortex_llm.config.base_url` (default: OpenMind
  portal via OM_API_KEY) — NOT per-provider base-URL env vars. Redirect = set that
  config field to the proxy (surfaced as ProxyConfig.config_fields).
- The Cortex LLM emits tool calls (internal/llm ToolCall). `Move` is a DISCRETE label
  {turn left, turn right, move forwards, move back, stand still} (go2 autonomy
  move.go), executed as a CDR geometry_msgs/Twist over Zenoh to `cmd_vel`.

STILL OPEN (needs a real recorded episode; kept UNVERIFIED inline): the exact
`cmd_vel` Zenoh key expression (namespace / ros2dds-bridge prefix), the exact
tool-call wire shape OM1 emits, and the default portal URL string.
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
)
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.proxy.normalizers import contains_image
from plumbline.transport.zenoh_tap import ZenohSession, ZenohTap


def _family_proxy_config(
    proxy_base_url: str, config_base_url_paths: Sequence[str], *, append_v1: bool
) -> ProxyConfig:
    """OM1-family (OM1/G1, same Go runtime) proxy redirect. OM1 configures model
    endpoints in JSON5 (`cortex_llm.config.base_url`), not base-URL env vars — so the
    redirect is a set of config-field overrides, and env is empty. Auth (OM_API_KEY /
    portal) is unchanged; the proxy carries the request through (§9.2)."""
    base = proxy_base_url.rstrip("/")
    value = f"{base}/v1" if append_v1 else base
    config_fields = {path: value for path in config_base_url_paths}
    return ProxyConfig(proxy_base_url=base, env={}, config_fields=config_fields)


@dataclass(frozen=True)
class OM1ActionSchema:
    """OM1's Cortex tool-call decisions typed for comparison (§9.2).

    OM1 emits function/tool calls; the reference Go2 config uses `Move` (a discrete
    label — the behavior compared for drift), `speak`, and `emotion`. Parses OpenAI
    (`tool_calls`) and Gemini (`functionCall`) response shapes, plus a bare
    reconstructed `{"action": ...}`. See docs/om1-integration.md.
    """

    commands: tuple[str, ...] = ("Move", "speak", "emotion")

    def parse(self, payload: Payload) -> tuple[Action, ...]:
        actions: list[Action] = []
        for name, args in _tool_calls(payload.inline):
            if name == "Move":
                actions.append(Action("move", _as_str(args.get("action")), {}))
            elif name == "speak":
                actions.append(Action("speak", "speak", {"text": args.get("text")}))
            elif name == "emotion":
                actions.append(
                    Action("emotion", _as_str(args.get("emotion") or args.get("action")), {})
                )
            else:
                actions.append(Action(name.lower(), name, dict(args)))
        return tuple(actions)


@dataclass
class OM1Adapter:
    """Wires OM1 into the substrate (§9.2). Implements the `Adapter` protocol."""

    proxy_base_url: str
    # The JSON5 config path(s) whose value OM1 uses as the (OpenAI-compatible) model
    # endpoint. Confirmed: cortex_llm.config.base_url. Add OpenAI-compatible VLM/ASR
    # input base_url paths here if a deployment uses them.
    config_base_url_paths: tuple[str, ...] = ("cortex_llm.config.base_url",)
    append_v1: bool = True
    # UNVERIFIED (needs a real episode): the Go2 autonomy `Move` connector publishes a
    # CDR Twist over Zenoh to the `cmd_vel` topic (config cmd_vel_topic: "cmd_vel");
    # the fully-qualified key (namespace / ros2dds-bridge prefix) is not confirmed.
    action_key_expressions: tuple[str, ...] = ("cmd_vel", "**/cmd_vel")
    # Sensor/data-bus keys (reserved; not tapped as actions). Still open.
    data_bus_key_expressions: tuple[str, ...] = ("**/data_bus/**",)
    # Injected so the tap is testable and the substrate carries no `zenoh` dep;
    # None means no bus tap is available (record model seams only).
    zenoh_session: ZenohSession | None = field(default=None)

    def configure_proxy(self) -> ProxyConfig:
        """Redirect OM1's model calls at the proxy via config-field overrides —
        env/config only, zero OM1 source changes (§9.2)."""
        return _family_proxy_config(
            self.proxy_base_url, self.config_base_url_paths, append_v1=self.append_v1
        )

    def bus_tap(self) -> BusTap | None:
        if self.zenoh_session is None:
            return None
        # Action keys only — data-bus telemetry would be mis-recorded as DECIDE_TO_ACT.
        return ZenohTap(session=self.zenoh_session, key_expressions=self.action_key_expressions)

    def seam_of(self, request: Payload, endpoint: str) -> Seam:
        """Classify a captured call into a seam (§9.2).

        CAPTION_TO_FUSE is never returned here: it has no model call of its own and is
        reconstructed within a tick (see `reconstruct_caption_to_fuse`).
        """
        lowered = endpoint.lower()
        if self._is_action_endpoint(lowered):
            return Seam.DECIDE_TO_ACT  # the cmd_vel Twist on Zenoh
        if _is_asr_endpoint(lowered):
            return Seam.SENSOR_TO_CAPTION  # ASR is a sensor->caption call
        if contains_image(request.inline):
            return Seam.SENSOR_TO_CAPTION  # VLM caption call
        return Seam.FUSE_TO_DECIDE  # the Cortex chat completion (fused prompt -> tool calls)

    def action_schema(self) -> ActionSchema:
        return OM1ActionSchema()

    def clock_hook(self) -> ClockHook | None:
        """No clock hook for now (§9.2, §14.4).

        Plumbline therefore guarantees deterministic model I/O on replay, NOT
        deterministic wall-clock scheduling (§3.4). If OM1 later exposes a hook on its
        hertz loop, this returns it to upgrade to full scheduler determinism.
        """
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
        """Reconstruct CAPTION_TO_FUSE by associating the tick's captions (VLM
        responses) with the subsequent fused prompt (Cortex request) (§9.2). No model
        call at this seam, so the event is derived from already-captured payloads."""
        request = Payload(inline={"captions": list(captions)})
        response = Payload(inline={"fused_prompt": fused_prompt})
        return SeamEvent(
            episode_id=episode_id,
            seq=seq,
            seam=Seam.CAPTION_TO_FUSE,
            logical_tick=logical_tick,
            wall_ts=wall_ts,
            request=request,
            response=response,
            model_id=None,
            params={},
            request_digest=canonicalize(request).digest,
            latency_ms=0.0,
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
        """Derive the semantic DECIDE_TO_ACT from the Cortex tool-call decision — the
        `Move`/`speak`/`emotion` calls (§9.2). This is the JSON-comparable action; the
        physical cmd_vel Twist on Zenoh is the low-level execution captured by the tap.
        The action is a pure function of the recorded decide response, so faithful
        replay is byte-identical."""
        calls = _tool_calls(decision_response.inline)
        actions: list[JSONValue] = [{"function": name, "args": dict(args)} for name, args in calls]
        request = Payload(inline={"actions": actions})
        response = Payload(inline={"dispatched": True})
        return SeamEvent(
            episode_id=episode_id,
            seq=seq,
            seam=Seam.DECIDE_TO_ACT,
            logical_tick=logical_tick,
            wall_ts=wall_ts,
            request=request,
            response=response,
            model_id=None,
            params={},
            request_digest=canonicalize(request).digest,
            latency_ms=0.0,
        )

    def _is_action_endpoint(self, endpoint: str) -> bool:
        if endpoint.startswith("zenoh:") or "/action" in endpoint:
            return True
        return any(_key_base(key_expr) in endpoint for key_expr in self.action_key_expressions)


def _tool_calls(inline: JSONValue) -> list[tuple[str, Mapping[str, JSONValue]]]:
    """Extract (function_name, args) pairs from an OM1 Cortex decision — OpenAI
    `tool_calls`, Gemini `functionCall`, or a bare reconstructed `{"actions": [...]}`
    / `{"action": ...}`. Tolerant: unknown shapes yield no calls."""
    if not isinstance(inline, dict):
        return []
    calls: list[tuple[str, Mapping[str, JSONValue]]] = []
    # Reconstructed shapes (round-trip of reconstruct_decide_to_act / a bare action).
    reconstructed = inline.get("actions")
    if isinstance(reconstructed, list):
        for item in reconstructed:
            if not isinstance(item, dict):
                continue
            name = item.get("function")
            if isinstance(name, str):
                args = item.get("args")
                calls.append((name, args if isinstance(args, dict) else {}))
        return calls
    action = inline.get("action")
    if isinstance(action, str):
        return [("Move", {"action": action})]
    # OpenAI tool_calls (top-level or under choices[].message).
    raw = inline.get("tool_calls")
    if raw is None:
        for choice in _as_list(inline.get("choices")):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
                raw = message["tool_calls"]
                break
    for call in _as_list(raw):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        fn = function if isinstance(function, dict) else call
        name = fn.get("name")
        if isinstance(name, str):
            calls.append((name, _tool_args(fn.get("arguments"))))
    if calls:
        return calls
    # Gemini functionCall parts.
    for candidate in _as_list(inline.get("candidates")):
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in _as_list(parts):
            fc = part.get("functionCall") if isinstance(part, dict) else None
            if isinstance(fc, dict):
                name = fc.get("name")
                args = fc.get("args")
                if isinstance(name, str):
                    calls.append((name, args if isinstance(args, dict) else {}))
    return calls


def _tool_args(arguments: JSONValue) -> Mapping[str, JSONValue]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_list(value: JSONValue) -> list[JSONValue]:
    return value if isinstance(value, list) else []


def _key_base(key_expr: str) -> str:
    """A matchable token from a Zenoh key expression: strip globs and take the last
    non-glob segment (`cmd_vel` -> `cmd_vel`, `**/cmd_vel` -> `cmd_vel`)."""
    cleaned = key_expr.replace("**", "").replace("*", "").strip("/")
    return cleaned.split("/")[-1] if cleaned else key_expr


def _is_asr_endpoint(endpoint: str) -> bool:
    return any(token in endpoint for token in ("transcription", "/audio", "/asr", "whisper"))


def _as_str(value: JSONValue) -> str:
    return value if isinstance(value, str) else ""
