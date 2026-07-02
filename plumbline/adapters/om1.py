"""The OM1 adapter (engineering spec §9.2) — the flagship reference integration.

Targets OM1's documented language-bus contract and provider interface, not
fragile internal structs, so it survives OM1's beta churn. Two mechanisms:
the recording HTTP proxy for the model seams (vision/ASR and the Cortex LLM) and
a Zenoh tap for the action seam.
"""

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

# Providers whose OpenAI-compatible endpoints expect a `/v1` base-URL suffix.
# UNVERIFIED (like the keys below): which providers OM1 treats as OpenAI-compatible
# (and thus need `/v1`) is inferred, not confirmed against a real OM1 build.
_OPENAI_COMPATIBLE = frozenset({"openai", "deepseek", "xai"})

# UNVERIFIED (CLAUDE.md medium-leash / WS5): these base-URL env-var names are
# inferred from each provider SDK's conventions, NOT confirmed against a real OM1
# build. Verify against the actual OM1 config/provider clients before relying on
# the zero-touch redirect; correct here if OM1 routes all LLM calls through a
# single OpenMind portal endpoint instead.
_PROVIDER_ENV: Mapping[str, tuple[str, ...]] = {
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "anthropic": ("ANTHROPIC_BASE_URL",),
    "gemini": ("GEMINI_API_BASE", "GOOGLE_GEMINI_BASE_URL"),
    "deepseek": ("DEEPSEEK_BASE_URL",),
    "xai": ("XAI_BASE_URL",),
    "ollama": ("OLLAMA_HOST",),
}


@dataclass(frozen=True)
class OM1ActionSchema:
    """OM1's elemental commands typed for structural comparison (§9.2).

    move(x, y, yaw), named skills ("shake paw"), speech acts, and expressions.
    """

    commands: tuple[str, ...] = ("move", "skill", "speak", "express")

    def parse(self, payload: Payload) -> tuple[Action, ...]:
        inline = payload.inline
        raw_commands = inline.get("commands") if isinstance(inline, dict) else None
        if not isinstance(raw_commands, list):
            return ()
        actions: list[Action] = []
        for command in raw_commands:
            if not isinstance(command, dict):
                continue
            kind = command.get("type")
            if kind == "move":
                actions.append(
                    Action(
                        "move", "move", {k: command[k] for k in ("x", "y", "yaw") if k in command}
                    )
                )
            elif kind == "skill":
                actions.append(Action("skill", _as_str(command.get("name")), {}))
            elif kind == "speak":
                actions.append(Action("speak", "speak", {"text": command.get("text")}))
            elif kind == "express":
                actions.append(Action("express", _as_str(command.get("name")), {}))
        return tuple(actions)


@dataclass
class OM1Adapter:
    """Wires OM1 into the substrate (§9.2). Implements the `Adapter` protocol."""

    proxy_base_url: str
    providers: tuple[str, ...] = ("openai", "anthropic", "gemini", "deepseek", "xai", "ollama")
    # UNVERIFIED (CLAUDE.md medium-leash / WS5): these Zenoh key expressions are
    # placeholders, NOT confirmed against a real OM1 build. Grep the actual OM1
    # source for its declare_publisher/subscriber keys and correct them before
    # recording a real episode. They are constructor fields so a caller can
    # override without editing this file.
    action_key_expressions: tuple[str, ...] = ("om1/agent/actions/**",)
    data_bus_key_expressions: tuple[str, ...] = ("om1/agent/data_bus/**",)
    # Injected so the tap is testable and the substrate carries no `zenoh` dep;
    # None means no bus tap is available (record model seams only).
    zenoh_session: ZenohSession | None = field(default=None)

    def configure_proxy(self) -> ProxyConfig:
        """Point each configured provider's base URL at the proxy — env/config
        only, zero OM1 source changes (§9.2)."""
        base = self.proxy_base_url.rstrip("/")
        env: dict[str, str] = {}
        config_fields: dict[str, str] = {}
        for provider in self.providers:
            value = f"{base}/v1" if provider in _OPENAI_COMPATIBLE else base
            for var in _PROVIDER_ENV.get(provider, ()):
                env[var] = value
            # UNVERIFIED: `<provider>.base_url` is an assumed OM1 config-field path;
            # confirm the real field names against OM1's config/*.json5 schema.
            config_fields[f"{provider}.base_url"] = value
        return ProxyConfig(proxy_base_url=base, env=env, config_fields=config_fields)

    def bus_tap(self) -> BusTap | None:
        if self.zenoh_session is None:
            return None
        return ZenohTap(
            session=self.zenoh_session,
            key_expressions=self.action_key_expressions + self.data_bus_key_expressions,
        )

    def seam_of(self, request: Payload, endpoint: str) -> Seam:
        """Classify a captured call into a seam (§9.2).

        CAPTION_TO_FUSE is never returned here: it has no model call of its own
        and is reconstructed from the captions and the subsequent fused prompt
        within a tick (see `reconstruct_caption_to_fuse`).
        """
        lowered = endpoint.lower()
        if self._is_action_endpoint(lowered):
            return Seam.DECIDE_TO_ACT
        if _is_asr_endpoint(lowered):
            return Seam.SENSOR_TO_CAPTION  # ASR is a sensor->caption call
        if contains_image(request.inline):
            return Seam.SENSOR_TO_CAPTION  # vision caption call
        return Seam.FUSE_TO_DECIDE  # the Cortex chat completion (fused prompt -> decision)

    def action_schema(self) -> ActionSchema:
        return OM1ActionSchema()

    def clock_hook(self) -> ClockHook | None:
        """No clock hook for now (§9.2, §14.4).

        Plumbline therefore guarantees deterministic model I/O on replay, NOT
        deterministic wall-clock scheduling (§3.4). If OM1 later exposes a hook on
        its hertz loop, this returns it to upgrade to full scheduler determinism.
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
        """Reconstruct the CAPTION_TO_FUSE seam by associating the tick's captions
        (VLM responses) with the subsequent fused prompt (Cortex request) (§9.2).

        There is no separate model call at this seam, so the event is derived from
        already-captured payloads rather than intercepted.
        """
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

    def _is_action_endpoint(self, endpoint: str) -> bool:
        if endpoint.startswith("zenoh:") or "/cmd_vel" in endpoint or "/action" in endpoint:
            return True
        for key_expr in self.action_key_expressions:
            prefix = key_expr[:-2] if key_expr.endswith("**") else key_expr
            if endpoint.startswith(prefix.rstrip("/")):
                return True
        return False


def _is_asr_endpoint(endpoint: str) -> bool:
    return any(token in endpoint for token in ("transcription", "/audio", "/asr", "whisper"))


def _as_str(value: JSONValue) -> str:
    return value if isinstance(value, str) else ""
