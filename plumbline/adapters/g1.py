"""The Unitree G1 humanoid adapter (engineering spec §9.3, §13.1) — cross-embodiment.

The OM1-family cross-embodiment case: the SAME two mechanisms as OM1 (recording
HTTP proxy for the model seams + a Zenoh tap for the action seam) and the same
`seam_of` structure, re-parameterized for a BIPEDAL HUMANOID — a different action
vocabulary (walk/turn/gesture/speak/pose) and different bus keys (`g1/agent/...`).

It is deliberately NOT a new contract shape: the generic adapter already proved the
contract needs no bus; G1's complementary proof is that the SAME transport shape
carries a different embodiment with ZERO `core/`/`base.py` change — the frozen
`Action.kind` (a free string) and `ActionSchema.commands` absorb the new vocabulary.
The numeric `walk(vx, vy, vyaw)` velocities are the natural first consumer of
`ActionSchemaMatcher` (numeric-tolerant behavioral drift).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from plumbline.adapters.base import Action, ActionSchema, BusTap, ClockHook, ProxyConfig

# OM1-family shared wiring (single source of truth): G1 runs on the same OM1
# runtime, so its provider conventions are identical — import rather than fork a
# second (already UNVERIFIED) guess. This is intra-package reuse of stable
# constants, not a cross-boundary refactor of om1.py (invariant 6).
from plumbline.adapters.om1 import _OPENAI_COMPATIBLE, _PROVIDER_ENV, _as_str, _is_asr_endpoint
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.proxy.normalizers import contains_image
from plumbline.transport.zenoh_tap import ZenohSession, ZenohTap


@dataclass(frozen=True)
class G1ActionSchema:
    """Unitree G1 bipedal humanoid commands, typed for comparison (§9.3).

    walk(vx, vy, vyaw) velocity locomotion, turn(vyaw), named gestures ("wave"),
    speech, and named whole-body poses ("bow") — replacing OM1's quadruped
    move/skill/speak/express vocabulary. `Action.kind` carries the humanoid kinds,
    which the frozen dataclass accepts unchanged.
    """

    commands: tuple[str, ...] = ("walk", "turn", "gesture", "speak", "pose")

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
            if kind == "walk":
                actions.append(
                    Action(
                        "walk",
                        "walk",
                        {k: command[k] for k in ("vx", "vy", "vyaw") if k in command},
                    )
                )
            elif kind == "turn":
                actions.append(
                    Action("turn", "turn", {k: command[k] for k in ("vyaw",) if k in command})
                )
            elif kind == "gesture":
                actions.append(Action("gesture", _as_str(command.get("name")), {}))
            elif kind == "speak":
                actions.append(Action("speak", "speak", {"text": command.get("text")}))
            elif kind == "pose":
                actions.append(Action("pose", _as_str(command.get("name")), {}))
        return tuple(actions)


@dataclass
class G1Adapter:
    """Wires a Unitree G1 (OM1 runtime, humanoid HAL) into the substrate (§9.3).

    Same shape as OM1Adapter, re-parameterized for the humanoid embodiment.
    """

    proxy_base_url: str
    providers: tuple[str, ...] = ("openai", "anthropic", "gemini", "deepseek", "xai", "ollama")
    # UNVERIFIED (CLAUDE.md medium-leash / WS5): G1's Zenoh key expressions are
    # placeholders, NOT confirmed against a real G1 build. Grep the real G1 HAL for
    # its declare_publisher/subscriber keys. Constructor fields so a caller overrides.
    action_key_expressions: tuple[str, ...] = ("g1/agent/actions/**",)
    data_bus_key_expressions: tuple[str, ...] = ("g1/agent/data_bus/**",)  # reserved; not tapped
    zenoh_session: ZenohSession | None = field(default=None)

    def configure_proxy(self) -> ProxyConfig:
        """Point each provider's base URL at the proxy (OM1-family wiring, §9.3)."""
        base = self.proxy_base_url.rstrip("/")
        env: dict[str, str] = {}
        config_fields: dict[str, str] = {}
        for provider in self.providers:
            value = f"{base}/v1" if provider in _OPENAI_COMPATIBLE else base
            for var in _PROVIDER_ENV.get(provider, ()):
                env[var] = value
            # UNVERIFIED: the G1 config-file layout may differ from OM1's config/*.json5.
            config_fields[f"{provider}.base_url"] = value
        return ProxyConfig(proxy_base_url=base, env=env, config_fields=config_fields)

    def bus_tap(self) -> BusTap | None:
        if self.zenoh_session is None:
            return None
        # Action keys only — data-bus telemetry would be mis-recorded as DECIDE_TO_ACT.
        return ZenohTap(session=self.zenoh_session, key_expressions=self.action_key_expressions)

    def seam_of(self, request: Payload, endpoint: str) -> Seam:
        """Classify a captured call into a seam (§9.3). CAPTION_TO_FUSE is never
        returned here — it is reconstructed within a tick."""
        lowered = endpoint.lower()
        if self._is_action_endpoint(lowered):
            return Seam.DECIDE_TO_ACT
        if _is_asr_endpoint(lowered):
            return Seam.SENSOR_TO_CAPTION
        if contains_image(request.inline):
            return Seam.SENSOR_TO_CAPTION
        return Seam.FUSE_TO_DECIDE

    def action_schema(self) -> ActionSchema:
        return G1ActionSchema()

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
        """Reconstruct CAPTION_TO_FUSE (no model call at this seam). Duplicated
        per-adapter (invariant 6: no cross-boundary refactor for this slice)."""
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
        return any(
            endpoint.startswith(
                (key_expr[:-2] if key_expr.endswith("**") else key_expr).rstrip("/")
            )
            for key_expr in self.action_key_expressions
        )
