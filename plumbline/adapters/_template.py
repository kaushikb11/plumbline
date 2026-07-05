"""Copy-paste skeleton for a new runtime adapter (engineering spec §9.1).

Start here to integrate a runtime Plumbline does not yet know. Copy this file to
`plumbline/adapters/<your_runtime>.py`, rename the two classes, and work through the
`TODO` markers. Derived from `generic.py` (the honest, non-OM1 template): a single
proxy endpoint, NO bus, two classified seams, and two reconstructed seams — the
minimum a runtime-agnostic adapter must supply.

As written it is a *trivially valid* adapter: it imports cleanly and
`plumbline.adapters.conformance.assert_conforms(TemplateAdapter("http://localhost"))`
passes. Every `TODO` marks a spot where a real runtime needs real wiring; until you
fill them the adapter is structurally conformant but semantically a stub.

Verify as you go::

    from plumbline.adapters.conformance import assert_conforms
    assert_conforms(TemplateAdapter("http://localhost:8080"))
"""

from collections.abc import Sequence
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

# TODO: replace with your runtime's real action vocabulary — the command names the
# decision model can emit. Used only for typed behavioral comparison (drift), so it
# does not have to be exhaustive to start.
TEMPLATE_ACTIONS: tuple[str, ...] = ("noop",)


@dataclass(frozen=True)
class TemplateActionSchema:
    """Parse your runtime's decision response into typed `Action`s (§9.1).

    `Action.kind` is a free string; pick a vocabulary that makes drift comparisons
    meaningful for your embodiment (OM1 uses move/speak/emotion; the generic adapter
    uses a flat "act"). Must tolerate empty/unknown shapes and yield no actions.
    """

    commands: tuple[str, ...] = TEMPLATE_ACTIONS

    def parse(self, payload: Payload) -> tuple[Action, ...]:
        inline = payload.inline
        if not isinstance(inline, dict):
            return ()
        # TODO: extract the runtime's real action(s). This handles only the simplest
        # `{"action": "<name>"}` shape — replace with your tool-call / plan parsing.
        action = inline.get("action")
        if isinstance(action, str):
            return (Action("act", action, {}),)
        return ()


@dataclass
class TemplateAdapter:
    """Skeleton `Adapter`: fill the TODOs for your runtime (§9.1).

    Implements all seven contract methods. Defaults are chosen so `assert_conforms`
    passes immediately; the TODOs mark where real runtime wiring must replace them.
    """

    proxy_base_url: str
    # TODO: the env var (or config field) your runtime reads for its model endpoint.
    base_url_env_var: str = "OPENAI_BASE_URL"
    append_v1: bool = True
    # Endpoint substrings that mark a perception/caption call vs the decide call.
    # TODO: fill with markers specific to your runtime's caption endpoint(s).
    caption_markers: tuple[str, ...] = ()
    # TODO: if your runtime exposes a passive action bus, construct and return a
    # BusTap from bus_tap(); leave None to derive the action seam instead.
    action_tap: BusTap | None = field(default=None)

    def configure_proxy(self) -> ProxyConfig:
        """Point the runtime's model client at the proxy — env/config only, no source
        edits. TODO: adjust env/config_fields to how your runtime reads its endpoint."""
        base = self.proxy_base_url.rstrip("/")
        value = f"{base}/v1" if self.append_v1 else base
        return ProxyConfig(
            proxy_base_url=base, env={self.base_url_env_var: value}, config_fields={}
        )

    def bus_tap(self) -> BusTap | None:
        """Return a passive action tap, or None to run with no bus (the action seam is
        then reconstructed from the decision response). TODO."""
        return self.action_tap

    def seam_of(self, request: Payload, endpoint: str) -> Seam:
        """Classify a captured model call. Return only SENSOR_TO_CAPTION or
        FUSE_TO_DECIDE here — the other two seams are reconstructed, not classified.
        Must never raise. TODO: replace with your runtime's real classification."""
        lowered = endpoint.lower()
        if any(marker.lower() in lowered for marker in self.caption_markers):
            return Seam.SENSOR_TO_CAPTION
        # TODO: detect image/audio perception calls too (see generic.py's use of
        # plumbline.proxy.normalizers.contains_image).
        return Seam.FUSE_TO_DECIDE

    def action_schema(self) -> ActionSchema:
        return TemplateActionSchema()

    def clock_hook(self) -> ClockHook | None:
        """Return None unless your runtime exposes a loop-clock hook. Without one,
        Plumbline guarantees model-I/O determinism, NOT wall-clock scheduling
        (§3.4, §14.4). TODO if applicable."""
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
        """Derive CAPTION_TO_FUSE (no model call at this seam) from a tick's captions
        and the fused prompt. TODO: shape request/response to your runtime's fusion."""
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
        """Derive DECIDE_TO_ACT from the decision response — a pure function of it, so
        faithful replay stays byte-identical. TODO: extract the real dispatched action
        instead of echoing the raw response."""
        return derived_seam_event(
            seam=Seam.DECIDE_TO_ACT,
            episode_id=episode_id,
            seq=seq,
            logical_tick=logical_tick,
            request=Payload(inline={"decision": decision_response.inline}),
            response=Payload(inline={"dispatched": True}),
            wall_ts=wall_ts,
        )
