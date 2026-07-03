"""The Unitree G1 humanoid adapter (engineering spec §9.3, §13.1) — cross-embodiment.

GROUNDED IN OM1's REAL SOURCE (main/70c23e2; config/unitree_g1_conversation.json5,
plugins/actions/unitree/g1/arm/zenoh.go, plugins/actions/emotion/zenoh.go,
plugins/actions/speak/elevenlabs_tts.go). The G1's real surface, which this file
replaces the earlier placeholder schema with:

- The G1 config has NO locomotion action at all — no walk/turn, no cmd_vel. Its
  physical output is a DISCRETE ARM/BODY GESTURE set (`unitree_g1_arm`, llm_label
  `robot_action`): 24 named gestures published as a CDR-LE Unitree sport-mode
  Request (api_id 9001, `{"action": "<gesture>"}` parameter) on the bare Zenoh key
  `api/sport/request`.
- `emotion` publishes via the avatar provider on `om/avatar/request`; `speak` is
  ElevenLabs REST TTS (no bus traffic).
- The Cortex emits TOOL CALLS (`speak` / `emotion` / `robot_action`), each carrying
  a single `{"action": "<string>"}` argument — the same wire family run-verified
  for the Go2 (docs/om1-integration.md); there is no `{"commands": [...]}` shape.
- The reference cortex is `GeminiLLM` — the same OpenAI-compat client whose
  `config.base_url` redirect the Go2 SIL episode run-verified.

Cross-embodiment thesis unchanged: the SAME transport shape (HTTP proxy + Zenoh
tap) carries a different embodiment with ZERO `core/`/`base.py` change — a
different action vocabulary absorbed by the frozen `Action`/`ActionSchema` types.
"""

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field

from plumbline.adapters.base import Action, ActionSchema, BusTap, ClockHook, ProxyConfig

# OM1-family shared wiring (single source of truth): G1 runs on the same OM1 Go
# runtime, so the proxy redirect and tool-call parsing are identical — reuse rather
# than fork a second guess (intra-package reuse, not a cross-boundary refactor).
from plumbline.adapters.om1 import _as_str, _family_proxy_config, _is_asr_endpoint, _tool_calls
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.proxy.normalizers import contains_image
from plumbline.transport.zenoh_tap import ZenohSession, ZenohTap

# plugins/actions/unitree/g1/arm/zenoh.go ArmAction.EnumValues (idle is a no-op).
G1_GESTURES: tuple[str, ...] = (
    "idle",
    "shake_hand",
    "face_wave",
    "hands_up",
    "stand_still",
    "show_hand",
    "come_closer",
    "flexible",
    "heart",
    "push",
    "rotate_hands",
    "salute",
    "shrug",
    "talking_2s",
    "talking_4s",
    "talking_6s",
    "talking_8s",
    "talking_10s",
    "talking_12s",
    "talking_14s",
    "talking_16s",
    "talking_18s",
    "talking_20s",
)
# plugins/actions/emotion (EmotionInput enums).
G1_EMOTIONS: tuple[str, ...] = ("happy", "confused", "curious", "excited", "sad", "think")

_SPORT_REQUEST_KEY = "api/sport/request"
_AVATAR_REQUEST_KEY = "om/avatar/request"

# unitree_api/Request as the arm connector serializes it (arm/zenoh.go
# serializeUnitreeRequest): CDR-LE header, identity.id int64, identity.api_id int64,
# lease.id int64, policy.priority uint32, policy.noreply u8, 3B pad, parameter
# length uint32 (incl. NUL), parameter bytes, pad to 4, binary seq length uint32.
_SPORT_HEADER = b"\x00\x01\x00\x00"
_SPORT_PARAM_LEN_OFFSET = 4 + 32  # header + identity/lease/policy block


def decode_sport_request(raw: bytes) -> JSONValue | None:
    """Decode a Unitree sport-mode Request from the G1 arm connector's wire bytes
    into a typed view for behavioral comparison. Returns None if not a sport
    request — the tap then falls back to its generic decode."""
    if len(raw) < _SPORT_PARAM_LEN_OFFSET + 4 or raw[:4] != _SPORT_HEADER:
        return None
    api_id = struct.unpack_from("<q", raw, 4 + 8)[0]
    (param_len,) = struct.unpack_from("<I", raw, _SPORT_PARAM_LEN_OFFSET)
    start = _SPORT_PARAM_LEN_OFFSET + 4
    if param_len == 0 or start + param_len > len(raw):
        return None
    parameter = raw[start : start + param_len - 1].decode("utf-8", "replace")  # strip NUL
    parsed: JSONValue
    try:
        parsed = json.loads(parameter)
    except ValueError:
        parsed = parameter
    return {"unitree_api/Request": {"api_id": api_id, "parameter": parsed}}


def _decode_bus_payload(key_expr: str, raw: bytes) -> JSONValue | None:
    if key_expr == _SPORT_REQUEST_KEY or key_expr.endswith("/" + _SPORT_REQUEST_KEY):
        return decode_sport_request(raw)
    return None  # avatar/emotion payloads fall through to the generic decode


@dataclass(frozen=True)
class G1ActionSchema:
    """Unitree G1 commands, typed for comparison (§9.3) — the REAL vocabulary.

    The Cortex emits tool calls `speak` / `emotion` / `robot_action`, each with an
    `{"action": "<value>"}` argument (arm/zenoh.go `args["action"]`, SpeakInput /
    EmotionInput / ArmInput all bind a single `action` field). There is no
    velocity locomotion on the G1 config; `robot_action` is a discrete gesture.
    """

    commands: tuple[str, ...] = ("speak", "emotion", "robot_action")

    def parse(self, payload: Payload) -> tuple[Action, ...]:
        actions: list[Action] = []
        for name, args in _tool_calls(payload.inline):
            value = _as_str(args.get("action"))
            if name == "robot_action":
                actions.append(Action("skill", value, {}))  # a named gesture
            elif name == "emotion":
                actions.append(Action("express", value, {}))
            elif name == "speak":
                actions.append(Action("speak", "speak", {"text": value}))
        return tuple(actions)


@dataclass
class G1Adapter:
    """Wires a Unitree G1 (OM1 runtime, humanoid) into the substrate (§9.3).

    Same shape as OM1Adapter, re-parameterized for the humanoid embodiment.
    """

    proxy_base_url: str
    # Same JSON5 redirect mechanism as OM1 (cortex_llm.config.base_url) — the
    # OpenAI-compat client behind GeminiLLM honors base_url (run-verified on Go2).
    config_base_url_paths: tuple[str, ...] = ("cortex_llm.config.base_url",)
    append_v1: bool = True
    # The G1's real action keys (source-verified, arm/zenoh.go + avatar provider):
    # gestures on api/sport/request, emotions on om/avatar/request. Bare keys, no
    # g1/ namespace. A real G1 episode is still pending to run-verify (WS5).
    action_key_expressions: tuple[str, ...] = (_SPORT_REQUEST_KEY, _AVATAR_REQUEST_KEY)
    zenoh_session: ZenohSession | None = field(default=None)

    def configure_proxy(self) -> ProxyConfig:
        """Redirect OM1's model calls at the proxy via config-field overrides
        (OM1-family mechanism, §9.3)."""
        return _family_proxy_config(
            self.proxy_base_url, self.config_base_url_paths, append_v1=self.append_v1
        )

    def bus_tap(self) -> BusTap | None:
        if self.zenoh_session is None:
            return None
        # Action keys only — data-bus telemetry would be mis-recorded as DECIDE_TO_ACT.
        return ZenohTap(
            session=self.zenoh_session,
            key_expressions=self.action_key_expressions,
            payload_decoder=_decode_bus_payload,
        )

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
        """Reconstruct CAPTION_TO_FUSE (no model call at this seam)."""
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
        """Derive the semantic DECIDE_TO_ACT from the Cortex tool-call decision —
        the `speak`/`emotion`/`robot_action` calls (§9.3). The physical gesture
        request on `api/sport/request` is the low-level execution captured by the
        tap. A pure function of the recorded response, so replay is byte-identical."""
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
        return any(
            endpoint == key or endpoint.endswith("/" + key) for key in self.action_key_expressions
        )
