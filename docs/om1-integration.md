# OM1 integration — verified interface notes

Ground truth for the OM1 adapter, established by reading OM1's actual source and docs
(the reference runtime, [github.com/OpenMind/OM1](https://github.com/OpenMind/OM1),
`v1.0.0-beta.1`). This replaces the earlier `UNVERIFIED`-flagged guesses. Each fact
below cites its source. Items still genuinely open (needing a real recorded episode)
are called out explicitly — those keep an `UNVERIFIED` flag in code.

## Architecture

OM1 migrated to a **Go** single-binary runtime (the `python` branch is deprecated).
Internal packages map cleanly onto Plumbline's four seams:

| OM1 package | Plumbline seam |
|---|---|
| `internal/providers/vlm`, ASR inputs | `SENSOR_TO_CAPTION` |
| `internal/fuser` | `CAPTION_TO_FUSE` (reconstructed) |
| `internal/llm` (`LLM`, `Response`, `ToolCall`) | `FUSE_TO_DECIDE` |
| `internal/actions`, `plugins/actions/**`, `internal/zenoh` | `DECIDE_TO_ACT` |

Source: `internal/` and `plugins/actions/` directory listings (GitHub contents API).
OM1 also ships its own `grafana/` + `prometheus.yml` + `internal/providers/tracer.go`
— i.e. the exact latency/tracer baselines Experiment B contrasts against.

## Model endpoints & config

- Config is **JSON5** in `config/` (e.g. `config/unitree_go2_autonomy.json5`). Values
  support `${ENV_VAR:-default}` substitution.
- The LLM endpoint is **`cortex_llm.config.base_url`** (optional). When empty it
  defaults to the **OpenMind portal**; auth is `api_key: "${OM_API_KEY:-openmind_free}"`.
  So OM1 routes model calls through a configured base URL, **not** per-provider
  base-URL environment variables. *(This corrects the adapter's earlier
  `<provider>.base_url` / per-provider-env-var guess.)*
- Providers OM1 supports: OpenAI, xAI, DeepSeek, Anthropic, Meta, Gemini, NearAI,
  Ollama. The reference Go2 autonomy config uses `cortex_llm.type: "GeminiLLM"`.
- VLM/ASR are their own input types with their own `config` (e.g. `VLMGeminiRTSP`,
  `GoogleASRInput{api_version, rate, chunk}`, `VLM_COCO_Local{camera_index}`); not
  all expose a `base_url` (some are local or provider-managed).

Source: `config/unitree_go2_autonomy.json5`, docs "Configuration".

**Zero-touch redirect:** point OM1 at the recording proxy by setting
`cortex_llm.config.base_url` (and any OpenAI-compatible input's `config.base_url`) to
the proxy in the JSON5 — the adapter surfaces these as `ProxyConfig.config_fields`.
Pure env-var redirection is not OM1's mechanism.

## Actions

The LLM emits **function/tool calls** (OM1 has an `internal/llm` `ToolCall` type). Each
`agent_actions` entry maps an `llm_label` to a `connector`:

```json5
{ name: "unitree_go2_autonomy", llm_label: "Move", connector: "move",
  config: { cmd_vel_topic: "cmd_vel" } }
{ name: "speak",   llm_label: "speak",   connector: "elevenlabs_tts", ... }
{ name: "emotion", llm_label: "emotion", connector: "zenoh" }
```

The `Move` action is a **discrete label**, not continuous x/y/yaw
(`plugins/actions/unitree/go2/autonomy/move.go`):

```go
type MoveInput struct { Action MoveAction `json:"action"` }
func (MoveAction) EnumValues() []string {
    return []string{"turn left","turn right","move forwards","move back","stand still"}
}
```

The `move` connector translates a label into a `geometry_msgs/Twist` and publishes it
**CDR-serialized over Zenoh** to the `cmd_vel` topic
(`plugins/actions/unitree/go2/autonomy/cmd_vel.go`: `serializeTwist()` with the
`0x00 0x01 0x00 0x00` CDR header; imports `internal/zenoh`). *(This corrects the
adapter's invented `{"commands":[{"type":"move","x":...}]}` schema.)*

So two authoritative interception points, both real:
- **`FUSE_TO_DECIDE` response** — the Cortex tool calls (`Move`/`speak`/`emotion`),
  captured by the HTTP proxy. The semantic decision; JSON; ideal for drift scoring.
- **`DECIDE_TO_ACT`** — the CDR `Twist` on the Zenoh `cmd_vel` key, captured by the
  Zenoh tap. The physical execution; binary.

The adapter parses the tool-call decision (`OM1ActionSchema`) and reconstructs
`DECIDE_TO_ACT` from it (semantic, comparable), mirroring the generic adapter.

## Gazebo run

```bash
# 1. sim
source install/setup.bash && ros2 launch go2_gazebo_sim go2_launch.py
# 2. Zenoh <-> ROS2 DDS bridge
zenoh-bridge-ros2dds -c ./zenoh/zenoh_bridge_config.json5
# 3. OM1 agent (Go2 autonomy, sim mode)
CONFIG=unitree_go2_autonomy USE_SIM=true make dev
```

Source: docs "Gazebo". See [record-om1-gazebo.md](record-om1-gazebo.md) for wiring the
proxy + tap into this flow.

## Pinned by a real recorded episode (SIL run `om1-sil-001`)

The three previously-open items were pinned by a **software-in-the-loop episode**:
the real OM1 Go binary (built from `main`, commit `70c23e2`) with `agent_inputs: []`,
its Cortex pointed through the Plumbline recording proxy at a real cloud LLM, its
`Move` action publishing over real Zenoh, and Plumbline's tap as the only subscriber
plus a stubbed-HAL odom/paths publisher (`examples/record_om1_sil.py`). 36 Cortex
tool-call decisions and 1,470 `cmd_vel` CDR `Twist` frames captured; faithful replay
byte-identical over all 1,542 events; the action sequence recovered via
`OM1ActionSchema` matches OM1's own log (36× "move forwards").

1. **Zenoh `cmd_vel` key: exactly `cmd_vel`** — the configured `cmd_vel_topic`,
   verbatim, no namespace or prefix (`move.go` passes it straight to
   `DeclarePublisher`; the tap captured frames on the bare key). An `rt/`-style
   prefix applies only through the ros2dds bridge, so the adapter keeps
   `**/cmd_vel` as a secondary default. Recorded bus events now carry the
   originating key in `params["plumbline.bus_key"]`.
2. **Tool-call wire format via an OpenAI-compatible `base_url`: the OpenAI
   `tool_calls` array** — `{id, type: "function", function: {name: "Move",
   arguments: "{\"action\": \"move forwards\"}"}}`, captured verbatim at the
   FUSE_TO_DECIDE seam. All of OM1's LLM plugins share one OpenAI-compat client
   (`plugins/llm/openai_compat.go`); the Gemini `functionCall` branch in
   `OM1ActionSchema` stays for portal-native Gemini responses. Note `GeminiLLM`
   sets `tool_choice: "required"` while `OpenAILLM` uses `"auto"` — with small
   models, `"required"` is what guarantees a decision every tick.
3. **Portal base URLs (from source):** `OpenAILLM` defaults to
   `https://api.openmind.com/api/core/openai`, `GeminiLLM` to
   `https://api.openmind.com/api/core/gemini`; the cloud sim Zenoh broker is
   `wss://api.openmind.com/api/core/simulation/zenoh`. VLM/ASR inputs remain
   provider-managed (WS/RTSP), not `base_url`-overridable — the WS caption path is
   covered by the WS proxy (limitations gap #1).

## Pinned by the real Gazebo episode (Tier 3, `om1-gazebo-004`)

The full closed loop ran headlessly on Modal (`modal/gazebo_om1.py`): Gazebo
physics (go2_sim + champ from OpenMind/OM1-sim) + `zenoh-bridge-ros2dds` + the
real OM1 binary + a live cloud Cortex through the recording proxy. 90 decisions,
2,407 real `cmd_vel` CDR `Twist` frames tapped, **the simulated Go2 walked
3.455 m**, faithful replay byte-identical over all 2,587 events.

4. **Bridged key naming: bare topic names, no `rt/` prefix.** With
   `zenoh-bridge-ros2dds` defaults, ROS2 `/odom` ↔ zenoh key `odom`, `/cmd_vel`
   ↔ `cmd_vel` — OM1's bare-key subscriptions work against the bridge as-is.
   (The Tier-3 harness adds a `namespace: "/sim"` to the bridge deliberately, so
   its sim-gap shim can sit between OM1 and the raw sim topics.)
5. **Two sim gaps found (upstream-relevant, shimmmed zero-touch in the harness):**
   champ's odometry is planar (`pose.z = 0`) while OM1's Move connector requires
   the body height the real Go2 firmware reports there (> 0.24 m = "standing")
   before it will drive — the harness relays `sim/odom → odom` with z lifted to
   0.30 m; and OM1-sim's `om_path` publishes range-keyed `/om/paths/r{K}` while
   OM1 subscribes the bare `om/paths` — the harness publishes an all-clear
   `om/paths` stub. Also upstream: OM1's `docs/simulators/gazebo.md` names a
   stale package (`go2_gazebo_sim`; the real one is `go2_sim`), and
   `USE_SIM=true` routes `cmd_vel` through OpenMind's cloud broker with no URL
   override (all-local recording must keep it unset).
