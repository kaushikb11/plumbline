# OM1 integration — verified interface notes (WS5)

Ground truth for the OM1 adapter, established by reading OM1's actual source and docs
(the reference runtime, [github.com/OpenMind/OM1](https://github.com/OpenMind/OM1),
`v1.0.0-beta.1`). This replaces the earlier `UNVERIFIED`-flagged guesses. Each fact
below cites its source. Items still genuinely open (needing a real recorded episode)
are called out explicitly — those keep an `UNVERIFIED` flag in code.

## Architecture (confirmed)

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

## Model endpoints & config (confirmed)

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

**Zero-touch redirect (corrected):** point OM1 at the recording proxy by setting
`cortex_llm.config.base_url` (and any OpenAI-compatible input's `config.base_url`) to
the proxy in the JSON5 — the adapter surfaces these as `ProxyConfig.config_fields`.
Pure env-var redirection is not OM1's mechanism.

## Actions (confirmed)

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

## Gazebo run (confirmed)

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

## Still open (needs a real recorded episode — kept `UNVERIFIED` in code)

1. **Exact Zenoh `cmd_vel` key expression.** The config topic is `cmd_vel` and the
   Go2 publishes via `internal/zenoh`, but the fully-qualified key (any robot
   namespace / `rt/` ros2dds-bridge prefix) is not confirmed from source — the
   publish line in `cmd_vel.go` was not visible. Adapter default: `cmd_vel` /
   `**/cmd_vel`, overridable.
2. **Exact tool-call wire format** (OpenAI `tool_calls` vs Gemini `functionCall`) as
   OM1 emits it on the wire — `OM1ActionSchema` parses both shapes tolerantly, to be
   pinned against a captured Cortex response.
3. **The default OpenMind portal base URL string** and whether VLM/ASR inputs accept
   a `base_url` override for the same portal.
