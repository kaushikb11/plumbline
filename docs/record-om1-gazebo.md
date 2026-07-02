# Recording a real OM1 + Gazebo episode

How to capture a faithful OM1 episode (Unitree Go2 in Gazebo) with Plumbline, so the
OM1 adapter can be validated against a real run rather than synthetic fixtures (WS5
definition-of-done). Prerequisites are an environment Plumbline can't provide itself:
Ubuntu + ROS2 + Gazebo + the OM1 Go binary. Interface facts referenced here are
established in [om1-integration.md](om1-integration.md).

## What gets captured at each seam

- **SENSOR_TO_CAPTION** (VLM / ASR) — HTTP proxy **only if the input type makes an
  HTTP (OpenAI-compatible) call**. ⚠️ OM1's *reference* config uses `VLMGeminiRTSP`
  (RTSP video) and `GoogleASRInput` (streaming), which move data over RTSP + WebSocket
  (`wss://api.openmind.com`) — the HTTP proxy **cannot** capture these, so this seam is
  not recordable for the default config without a WebSocket/RTSP tap (see
  [limitations.md](limitations.md) gap #1). Local inputs (`VLM_COCO_Local`) make no
  network call at all. HTTP-based perception endpoints are the capturable case.
- **CAPTION_TO_FUSE** — reconstructed by the adapter (no model call of its own).
- **FUSE_TO_DECIDE** — HTTP proxy: the Cortex LLM chat call and its tool-call decision.
- **DECIDE_TO_ACT** — Zenoh tap on the `cmd_vel` key (the CDR `Twist`), and/or
  reconstructed from the Cortex tool call.

## Steps

### 1. Start the recording proxy

```bash
plumbline record --upstream <the base URL OM1 would otherwise call> \
  --store ./traces --episode go2-gazebo-001
# listens on 127.0.0.1:8900 by default
```

`--upstream` is whatever `cortex_llm.config.base_url` (or the OpenMind portal) points
at. The proxy forwards, records, and returns responses unaltered (zero-touch).

### 2. Point OM1 at the proxy (no OM1 source changes)

Use the adapter to compute the config overrides, then set them in the JSON5 config:

```python
from plumbline.adapters.om1 import OM1Adapter

cfg = OM1Adapter(proxy_base_url="http://localhost:8900").configure_proxy()
print(cfg.config_fields)
# {'cortex_llm.config.base_url': 'http://localhost:8900/v1', ...}
```

Edit `config/unitree_go2_autonomy.json5` so `cortex_llm.config.base_url` (and any
OpenAI-compatible VLM/ASR input's `config.base_url`) equals the printed proxy URL.
Keep `api_key`/`OM_API_KEY` as-is — the proxy carries the request through unchanged.

### 3. Launch Gazebo + the Zenoh bridge

```bash
source install/setup.bash && ros2 launch go2_gazebo_sim go2_launch.py
zenoh-bridge-ros2dds -c ./zenoh/zenoh_bridge_config.json5
```

### 4. Attach the action tap

Plumbline taps the action Zenoh key(s) on the SAME Zenoh session/router the bridge
uses. Confirm the real `cmd_vel` key expression first (see om1-integration.md open
item #1) and pass it in if it differs from the default:

```python
import zenoh
from plumbline.adapters.om1 import OM1Adapter
from plumbline.session import RecordingSession
# ... open a RecordingSession on ./traces, episode go2-gazebo-001 ...

adapter = OM1Adapter(
    proxy_base_url="http://localhost:8900",
    zenoh_session=zenoh.open(zenoh.Config()),   # the injected real session
    action_key_expressions=("cmd_vel", "**/cmd_vel"),  # confirm against the bridge
)
tap = adapter.bus_tap()
tap.subscribe(session.record_bus_sample)
```

### 5. Run the agent and drive an episode

```bash
CONFIG=unitree_go2_autonomy USE_SIM=true make dev
# optionally teleop: ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Exercise a short scenario (approach an obstacle, turn, etc.), then close the session.

### 6. Verify

```bash
plumbline replay --store ./traces --episode go2-gazebo-001    # faithful replay
plumbline export go2-gazebo-001 --store ./traces -o spans.json --format otlp
```

Faithful replay must reproduce byte-identical model I/O and a matching action
sequence. Once a real episode is captured, pin the three open items in
om1-integration.md (the exact `cmd_vel` key, the tool-call wire format, the portal
URL) and drop their `UNVERIFIED` flags.
