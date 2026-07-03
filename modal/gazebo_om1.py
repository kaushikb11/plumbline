"""Modal app — record a REAL OM1 + Gazebo episode (Tier 3; the WS5 run-verified
episode with sim physics).

Replaces the earlier unrun skeleton with a staged, verifiable build, corrected by
the Tier-3 scouting pass and the SIL run (docs/om1-integration.md):

- The sim package is `go2_sim` in github.com/OpenMind/OM1-sim (the OM1 docs' name
  `go2_gazebo_sim` is stale). ROS2 Humble + Gazebo (ros_gz), champ vendored.
- **USE_SIM stays false/unset.** OM1's USE_SIM=true swaps in a hybrid Zenoh session
  that routes cmd_vel (and other topics) to OpenMind's cloud broker with no URL
  override — which would blind the local tap. All-local is the SIL-verified
  topology: OM1 zenoh-client -> the bridge's zenoh router (tcp/127.0.0.1:7447)
  <-> ROS2 DDS <-> Gazebo.
- Real physics closes the loop the SIL run stubbed: champ executes cmd_vel, the
  robot MOVES, odom (bridged from the sim) feeds OM1's UnitreeGo2Odom input. Only
  om/paths keeps a stub publisher: the sim's om_path node publishes range-keyed
  /om/paths/r{K}, not the bare om/paths OM1's provider subscribes to (observed
  discrepancy, reported in the run summary).

Usage (deploy the Tier-1 LLM first — modal deploy modal/llm.py):

    modal run modal/gazebo_om1.py::doctor                       # component checks
    modal run modal/gazebo_om1.py::record --llm-url https://... --seconds 90

The trace lands in the `plumbline-traces` Volume (modal volume get) and the run
prints seam counts, observed bus keys, faithful-replay verdict, and how far the
robot physically moved.
"""

import re
import subprocess
import time
from pathlib import Path

import modal

# Strings interpolated into `bash -lc` commands and filesystem paths (world /
# episode / scene names from CLI args or a data file) are validated to a safe
# charset so a value with quotes / ; / $() / .. can't break out of the command
# or escape the store root. Numeric pose values are float()-coerced at use.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe(value: str, kind: str) -> str:
    if not _SAFE_NAME.match(value):
        raise ValueError(f"unsafe {kind}: {value!r} (allowed: letters, digits, '.', '_', '-')")
    return value


MINUTES = 60
OM1_COMMIT = "70c23e21fa3a9154e602009033c8cf05af262a18"  # the source-audited pin
GO_VERSION = "1.25.3"  # OM1 go.mod requires go >= 1.25.0

volume = modal.Volume.from_name("plumbline-traces", create_if_missing=True)

_ROS_PKGS = (
    "ros-humble-ros-gz",
    "ros-humble-gz-ros2-control",
    "ros-humble-ros2-control",
    "ros-humble-ros2-controllers",
    "ros-humble-joint-state-publisher",
    "ros-humble-xacro",
)

image = (
    modal.Image.from_registry("osrf/ros:humble-desktop-full", add_python="3.12")
    .apt_install(
        "git",
        "wget",
        "curl",
        "unzip",
        "build-essential",
        "python3-colcon-common-extensions",
        "python3-rosdep",
        *_ROS_PKGS,
    )
    .run_commands(
        # --- the Go2 Gazebo sim workspace (champ is vendored in-repo) ---
        "git clone --depth 1 https://github.com/OpenMind/OM1-sim /opt/om1-sim",
        "bash -lc 'rosdep update || true'",
        "bash -lc 'source /opt/ros/humble/setup.bash && cd /opt/om1-sim && "
        "(rosdep install --from-paths gazebo_sim unitree_api unitree_go om_api "
        " --ignore-src -r -y || true)'",
        # Mixed-python trap: Modal's add_python (3.12) is the python3 CMake finds,
        # but ROS Humble's rosidl toolchain lives in the system 3.10. Pin CMake to
        # the system interpreter AND give 3.12 the build modules as a belt.
        "python -m pip install -q 'empy==3.3.4' catkin_pkg lark numpy pyyaml setuptools",
        "bash -lc 'source /opt/ros/humble/setup.bash && cd /opt/om1-sim && "
        "colcon build --symlink-install --packages-up-to go2_sim om_path "
        "--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3'",
        # --- zenoh-bridge-ros2dds (its zenoh router listens on tcp/7447) ---
        "bash -c \"echo 'deb [trusted=yes] https://download.eclipse.org/zenoh/debian-repo/ /' "
        '> /etc/apt/sources.list.d/zenoh.list" '
        "&& apt-get update && apt-get install -y zenoh-bridge-ros2dds",
        # --- Go toolchain + the OM1 binary (pinned to the audited commit) ---
        f"wget -q https://go.dev/dl/go{GO_VERSION}.linux-amd64.tar.gz -O /tmp/go.tgz "
        "&& tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz",
        f"git clone https://github.com/OpenMind/OM1 /opt/OM1 && cd /opt/OM1 "
        f"&& git checkout {OM1_COMMIT}",
    )
    # OM1's ASR input links github.com/gordonklaus/portaudio -> needs the
    # portaudio C library (the build failure hides mid-stream in go build -v).
    .apt_install("portaudio19-dev", "pkg-config")
    .run_commands(
        "cd /opt/OM1 && PATH=/usr/local/go/bin:$PATH make build",
        # go2_description (worlds/urdf/models) and champ are found at runtime via
        # FindPackageShare but are NOT declared deps of go2_sim — build them too.
        "bash -lc 'source /opt/ros/humble/setup.bash && cd /opt/om1-sim && "
        "colcon build --symlink-install "
        "--packages-select go2_description champ champ_msgs champ_base "
        "--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3'",
    )
    # Cyclone RMW: OM1-sim ships cyclonedds.xml (loopback iface, multicast off,
    # MaxAutoParticipantIndex=120) — required in a container where multicast is
    # unavailable and the sim spawns a dozen-plus DDS participants.
    .apt_install("ros-humble-rmw-cyclonedds-cpp")
    # Second round of the mixed-python trap, found at RUNTIME: ament_python node
    # scripts use `#!/usr/bin/env python3` (Modal's 3.12 -> rclpy's 3.10 pybind
    # ext fails), and rosidl generated om_api/unitree_* Python typesupport for
    # 3.12 despite the CMake pin. Rebuild the message packages + python nodes
    # with `python3` RESOLVING to the system 3.10 (PATH pin beats every
    # discovery mechanism at once); _SIM_ENV applies the same pin at runtime.
    .run_commands(
        # CLEAN rebuild: an incremental colcon pass reuses the cached CMake
        # configure and keeps the 3.12-flavored generated typesupport.
        "bash -lc 'source /opt/ros/humble/setup.bash && cd /opt/om1-sim && "
        "rm -rf build/om_api build/unitree_api build/unitree_go build/om_path build/go2_sim "
        "install/om_api install/unitree_api install/unitree_go install/om_path install/go2_sim "
        "&& PATH=/usr/bin:$PATH colcon build --symlink-install "
        "--packages-select om_api unitree_api unitree_go om_path go2_sim "
        "--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3'",
    )
    .pip_install("eclipse-zenoh>=1.0.0", "httpx>=0.27", "uvicorn>=0.30", "websockets>=12")
    # EGL/GPU rendering for headless Gazebo sensor cameras.
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all"})
    .add_local_dir(
        ".",
        "/root/plumbline-repo",
        ignore=[".venv/**", ".git/**", "traces*/**", ".pytest_cache/**", "**/__pycache__/**"],
    )
)
app = modal.App("plumbline-gazebo-om1")

_WORLD = "/opt/om1-sim/install/go2_description/share/go2_description/worlds/home_world.sdf"
_SIM_ENV = (
    # PATH pin first: `env python3` must resolve to the system 3.10 that ROS's
    # compiled extensions target, not Modal's 3.12 (see the image-build note).
    "export PATH=/usr/bin:$PATH && "
    "source /opt/ros/humble/setup.bash && source /opt/om1-sim/install/setup.bash && "
    "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
    "CYCLONEDDS_URI=file:///opt/om1-sim/cyclonedds/cyclonedds.xml"
)
_BRIDGE_CONFIG = "/opt/om1-sim/zenoh/zenoh_bridge_config.json5"

# The plumbline SIL config, re-targeted at the sim: REAL odom (bridged from
# Gazebo), Move -> cmd_vel (bridged INTO Gazebo, champ executes it). GeminiLLM =
# the OpenAI-compat client with tool_choice=required (a Move every tick).
_OM1_CONFIG = """{
  version: "v1.1.0",
  hertz: 1,
  name: "plumbline_gazebo",
  api_key: "openmind_free",
  system_prompt_base: "You are Bits, a curious robot dog exploring an unfamiliar building. On EVERY response you MUST call the Move function exactly once. Choose ONLY from the movement directions currently listed as safe in your observations. Policy: prefer 'move forwards' whenever it is listed as safe; when it is not, turn toward open space ('turn left' or 'turn right', whichever is listed as safe); if almost nothing is safe, choose 'move back' or 'stand still'.",
  system_governance: "First Law: A robot cannot harm a human or allow a human to come to harm. Never choose a movement direction that is not currently listed as safe.",
  system_prompt_examples: "Example: if 'move forwards' is not among the safe directions but 'turn left' is, call Move with 'turn left'.",
  agent_inputs: [
    { type: "UnitreeGo2Odom" },
    { type: "Paths" },
  ],
  cortex_llm: {
    type: "GeminiLLM",
    config: {
      base_url: "http://127.0.0.1:8900/v1",
      api_key: "sil-local",
      model: "cortex",
      agent_name: "Bits",
      history_length: 2,
    },
  },
  agent_actions: [
    {
      name: "unitree_go2_autonomy",
      llm_label: "Move",
      connector: "move",
      config: { cmd_vel_topic: "cmd_vel" },
    },
  ],
}
"""


def _sh(command: str, timeout: int = 120) -> tuple[int, str]:
    proc = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail[:400]}")
    return ok


@app.function(image=image, gpu="T4", timeout=20 * MINUTES)
def doctor() -> None:
    """Per-component health checks — iterate here before burning time on the
    full closed loop."""
    results = []

    code, out = _sh("gz sim --version || ign gazebo --version")
    results.append(_check("gazebo binary", code == 0, out.splitlines()[0] if out else ""))

    code, out = _sh("nvidia-smi -L")
    results.append(_check("gpu visible", code == 0, out))

    code, out = _sh(f"{_SIM_ENV} && ros2 pkg prefix go2_sim && ros2 pkg prefix om_path")
    results.append(_check("go2_sim built", code == 0, out))

    code, out = _sh("zenoh-bridge-ros2dds --version")
    results.append(_check("zenoh bridge", code == 0, out))

    code, out = _sh(
        "LD_LIBRARY_PATH=/opt/OM1/.zenoh-c/lib /opt/OM1/build/om1 --help 2>&1 | head -3"
    )
    results.append(_check("om1 binary", code == 0, out))

    code, out = _sh(
        "cd /root && pip install -q -e plumbline-repo 2>&1 | tail -1 && "
        "python -c 'import plumbline.adapters.om1, zenoh; print(\"import ok\")'"
    )
    results.append(_check("plumbline + zenoh py", code == 0, out))

    # The riskiest piece: a headless Gazebo server with GPU rendering, briefly.
    # Fortress (Humble's ros_gz pairing) ships the `ign` CLI, not `gz`.
    sim = subprocess.Popen(
        ["bash", "-lc", f'{_SIM_ENV} && ign gazebo -s -r --headless-rendering "{_WORLD}"'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(25)
    alive = sim.poll() is None
    sim.terminate()
    tail = (sim.stdout.read() if sim.stdout else "")[-400:]
    results.append(_check("headless gz server (25s)", alive, tail or "server stayed up"))

    print(f"\ndoctor: {sum(results)}/{len(results)} checks passed")
    if not all(results):
        raise SystemExit(1)


@app.function(image=image, gpu="T4", timeout=15 * MINUTES)
def probe(world: str = "maze_world") -> None:
    """Boot ONLY the sim and interrogate the lidar->scan->om_path chain link by
    link — the cheap diagnosis for why om/paths/r{K} might be silent."""
    world = _safe(world, "world")
    world_path = f"/opt/om1-sim/install/go2_description/share/go2_description/worlds/{world}.sdf"
    subprocess.Popen(
        [
            "bash",
            "-lc",
            f"{_SIM_ENV} && ros2 launch go2_sim go2_launch.py gui:=false "
            f"world:='{world_path} -s --headless-rendering'",
        ],
        stdout=open("/tmp/sim.log", "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
    )
    time.sleep(75)  # let controllers spawn and sensors start

    for name, cmd, timeout in (
        ("nodes", "ros2 node list", 30),
        ("topics", "ros2 topic list", 30),
        ("scan msg (10s)", "timeout 10 ros2 topic echo --once /scan | head -5", 20),
        (
            "om/paths/r100 msg (10s)",
            "timeout 10 ros2 topic echo --once /om/paths/r100 | head -5",
            20,
        ),
        ("gz topics", "timeout 10 ign topic -l | head -30", 20),
        # Experiment-A feasibility: a real camera frame + ground-truth pose.
        (
            "rgb frame shape (15s)",
            "timeout 15 ros2 topic echo --once /rgb_image --field height && "
            "timeout 15 ros2 topic echo --once /rgb_image --field width && "
            "timeout 15 ros2 topic echo --once /rgb_image --field encoding",
            50,
        ),
        (
            "gz ground-truth pose (10s)",
            "timeout 10 ign topic -e -t /model/go2/pose -n 1 2>/dev/null | head -12",
            20,
        ),
        (
            "gz set_pose service (teleport for scene sampling)",
            "ign service -l | grep set_pose | head -3",
            20,
        ),
    ):
        code, out = _sh(f"{_SIM_ENV} && {cmd}", timeout=timeout)
        print(f"--- {name} (exit {code}) ---\n{out[:1200]}\n")

    log = Path("/tmp/sim.log").read_text()
    for marker in ("om_path", "error", "Error", "scan"):
        lines = [line for line in log.splitlines() if marker in line][:8]
        print(f"--- sim.log lines containing {marker!r} ---")
        print("\n".join(line[:200] for line in lines) or "(none)")


_FRAME_SAVER = """
import json, sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class Saver(Node):
    def __init__(self, out):
        super().__init__("plumbline_frame_saver")
        self.out = out
        self.done = False
        self.create_subscription(Image, "/rgb_image", self.on_frame, 1)

    def on_frame(self, msg):
        if self.done:
            return
        with open(self.out + ".raw", "wb") as fh:
            fh.write(bytes(msg.data))
        with open(self.out + ".meta.json", "w") as fh:
            json.dump({"height": msg.height, "width": msg.width,
                       "encoding": msg.encoding, "step": msg.step}, fh)
        self.done = True

rclpy.init()
node = Saver(sys.argv[1])
for _ in range(20):
    if node.done:
        break
    rclpy.spin_once(node, timeout_sec=1.0)
node.destroy_node()
rclpy.shutdown()
raise SystemExit(0 if node.done else 1)
"""


def _parse_go2_pose(dump: str) -> dict[str, float] | None:
    """Extract the go2 base pose from an `ign topic -e -t /model/go2/pose` dump
    (Pose_V text proto: pick the pose block named exactly 'go2', else the first)."""
    import math
    import re

    blocks = re.split(r"\npose \{", dump)
    chosen = None
    for block in blocks[1:]:
        if re.search(r'name: "go2"\n', block) or 'value: "go2"' in block:
            chosen = block
            break
    if chosen is None and len(blocks) > 1:
        chosen = blocks[1]
    if chosen is None:
        return None
    pos = re.search(
        r"position \{\s*(?:x: ([\-\d.e]+))?\s*(?:y: ([\-\d.e]+))?\s*(?:z: ([\-\d.e]+))?", chosen
    )
    ori = re.search(
        r"orientation \{\s*(?:x: ([\-\d.e]+))?\s*(?:y: ([\-\d.e]+))?"
        r"\s*(?:z: ([\-\d.e]+))?\s*(?:w: ([\-\d.e]+))?",
        chosen,
    )
    if not pos or not ori:
        return None
    px, py, pz = (float(v) if v else 0.0 for v in pos.groups())
    qx, qy, qz = (float(v) if v else 0.0 for v in ori.groups()[:3])
    qw = float(ori.group(4)) if ori.group(4) else 1.0
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    return {"x": px, "y": py, "z": pz, "yaw": yaw, "pitch": pitch, "roll": roll}


@app.function(image=image, gpu="T4", timeout=30 * MINUTES, volumes={"/traces": volume})
def capture_scenes(poses: str, name: str = "maze-scenes", world: str = "maze_world") -> str:
    """Capture ground-truth-labeled camera scenes for sim-grounded Experiment A:
    teleport the robot to each pose (bench/maze_scenes.py output, JSON), let
    physics settle, record the ACHIEVED pose (gz ground truth — render(G) is
    recomputed from this locally), and save one fresh RGB frame.

    ⚠️ KNOWN ISSUE (blocks Experiment A): `set_pose` on the champ-controlled
    quadruped is immediately overridden by the standing controller — the first
    capture run landed all 12 scenes at spawn (~-0.19,-0.14), pitched ~54° over.
    A working version must pause physics before the teleport (gz `/world/<w>/
    control` set_paused=true), set_pose, step, then unpause — OR reset the
    controllers per scene. Until then the achieved poses are not the requested
    ones and the frames are unusable. The ground-truth geometry (bench/
    maze_scenes.py) is independent of this and correct."""
    import json as json_module
    import math

    pose_list = json_module.loads(poses)
    world = _safe(world, "world")
    world_path = f"/opt/om1-sim/install/go2_description/share/go2_description/worlds/{world}.sdf"
    subprocess.Popen(
        [
            "bash",
            "-lc",
            f"{_SIM_ENV} && ros2 launch go2_sim go2_launch.py gui:=false "
            f"world:='{world_path} -s --headless-rendering'",
        ],
        stdout=open("/tmp/sim.log", "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        code, out = _sh(f"{_SIM_ENV} && ros2 topic list", timeout=30)
        if code == 0 and "/rgb_image" in out:
            break
        time.sleep(5)
    else:
        raise RuntimeError("sim never published /rgb_image")
    time.sleep(10)  # let the controller reach standing
    Path("/tmp/save_frame.py").write_text(_FRAME_SAVER)

    out_dir = Path(f"/traces/{_safe(name, 'name')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    captured: list[dict[str, object]] = []
    for pose in pose_list:
        scene_id = _safe(str(pose["scene_id"]), "scene_id")
        # float() every numeric before it enters the command — a non-numeric value
        # in the JSON pose file cannot then inject shell/proto content.
        px, py, yaw = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
        qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
        request = (
            f'name: "go2", position: {{x: {px}, y: {py}, z: 0.45}}, '
            f"orientation: {{z: {qz}, w: {qw}}}"
        )
        code, out = _sh(
            f"{_SIM_ENV} && ign service -s /world/{world}/set_pose "
            f"--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 3000 "
            f"--req '{request}'",
            timeout=30,
        )
        if code != 0:
            print(f"{scene_id}: teleport FAILED: {out[:200]}")
            continue
        time.sleep(4)  # physics settle under the standing controller
        code, dump = _sh(
            f"{_SIM_ENV} && timeout 8 ign topic -e -t /model/go2/pose -n 1", timeout=20
        )
        achieved = _parse_go2_pose(dump) if code == 0 else None
        frame_out = str(out_dir / scene_id)
        code, out = _sh(
            f"{_SIM_ENV} && timeout 25 python3 /tmp/save_frame.py {frame_out}", timeout=40
        )
        entry: dict[str, object] = {
            "scene_id": scene_id,
            "requested": pose,
            "achieved": achieved,
            "frame_saved": code == 0,
        }
        captured.append(entry)
        print(f"{scene_id}: frame={'ok' if code == 0 else 'FAIL'} achieved={achieved}")
    (out_dir / "capture.json").write_text(json_module.dumps(captured, indent=1))
    volume.commit()
    saved = sum(1 for entry in captured if entry["frame_saved"])
    print(f"captured {saved}/{len(pose_list)} scenes -> volume {name}/")
    return f"{saved}/{len(pose_list)}"


@app.function(image=image, gpu="T4", timeout=30 * MINUTES, volumes={"/traces": volume})
def record(
    llm_url: str,
    seconds: int = 240,
    episode_id: str = "om1-gazebo-001",
    world: str = "maze_world",
    paths_range: str = "r100",
) -> str:
    """The Tier-3 closed loop: Gazebo physics + bridge + OM1 + Plumbline record.

    `world` picks the go2_description world (maze_world / walled_world /
    home_world); `paths_range` picks which of the sim's lidar-derived path
    horizons (r50/r100/r200, cm) is relayed to OM1's bare `om/paths` key."""
    import asyncio
    import json
    import re
    import struct
    import sys
    from collections import Counter

    episode_id = _safe(episode_id, "episode_id")  # -> /traces/<episode_id>-store
    paths_range = _safe(paths_range, "paths_range")  # -> the sim/om/paths/<range> key

    # A running interpreter won't see a mid-process editable install (.pth files
    # are only processed at startup) — put the mounted repo on sys.path directly;
    # its dependencies are already baked into the image.
    sys.path.insert(0, "/root/plumbline-repo")

    world = _safe(world, "world")
    world_path = f"/opt/om1-sim/install/go2_description/share/go2_description/worlds/{world}.sdf"

    # 1. Gazebo + go2_sim, headless. The launch composes gz_args from `world` +
    # " -r", so the headless server flags ride in through world:=.
    sim = subprocess.Popen(  # noqa: F841 - keep the process handle alive
        [
            "bash",
            "-lc",
            f"{_SIM_ENV} && ros2 launch go2_sim go2_launch.py gui:=false "
            f"world:='{world_path} -s --headless-rendering'",
        ],
        stdout=open("/tmp/sim.log", "w"),  # noqa: SIM115 - long-lived process log
        stderr=subprocess.STDOUT,
    )

    # Wait for the control stack: /odom (champ) and /cmd_vel consumers.
    for _ in range(60):
        code, out = _sh(f"{_SIM_ENV} && ros2 topic list", timeout=30)
        if code == 0 and "/odom" in out:
            break
        time.sleep(5)
    else:
        print(Path("/tmp/sim.log").read_text()[-3000:])
        raise RuntimeError("sim never published /odom — see sim.log above")
    print("sim up: /odom present")

    # 2. The Zenoh <-> ROS2 DDS bridge, with OM1-sim's shipped config PLUS a /sim
    # zenoh namespace: bridged sim topics land on sim/* keys so OM1 never consumes
    # raw sim odometry directly. The driver below is an explicit SIM-GAP SHIM
    # between the two: champ's odom is planar (pose.z = 0) while the real Go2
    # firmware reports body height there — and OM1's Move connector requires
    # "standing" (z > 0.24 m) before it will drive. The shim relays sim/odom ->
    # odom with z lifted to the real robot's standing height, and OM1's cmd_vel ->
    # sim/cmd_vel into the sim. Zero OM1 changes.
    patched = (
        Path(_BRIDGE_CONFIG)
        .read_text()
        .replace("domain: 0,", 'domain: 0,\n      namespace: "/sim",')
    )
    Path("/tmp/bridge_config.json5").write_text(patched)
    bridge = subprocess.Popen(  # noqa: F841 - keep the process handle alive
        ["bash", "-lc", f"{_SIM_ENV} && zenoh-bridge-ros2dds -c /tmp/bridge_config.json5"],
        stdout=open("/tmp/bridge.log", "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
    )
    for _ in range(30):
        if _sh("bash -c 'exec 3<>/dev/tcp/127.0.0.1/7447' 2>/dev/null", timeout=10)[0] == 0:
            break
        time.sleep(2)
    else:
        print(Path("/tmp/bridge.log").read_text()[-2000:])
        raise RuntimeError("bridge never listened on 7447 — see bridge.log above")
    print("bridge up: tcp/7447 listening")

    # 3. The Plumbline record harness, in-process (proxy + tap + coordinator).
    import zenoh
    from plumbline.adapters.om1 import OM1Adapter
    from plumbline.core.seam import Seam
    from plumbline.core.store import TraceStore
    from plumbline.core.trace import canonicalize
    from plumbline.proxy.recording import ReplayingProxy
    from plumbline.proxy.tick import PerCallTickPolicy
    from plumbline.recording import RecordingCoordinator
    from plumbline.transport.zenoh_shim import ZenohSessionAdapter

    zconf = zenoh.Config.from_json5(
        '{"mode": "client", "connect": {"endpoints": ["tcp/127.0.0.1:7447"]}}'
    )
    raw_session = zenoh.open(zconf)
    store = TraceStore(root="/tmp/traces")
    adapter = OM1Adapter(
        proxy_base_url="http://127.0.0.1:8900",
        zenoh_session=ZenohSessionAdapter(raw_session),
        # Bare key only: the default **/cmd_vel would also match our sim/cmd_vel
        # relay and double-record every frame.
        action_key_expressions=("cmd_vel",),
    )
    coordinator = RecordingCoordinator(store, episode_id=episode_id, adapter=adapter)
    coordinator.open()
    tap = adapter.bus_tap()
    assert tap is not None
    tap.subscribe(coordinator.record_bus_sample)

    # Track the robot's REAL pose from bridged odom (proof of physical motion).
    poses: list[tuple[float, float]] = []

    odom_out = raw_session.declare_publisher("odom")  # the shimmed stream OM1 consumes
    _STANDING_Z = struct.pack("<d", 0.30)  # real Go2 firmware reports body height here

    def on_sim_odom(sample: "zenoh.Sample") -> None:
        data = bytes(sample.payload)
        try:  # header(4)+stamp(8)+frame_id+child_frame_id -> 7 float64 (odom_zenoh.go)
            pos = 16
            flen = struct.unpack_from("<I", data, 12)[0]
            pos += flen + (4 - (pos + flen - 4) % 4) % 4
            clen = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + clen + (8 - (pos + clen) % 8) % 8
            x, y = struct.unpack_from("<2d", data, pos)[:2]
            poses.append((x, y))
            # SIM-GAP SHIM: champ's planar odom has pose.z = 0; splice in the
            # standing body height the real firmware provides, pass the rest through.
            shimmed = data[: pos + 16] + _STANDING_Z + data[pos + 24 :]
            odom_out.put(shimmed)
        except struct.error:
            pass

    odom_sub = raw_session.declare_subscriber("sim/odom", on_sim_odom)

    # Relay OM1's cmd_vel INTO the namespaced sim (raw bytes, untouched), counting
    # frames independently of the tap.
    cmd_vel_frames: list[int] = []
    sim_cmd_out = raw_session.declare_publisher("sim/cmd_vel")

    def on_cmd_vel(sample: "zenoh.Sample") -> None:
        data = bytes(sample.payload)
        cmd_vel_frames.append(len(data))
        sim_cmd_out.put(data)

    cmd_vel_sub = raw_session.declare_subscriber(  # noqa: F841 - keep sub alive
        "cmd_vel", on_cmd_vel
    )

    # REAL perception: relay the sim's lidar-derived path availability. om_path
    # publishes range-keyed /om/paths/r{K} while OM1 subscribes the bare
    # om/paths — a pure key rename, bytes untouched. The prompt's "safe movement
    # directions" text now tracks the actual simulated lidar, so decisions are
    # conditioned on real perception. Distinct path states are counted as the
    # episode's perception-variety metric.
    paths_pub = raw_session.declare_publisher("om/paths")
    sim_paths_relayed: list[int] = []
    path_states: Counter[str] = Counter()

    def on_sim_paths(sample: "zenoh.Sample") -> None:
        data = bytes(sample.payload)
        sim_paths_relayed.append(len(data))
        try:  # header(4)+stamp(8)+frame_id -> sequence<uint32> (deserializePaths)
            pos = 12
            flen = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + flen + (4 - (pos + flen) % 4) % 4
            count = struct.unpack_from("<I", data, pos)[0]
            indices = struct.unpack_from(f"<{count}I", data, pos + 4)
            path_states[",".join(map(str, sorted(indices)))] += 1
        except struct.error:
            pass
        paths_pub.put(data)  # relay raw bytes, untouched

    paths_sub = raw_session.declare_subscriber(  # noqa: F841 - keep sub alive
        f"sim/om/paths/{paths_range}", on_sim_paths
    )

    # All-clear stub: FALLBACK ONLY, if the sim publishes no paths (e.g. no /scan).
    def _cdr_paths() -> bytes:
        name = b"paths\x00"
        buf = bytearray(b"\x00\x01\x00\x00") + struct.pack("<iI", 0, 0)
        buf += struct.pack("<I", len(name)) + name
        buf += b"\x00" * ((4 - (len(buf) - 4) % 4) % 4)
        indices = tuple(range(10))
        buf += struct.pack("<I", len(indices)) + struct.pack(f"<{len(indices)}I", *indices)
        return bytes(buf)

    async def run_harness() -> None:
        import httpx
        import uvicorn
        from plumbline.proxy.http import AsyncHTTPProxy
        from plumbline.proxy.server import HttpxTransport, make_asgi_app

        # Warm the LLM BEFORE OM1 starts ticking: a scale-to-zero cold start eats
        # the whole recording window otherwise (observed: 1 decision in 90s).
        async with httpx.AsyncClient(timeout=600.0) as warm:
            response = await warm.get(f"{llm_url.rstrip('/')}/v1/models")
            response.raise_for_status()
        print("llm warm")

        proxy = AsyncHTTPProxy(
            transport=HttpxTransport(httpx.AsyncClient(timeout=300.0)),
            recorder=coordinator,
            store=store,
            tick_policy=PerCallTickPolicy(boundary_seam=Seam.FUSE_TO_DECIDE),
        )
        asgi = make_asgi_app(proxy, upstream=llm_url.rstrip("/"), episode_id=episode_id)
        server = uvicorn.Server(
            uvicorn.Config(asgi, host="127.0.0.1", port=8900, log_level="warning")
        )
        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(2)

        # 4. OM1, all-local (USE_SIM unset — the cloud-broker trap).
        Path("/opt/OM1/config/plumbline_gazebo.json5").write_text(_OM1_CONFIG)
        om1 = subprocess.Popen(
            [
                "bash",
                "-lc",
                "cd /opt/OM1 && LD_LIBRARY_PATH=/opt/OM1/.zenoh-c/lib "
                "./build/om1 -config plumbline_gazebo -log-level info",
            ],
            stdout=open("/tmp/om1.log", "w"),  # noqa: SIM115
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + seconds
        stub_grace = time.time() + 20  # give the sim's real paths time to appear
        while time.time() < deadline:
            if not sim_paths_relayed and time.time() > stub_grace:
                paths_pub.put(_cdr_paths())  # fallback only — reported in the summary
            await asyncio.sleep(0.1)
        om1.terminate()
        server.should_exit = True
        await serve_task

    asyncio.run(run_harness())
    coordinator.close()  # seal the episode on disk

    # PERSIST FIRST: nothing between capture and the Volume commit may fail
    # (learned the hard way — a hung zenoh close once discarded a full episode).
    subprocess.run(["cp", "-r", "/tmp/traces", f"/traces/{episode_id}-store"], check=True)
    Path(f"/traces/{episode_id}-poses.json").write_text(json.dumps(poses))
    volume.commit()
    print(f"trace persisted to volume: {episode_id}-store")

    # Best-effort transport teardown; the container dies right after anyway.
    for closer in (odom_sub.undeclare, cmd_vel_sub.undeclare, paths_sub.undeclare, tap.close):
        try:
            closer()
        except Exception as exc:  # noqa: BLE001 - teardown must never lose the run
            print(f"teardown (non-fatal): {type(exc).__name__}: {exc}")

    # 5. Verify: seam counts, observed bus keys, physical motion, faithful replay.
    episode = store.load_episode(episode_id)
    by_seam: dict[str, int] = {}
    keys: set[str] = set()
    for event in episode.events:
        by_seam[event.seam.value] = by_seam.get(event.seam.value, 0) + 1
        key = event.params.get("plumbline.bus_key")
        if isinstance(key, str):
            keys.add(key)
    replay = ReplayingProxy(store, episode_id)
    from plumbline.core.interceptor import Context

    ctx = Context(episode_id=episode_id, model_id=None, params={}, logical_tick=0)
    identical = all(
        canonicalize(replay.faithful(e.request, ctx)).digest == canonicalize(e.response).digest
        for e in episode.events
    )
    moved = 0.0
    if len(poses) >= 2:
        moved = sum(
            ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            for (x1, y1), (x2, y2) in zip(poses, poses[1:], strict=False)
        )
    om1_log = Path("/tmp/om1.log").read_text()
    om1_tail = om1_log[-1500:]
    ai_commands = om1_log.count('"AI command"')
    decision_histogram = dict(
        Counter(re.findall(r'"msg":"AI command","action":"([^"]+)"', om1_log))
    )
    # The Move connector's gates (move.go) — which one swallowed the commands?
    connector_gates = {
        gate: om1_log.count(gate)
        for gate in (
            "waiting for location data",
            "movement in progress",
            "robot already moving",
            "cannot advance due to barrier",
            "cannot retreat due to barrier",
            "AI control disabled",
            "stand still",
        )
    }

    summary = json.dumps(
        {
            "events": len(episode.events),
            "by_seam": by_seam,
            "observed_bus_keys": sorted(keys),
            "faithful_replay_byte_identical": identical,
            "ai_commands": ai_commands,
            "decision_histogram": decision_histogram,
            "connector_gates": connector_gates,
            "raw_cmd_vel_frames": len(cmd_vel_frames),
            "odom_samples": len(poses),
            "distance_traveled_m": round(moved, 3),
            "world": world,
            "paths_source": f"sim:{paths_range}" if sim_paths_relayed else "stub(all-clear)",
            "sim_paths_relayed": len(sim_paths_relayed),
            "distinct_path_states": len(path_states),
        },
        indent=2,
    )
    print(summary)
    print("\n--- om1 log tail ---\n" + om1_tail)
    return summary
