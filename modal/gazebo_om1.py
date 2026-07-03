"""Modal app SKELETON — record a REAL OM1 + Gazebo episode (Tier 3; the WS5 "run-
verified" episode).

⚠️  UNVERIFIED SCAFFOLD — NOT RUN. This captures the *architecture* and the real launch
commands (from OM1's docs), but the base image, OM1 build/run, the Go2 sim package, the
headless-render flags, and the cmd_vel Zenoh key are exactly the items
docs/om1-integration.md flags UNVERIFIED. Treat this as the skeleton to iterate on with
a real Modal account + patience — it will NOT run out of the box. See modal/README.md
(Tier 3) and docs/record-om1-gazebo.md for the record-harness wiring.

Pipeline (maps to docs/record-om1-gazebo.md):
  Gazebo (headless EGL) --ros2 cmd_vel--> zenoh-bridge-ros2dds --zenoh--> Plumbline tap
  OM1 Go binary --model calls (base_url = the proxy)--> Plumbline HTTP proxy --> Modal models
  A Plumbline RecordingSession owns the episode; the trace is saved to a Modal Volume.
"""

import subprocess

import modal

MINUTES = 60
volume = modal.Volume.from_name("plumbline-traces", create_if_missing=True)

# TODO(UNVERIFIED): base image + package set. osrf/ros:humble-desktop-full ships ROS2 +
# Gazebo; add the Unitree Go2 sim, the Zenoh<->ROS2DDS bridge, Go (to build OM1), Python.
image = (
    modal.Image.from_registry("osrf/ros:humble-desktop-full", add_python="3.12")
    .apt_install("git", "wget", "xvfb", "golang-go")
    .run_commands(
        # TODO(UNVERIFIED): clone + build the OM1 Go binary and the Go2 gazebo sim.
        # "git clone https://github.com/OpenMind/OM1 /opt/OM1 && cd /opt/OM1 && make build",
        # "git clone https://github.com/OpenMind/unitree_go2_ros2_sdk && colcon build",
        # "wget <zenoh-bridge-ros2dds release> -O /usr/local/bin/zenoh-bridge-ros2dds",
        "true",
    )
    .pip_install("eclipse-zenoh>=1.0.0", "httpx>=0.27", "uvicorn>=0.30")
    .add_local_dir(".", "/root/plumbline")  # this repo (the recording harness)
)
app = modal.App("plumbline-gazebo-om1")


@app.function(image=image, gpu="A10G", timeout=15 * MINUTES, volumes={"/traces": volume})
def record_episode(llm_url: str, vlm_url: str, seconds: int = 60) -> None:
    """Launch the sim + OM1 (pointed at the Modal models via the recording proxy),
    record for `seconds`, save the trace to the Volume.

    ⚠️  The launch commands are OM1's real documented ones, but the build/env is
    unproven — expect to iterate. Steps 4-6 (the Plumbline record wiring) follow
    docs/record-om1-gazebo.md; sketched here, not fleshed out."""
    import os

    subprocess.run(["pip", "install", "-e", "/root/plumbline[proxy,zenoh]"], check=True)

    # 1. Gazebo, headless (EGL / xvfb — no display on a Modal container).
    gazebo = subprocess.Popen(
        ["ros2", "launch", "go2_gazebo_sim", "go2_launch.py"],  # TODO: confirm the package
        env={**os.environ, "USE_SIM": "true", "DISPLAY": ""},
    )
    # 2. Zenoh <-> ROS2 DDS bridge (republishes cmd_vel onto Zenoh for the tap).
    bridge = subprocess.Popen(
        ["zenoh-bridge-ros2dds", "-c", "/root/plumbline/zenoh/zenoh_bridge_config.json5"]
    )
    # 3-5. TODO(record wiring, docs/record-om1-gazebo.md): start the Plumbline HTTP proxy
    #   (make_asgi_app + BoundaryTickPolicy + a RecordingCoordinator on the OM1 adapter);
    #   subscribe the OM1 adapter's cmd_vel bus_tap to session.record_bus_sample; set OM1's
    #   cortex_llm.config.base_url to the proxy (which forwards to llm_url / vlm_url); then:
    om1 = subprocess.Popen(
        ["make", "dev"],
        cwd="/opt/OM1",
        env={
            **os.environ,
            "CONFIG": "unitree_go2_autonomy",
            "USE_SIM": "true",
            # OM's cortex_llm.config.base_url is edited to the proxy; llm_url/vlm_url are
            # what the proxy forwards to. OM_API_KEY carries through unchanged.
            "OM_API_KEY": os.environ.get("OM_API_KEY", "openmind_free"),
        },
    )

    # 6. Run for `seconds`, then tear down and persist the trace.
    import time

    time.sleep(seconds)
    for process in (om1, bridge, gazebo):
        process.terminate()
    # TODO: session.close(); copy the store into /traces (the Volume).
    volume.commit()
    print(
        f"llm_url={llm_url} vlm_url={vlm_url}: recorded ~{seconds}s (SKELETON — verify the trace)"
    )
