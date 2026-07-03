"""Ground-truth scene machinery for sim-grounded Experiment A (§4, §7.3, §14.5).

The maze world is 23 STATIC axis-aligned box walls (go2_description
maze_world.sdf), so ground-truth scene state is exact geometry, not estimation:
from a robot pose, the true clearance in any direction is a ray-vs-AABB
intersection against the world model. `render(G)` — the §7.3 oracle context — is
built from THAT, never from any sensor.

Pipeline:
    python bench/maze_scenes.py <path/to/maze_world.sdf> > poses.json
    modal run modal/gazebo_om1.py::capture_scenes --poses "$(cat poses.json)"
    # then examples/experiment_a_sim.py builds scenes.json from the captured
    # frames + ACHIEVED poses and runs the verbosity sweep.

§14.5 honesty: render(G) states clearances and a blocked/clear verdict in plain
words, fixed BEFORE any captions are seen, and identically for every captioner
and verbosity level — it cannot be tuned to flatter the metric.
"""

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_RANGE_M = 12.0
ROBOT_RADIUS_M = 0.35
BLOCKED_AHEAD_M = 0.9  # a 0.5 m advance + margin: below this, moving forward hits


@dataclass(frozen=True)
class Wall:
    """An axis-aligned box wall projected to 2D: center + half-sizes."""

    cx: float
    cy: float
    hx: float
    hy: float


def parse_walls(sdf_path: str | Path) -> tuple[tuple[Wall, ...], tuple[Wall, ...]]:
    """Static box geometries as 2D AABBs, split into (walls, exclusion_zones).

    Axis-aligned boxes become clearance walls. Rotated models (the maze has a
    `ramp_upward`) are NOT in the ground-truth clearance model — instead their
    rotation-safe footprint (half-diagonal radius) becomes an exclusion zone so
    no scene is sampled where the un-modeled geometry could affect G."""
    sdf = Path(sdf_path).read_text(encoding="utf-8")
    walls: list[Wall] = []
    exclusions: list[Wall] = []
    for model in re.finditer(r'<model name="([^"]+)">(.*?)</model>', sdf, re.S):
        name, body = model.group(1), model.group(2)
        if "ground_plane" in name:
            continue
        pose_match = re.search(r"<pose>([^<]+)</pose>", body)
        size_match = re.search(r"<box>\s*<size>([^<]+)</size>", body)
        if not pose_match or not size_match:
            continue
        px, py, _pz, roll, pitch, yaw = (float(v) for v in pose_match.group(1).split())
        sx, sy, _sz = (float(v) for v in size_match.group(1).split())
        if (roll, pitch, yaw) == (0.0, 0.0, 0.0):
            walls.append(Wall(cx=px, cy=py, hx=sx / 2, hy=sy / 2))
        else:
            radius = math.hypot(sx / 2, sy / 2) + 1.0  # rotation-safe + standoff
            exclusions.append(Wall(cx=px, cy=py, hx=radius, hy=radius))
    assert walls, "no box walls parsed — wrong SDF?"
    return tuple(walls), tuple(exclusions)


def clearance(walls: tuple[Wall, ...], x: float, y: float, theta: float) -> float:
    """True distance from (x, y) along heading theta to the nearest wall
    (ray-vs-AABB slab method), capped at MAX_RANGE_M."""
    dx, dy = math.cos(theta), math.sin(theta)
    best = MAX_RANGE_M
    for wall in walls:
        tmin, tmax = 0.0, best
        ok = True
        for origin, direction, lo, hi in (
            (x, dx, wall.cx - wall.hx, wall.cx + wall.hx),
            (y, dy, wall.cy - wall.hy, wall.cy + wall.hy),
        ):
            if abs(direction) < 1e-12:
                if not lo <= origin <= hi:
                    ok = False
                    break
                continue
            t1, t2 = (lo - origin) / direction, (hi - origin) / direction
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                ok = False
                break
        if ok and tmin < best:
            best = tmin
    return best


def in_free_space(
    walls: tuple[Wall, ...], exclusions: tuple[Wall, ...], x: float, y: float
) -> bool:
    """Robot-radius-inflated free-space check, also outside all exclusion zones."""
    for zone in exclusions:
        if abs(x - zone.cx) <= zone.hx and abs(y - zone.cy) <= zone.hy:
            return False
    for wall in walls:
        if (
            abs(x - wall.cx) <= wall.hx + ROBOT_RADIUS_M
            and abs(y - wall.cy) <= wall.hy + ROBOT_RADIUS_M
        ):
            return False
    return -7.5 < x < 7.5 and -7.5 < y < 7.5


def render_g(walls: tuple[Wall, ...], x: float, y: float, yaw: float) -> str:
    """The §7.3 oracle context: true clearances + a blocked/clear verdict, in the
    same plain-language register a caption would use."""
    ahead = clearance(walls, x, y, yaw)
    left = clearance(walls, x, y, yaw + math.pi / 2)
    right = clearance(walls, x, y, yaw - math.pi / 2)
    verdict = (
        "the path ahead is blocked by a wall"
        if ahead < BLOCKED_AHEAD_M
        else "the path ahead is clear"
    )
    return (
        f"ground truth: the nearest wall is {ahead:.1f} m straight ahead, "
        f"{left:.1f} m to the left, and {right:.1f} m to the right; {verdict}"
    )


def select_poses(
    walls: tuple[Wall, ...],
    exclusions: tuple[Wall, ...] = (),
    *,
    n_blocked: int = 6,
    n_clear: int = 6,
    seed: int = 11,
) -> list[dict[str, float | str | bool]]:
    """A balanced, deterministic scene set: sample free poses, keep n_blocked with
    a wall inside BLOCKED_AHEAD_M and n_clear with >2 m of open floor ahead."""
    import random

    rng = random.Random(seed)
    blocked: list[dict[str, float | str | bool]] = []
    clear: list[dict[str, float | str | bool]] = []
    while (len(blocked) < n_blocked or len(clear) < n_clear) and rng.random() >= 0:
        x = rng.uniform(-7.0, 7.0)
        y = rng.uniform(-7.0, 7.0)
        yaw = rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2]) + rng.uniform(-0.3, 0.3)
        if not in_free_space(walls, exclusions, x, y):
            continue
        ahead = clearance(walls, x, y, yaw)
        pose: dict[str, float | str | bool] = {
            "x": round(x, 3),
            "y": round(y, 3),
            "yaw": round(yaw, 4),
            "true_ahead_m": round(ahead, 3),
            "blocked": ahead < BLOCKED_AHEAD_M,
        }
        if pose["blocked"] and len(blocked) < n_blocked and ahead > 0.45:
            pose["scene_id"] = f"blocked-{len(blocked) + 1:02d}"
            blocked.append(pose)
        elif not pose["blocked"] and ahead > 2.0 and len(clear) < n_clear:
            pose["scene_id"] = f"clear-{len(clear) + 1:02d}"
            clear.append(pose)
    return blocked + clear


def _self_check(walls: tuple[Wall, ...], exclusions: tuple[Wall, ...]) -> None:
    # Facing the north perimeter wall (y=+8, 0.2 thick) from (0, 7): 0.9 m away.
    assert abs(clearance(walls, 0.0, 7.0, math.pi / 2) - 0.9) < 1e-6
    # Facing away from it down the arena: several meters of maze, not MAX_RANGE.
    assert clearance(walls, 0.0, 7.0, -math.pi / 2) < MAX_RANGE_M
    # Inside a wall's inflated footprint is not free space.
    assert not in_free_space(walls, exclusions, 0.0, 8.0)
    assert in_free_space(walls, exclusions, 0.0, 7.0)


def main() -> None:
    walls, exclusions = parse_walls(sys.argv[1])
    _self_check(walls, exclusions)
    poses = select_poses(walls, exclusions)
    json.dump(poses, sys.stdout, indent=1)
    print(f"\n# {len(poses)} poses over {len(walls)} walls", file=sys.stderr)


if __name__ == "__main__":
    main()
