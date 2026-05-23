"""Parameterised environment factory for the big sweep.

Given (N agents, M waters, layout, hazard_density, seed), returns a
``MultiAgentGridWorld`` with agents spawned, water cells placed and
hazards scattered.  All placements are deterministic given the seed.

Layout strategies
-----------------
``symmetric``
    Each water cell is placed roughly in front of one agent's natural
    exploration direction.  Solo agents can find a water without help.
``asymmetric``
    Each water is placed OPPOSITE to an agent's natural direction.  Solo
    discovery is hard; collective memory helps a lot.
``random``
    Waters placed uniformly at random (respecting "no water within
    observation radius of any agent at t=0").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from multiagent_env import (
    HAZARD, MultiAgentGridWorld, WATER,
)

# Default grid; can be overridden for stress tests
GRID_W, GRID_H = 12, 10

# Agent palette — extend if you ever need >8 agents
AGENT_COLORS = [
    "#e74c3c", "#3498db", "#27ae60", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

# Eight evenly spread spawn positions (corners + edge midpoints)
SPAWN_POSITIONS_8 = [
    (0, 0),                    # NW corner
    (GRID_W - 1, 0),           # NE corner
    (0, GRID_H - 1),           # SW corner
    (GRID_W - 1, GRID_H - 1),  # SE corner
    (GRID_W // 2, 0),          # N midpoint
    (GRID_W - 1, GRID_H // 2), # E midpoint
    (GRID_W // 2, GRID_H - 1), # S midpoint
    (0, GRID_H // 2),          # W midpoint
]

# Natural starting direction so the first FORWARD doesn't immediately leave the grid
SPAWN_DIRECTIONS_8 = [
    0,   # NW → east
    2,   # NE → west
    3,   # SW → north
    3,   # SE → north
    1,   # N midpoint → south
    2,   # E midpoint → west
    3,   # S midpoint → north
    0,   # W midpoint → east
]


def _stable_rng(seed_namespace: str, seed: int) -> np.random.Generator:
    key = f"{seed_namespace}|{seed}".encode()
    return np.random.default_rng(int(hashlib.md5(key).hexdigest()[:8], 16))


def _candidate_water_positions() -> List[Tuple[int, int]]:
    """Grid cells that are not used for agent spawns, suitable for waters."""
    spawns = set(SPAWN_POSITIONS_8)
    return [(x, y) for x in range(GRID_W) for y in range(GRID_H)
            if (x, y) not in spawns]


def _opposite_quadrant(spawn_xy: Tuple[int, int]) -> Tuple[int, int]:
    """Map a spawn position to the centre of the opposite quadrant."""
    sx, sy = spawn_xy
    # Mirror around grid centre, then snap into a quadrant interior
    mx = GRID_W - 1 - sx
    my = GRID_H - 1 - sy
    # Pull a bit inside (avoid the very edge)
    mx = max(1, min(GRID_W - 2, mx))
    my = max(1, min(GRID_H - 2, my))
    return (mx, my)


def _front_quadrant(spawn_xy: Tuple[int, int]) -> Tuple[int, int]:
    """Map spawn to a point roughly in its natural exploration direction."""
    sx, sy = spawn_xy
    cx, cy = GRID_W // 2, GRID_H // 2
    # Move 30% of the way from spawn to the centre
    fx = sx + (cx - sx) // 2
    fy = sy + (cy - sy) // 2
    return (max(1, min(GRID_W - 2, fx)),
            max(1, min(GRID_H - 2, fy)))


def _pick_water_positions(
    n_agents: int, n_waters: int, layout: str, seed: int,
    obs_radius: int,
    spawn_positions: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Return ``n_waters`` placements satisfying invariants:
    - No water within ``obs_radius`` of any spawn (so agents don't see at t=0)
    - All waters distinct
    """
    rng = _stable_rng(f"waters_{layout}_{n_agents}_{n_waters}", seed)
    candidates = _candidate_water_positions()

    def too_close(xy: Tuple[int, int]) -> bool:
        for sx, sy in spawn_positions:
            if abs(xy[0] - sx) + abs(xy[1] - sy) <= obs_radius:
                return True
        return False

    valid = [xy for xy in candidates if not too_close(xy)]

    if layout == "random":
        idx = rng.choice(len(valid), size=n_waters, replace=False)
        return [valid[i] for i in idx]

    if layout == "symmetric":
        # Place each water near one agent's natural exploration target.
        # If fewer waters than agents, pick the first M agents.
        chosen: List[Tuple[int, int]] = []
        used = set()
        for i in range(n_waters):
            spawn_i = spawn_positions[i % n_agents]
            target = _front_quadrant(spawn_i)
            # Find the nearest valid candidate
            best = min(valid, key=lambda xy:
                       abs(xy[0] - target[0]) + abs(xy[1] - target[1])
                       + (10000 if xy in used else 0))
            chosen.append(best)
            used.add(best)
        return chosen

    if layout == "asymmetric":
        chosen = []
        used = set()
        for i in range(n_waters):
            spawn_i = spawn_positions[i % n_agents]
            target = _opposite_quadrant(spawn_i)
            best = min(valid, key=lambda xy:
                       abs(xy[0] - target[0]) + abs(xy[1] - target[1])
                       + (10000 if xy in used else 0))
            chosen.append(best)
            used.add(best)
        return chosen

    raise ValueError(f"Unknown layout: {layout}")


def _pick_hazards(
    hazard_density: float, seed: int,
    spawn_positions: List[Tuple[int, int]],
    water_positions: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Scatter hazards on cells that are not spawns or waters."""
    rng = _stable_rng(f"hazards_{hazard_density}", seed)
    n_hazards = int(round(hazard_density * GRID_W * GRID_H))
    if n_hazards <= 0:
        return []
    occupied = set(spawn_positions) | set(water_positions)
    candidates = [(x, y) for x in range(GRID_W) for y in range(GRID_H)
                  if (x, y) not in occupied]
    if n_hazards > len(candidates):
        n_hazards = len(candidates)
    idx = rng.choice(len(candidates), size=n_hazards, replace=False)
    return [candidates[i] for i in idx]


@dataclass
class BuiltEnv:
    env: MultiAgentGridWorld
    agent_ids: List[str]
    agent_colors: Dict[str, str]
    water_positions: List[Tuple[int, int]]
    hazard_positions: List[Tuple[int, int]]


def build_env(
    *,
    n_agents: int,
    n_waters: int,
    layout: str,
    hazard_density: float,
    seed: int,
    step_limit: int = 200,
    observation_radius: int = 2,
) -> BuiltEnv:
    assert 2 <= n_agents <= 8, "supported range 2..8 agents"
    assert 1 <= n_waters <= n_agents, "M must satisfy 1 <= M <= N"
    assert layout in ("symmetric", "asymmetric", "random")

    spawn_positions = SPAWN_POSITIONS_8[:n_agents]
    spawn_directions = SPAWN_DIRECTIONS_8[:n_agents]
    agent_ids = [f"agent-{chr(ord('A') + i)}" for i in range(n_agents)]
    agent_colors = {aid: AGENT_COLORS[i] for i, aid in enumerate(agent_ids)}

    water_positions = _pick_water_positions(
        n_agents, n_waters, layout, seed,
        obs_radius=observation_radius,
        spawn_positions=spawn_positions,
    )
    hazard_positions = _pick_hazards(
        hazard_density, seed, spawn_positions, water_positions,
    )

    env = MultiAgentGridWorld(
        width=GRID_W, height=GRID_H, step_limit=step_limit,
        observation_radius=observation_radius, rng_seed=seed,
    )
    for x, y in water_positions:
        env.set_cell(x, y, WATER)
    for x, y in hazard_positions:
        env.set_cell(x, y, HAZARD)
    for aid, xy, d in zip(agent_ids, spawn_positions, spawn_directions):
        env.spawn(aid, start_xy=xy, target_tag="water_source", direction=d)

    return BuiltEnv(
        env=env, agent_ids=agent_ids, agent_colors=agent_colors,
        water_positions=water_positions, hazard_positions=hazard_positions,
    )
