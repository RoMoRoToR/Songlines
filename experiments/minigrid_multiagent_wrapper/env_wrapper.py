"""Multi-agent water-search on a MiniGrid substrate (FourRooms layout).

Task 1 (portability) — closes the "two-substrate seam" by exposing the
multi-agent scarcity scenario as a MiniGrid-derived env, so single-agent
and multi-agent claims live in the same ecosystem.

Design: this wrapper *constructs* a FourRooms-style layout using
MiniGrid's grid primitives (Grid + Wall + Floor + Lava), then transfers
the layout to a custom-env-compatible MultiAgentGridWorld. All step/observation/
success logic comes from the custom MultiAgentGridWorld, so the existing
experiments/big_experiment/runner.py and the Q/R/M/C logger run
unchanged.

Acceptance criteria for §7.7 "Portability to standard substrate":

  (a) Q/R/M/C events extractable with identical operational definitions
      used in experiments/big_experiment (Targets_i / Locked_i / W
      ground truth, eps = 0.6 cell-distance).
  (b) Bottleneck-shift slope signs of Proposition 3 hold:
      Spearman(P(M*|R*), K) < 0, Spearman(P(C*|M*), K) > 0,
      both p < 0.05 at 20 seeds x 5 cadences.
  (c) Mean t_succ has interior minimum strictly below both endpoints.
"""

from __future__ import annotations

import dataclasses as dc
import random
from typing import List, Tuple

import numpy as np
from minigrid.core.grid import Grid as _MGGrid
from minigrid.core.world_object import Floor as _MGFloor
from minigrid.core.world_object import Lava as _MGLava
from minigrid.core.world_object import Wall as _MGWall

from multiagent_env import EMPTY, GOAL, HAZARD, WALL, WATER
from multiagent_env.grid_world import MultiAgentGridWorld


GridXY = Tuple[int, int]


def _build_fourrooms_minigrid(width: int, height: int,
                              rng: random.Random) -> _MGGrid:
    """Build a FourRooms layout with MiniGrid primitives."""
    g = _MGGrid(width, height)
    # surrounding walls
    g.horz_wall(0, 0)
    g.horz_wall(0, height - 1)
    g.vert_wall(0, 0)
    g.vert_wall(width - 1, 0)
    # mid walls + doorways
    mid_x = width // 2
    mid_y = height // 2
    g.vert_wall(mid_x, 0, height)
    g.horz_wall(0, mid_y, width)
    # one doorway in each mid-wall quadrant for connectivity
    g.set(mid_x, rng.randint(1, mid_y - 1), None)
    g.set(mid_x, rng.randint(mid_y + 1, height - 2), None)
    g.set(rng.randint(1, mid_x - 1), mid_y, None)
    g.set(rng.randint(mid_x + 1, width - 2), mid_y, None)
    return g


def _minigrid_to_numpy(g: _MGGrid) -> np.ndarray:
    """Convert a MiniGrid Grid to the custom-env numpy cell-value array.

    Cell value encoding (custom-env convention):
        EMPTY=0, WALL=1, WATER=2, HAZARD=3, GOAL=4.
    """
    w, h = g.width, g.height
    arr = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            cell = g.get(x, y)
            if isinstance(cell, _MGWall):
                arr[y, x] = WALL
            elif isinstance(cell, _MGFloor) and cell.color == "blue":
                arr[y, x] = WATER
            elif isinstance(cell, _MGLava):
                arr[y, x] = HAZARD
            else:
                arr[y, x] = EMPTY
    return arr


# ── public factory matching env_factory.build_env signature ────────────


@dc.dataclass
class Built:
    """Compatibility namespace matching env_factory.Built."""
    env: MultiAgentGridWorld
    agent_ids: List[str]
    water_positions: List[GridXY]


def build_minigrid_env(*, n_agents: int, n_waters: int,
                       layout: str = "fourrooms",
                       grid_size: int = 17,
                       hazard_density: float = 0.05,
                       seed: int = 0,
                       step_limit: int = 120,
                       observation_radius: int = 2) -> Built:
    """Drop-in replacement for env_factory.build_env on MiniGrid layout.

    Returns a `Built` namespace matching env_factory.Built so the
    existing peer-architecture runner (experiments/big_experiment/runner.py)
    can read this env without code changes.
    """
    if layout != "fourrooms":
        raise NotImplementedError(
            f"Layout '{layout}' not supported in this wrapper. "
            f"FourRooms is the canonical MiniGrid-portability substrate."
        )
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # 1. Build the layout with MiniGrid primitives.
    mg = _build_fourrooms_minigrid(grid_size, grid_size, rng)

    # 2. Pick water cells from open floor area.
    open_cells: List[GridXY] = []
    for y in range(grid_size):
        for x in range(grid_size):
            if mg.get(x, y) is None:
                open_cells.append((x, y))
    rng.shuffle(open_cells)
    water_positions: List[GridXY] = []
    for _ in range(n_waters):
        x, y = open_cells.pop()
        mg.set(x, y, _MGFloor(color="blue"))
        water_positions.append((x, y))

    # 3. Sprinkle hazard cells (Lava in MiniGrid terms).
    n_hazard = int(round(hazard_density * len(open_cells)))
    for _ in range(n_hazard):
        x, y = open_cells.pop()
        mg.set(x, y, _MGLava())

    # 4. Transfer layout to the custom-env MultiAgentGridWorld; pass remaining
    # open cells as agent spawn positions.
    grid = _minigrid_to_numpy(mg)
    env = MultiAgentGridWorld(
        width=grid_size, height=grid_size, grid=grid,
        step_limit=step_limit, observation_radius=observation_radius,
        rng_seed=seed,
    )

    agent_ids: List[str] = []
    for i in range(n_agents):
        if not open_cells:
            raise RuntimeError(
                f"Not enough open cells for {n_agents} agents in a "
                f"{grid_size}x{grid_size} FourRooms layout."
            )
        x, y = open_cells.pop()
        aid = f"a{i}"
        env.spawn(
            aid, start_xy=(x, y),
            target_tag="water_source",
            direction=int(np_rng.integers(0, 4)),
        )
        agent_ids.append(aid)

    return Built(env=env, agent_ids=agent_ids,
                 water_positions=water_positions)
