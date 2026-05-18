"""Minimal multi-agent grid world for collective memory experiments.

Designed to be cheaper than wrapping MiniGrid / BabyAI for multi-agent
(those environments are single-agent at the framework level) while still
providing the structure the songline_drive memory layer expects:

  - Discrete grid with cells tagged semantically (water/hazard/goal/empty/wall)
  - Multiple agents, each with (x, y, direction)
  - Per-step observations of nearby cells with semantic tags
  - Episode terminates when all agents reach their targets OR step_limit hit

Cells
-----
``EMPTY`` 0, ``WALL`` 1, ``WATER`` 2, ``HAZARD`` 3, ``GOAL`` 4

Actions per agent
-----------------
``TURN_LEFT`` 0, ``TURN_RIGHT`` 1, ``FORWARD`` 2, ``NOOP`` 3

Directions
----------
0 = east (+x), 1 = south (+y), 2 = west (-x), 3 = north (-y)

Agents never share a cell.  If two agents try to step into the same cell
simultaneously, neither moves.  An agent attempting to step into a wall
or off-grid does not move.

This env is **not** registered with gymnasium because it has a
multi-discrete action space that the gym API does not natively support
for two simultaneous agents in the same world.  Call ``step(actions)``
directly with a dict of ``{agent_id: action_int}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Cell types
EMPTY = 0
WALL = 1
WATER = 2
HAZARD = 3
GOAL = 4

CELL_TAGS = {
    EMPTY: "safe_neutral",
    WATER: "water_source",
    HAZARD: "hazard_edge",
    GOAL: "goal_region",
}

# Actions
TURN_LEFT = 0
TURN_RIGHT = 1
FORWARD = 2
NOOP = 3

# Direction deltas
DIR_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (1, 0),   # east
    1: (0, 1),   # south
    2: (-1, 0),  # west
    3: (0, -1),  # north
}


@dataclass
class AgentState:
    agent_id: str
    x: int
    y: int
    direction: int = 0
    target_tag: str = "water_source"
    success: bool = False
    n_steps: int = 0
    n_hazard_hits: int = 0


@dataclass
class StepInfo:
    """Single-step transition info per agent."""

    agent_id: str
    moved: bool
    new_xy: Tuple[int, int]
    cell_tag: str
    blocked_by: Optional[str] = None  # "wall", "agent:other_id", "boundary"


@dataclass
class StepResult:
    obs: Dict[str, Dict[str, Any]]   # per-agent observation
    info: Dict[str, StepInfo]
    rewards: Dict[str, float]
    done: bool
    all_succeeded: bool


class MultiAgentGridWorld:
    """Minimal multi-agent gridworld.

    Parameters
    ----------
    width, height : int
        Grid dimensions.
    grid : np.ndarray, optional
        Pre-built ``height × width`` integer grid of cell types.  If None,
        an empty grid is created and you can scatter cells via
        ``set_cell()``.
    step_limit : int
        Max steps per episode.
    observation_radius : int
        Manhattan radius of the local semantic observation.
    """

    def __init__(
        self,
        *,
        width: int = 10,
        height: int = 10,
        grid: Optional[np.ndarray] = None,
        step_limit: int = 200,
        observation_radius: int = 2,
        rng_seed: int = 0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        if grid is not None:
            assert grid.shape == (self.height, self.width), (
                f"Expected grid shape ({self.height}, {self.width}), got {grid.shape}"
            )
            self.grid = grid.astype(np.int32).copy()
        else:
            self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.step_limit = int(step_limit)
        self.observation_radius = int(observation_radius)
        self.rng = np.random.default_rng(rng_seed)

        self._agents: Dict[str, AgentState] = {}
        self._episode_step = 0

    # ──────────────────────────────────────────────────── cell helpers

    def set_cell(self, x: int, y: int, value: int) -> None:
        self.grid[y, x] = value

    def cell(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.grid[y, x])
        return WALL  # off-grid = wall

    def cell_tag(self, x: int, y: int) -> str:
        return CELL_TAGS.get(self.cell(x, y), "wall")

    # ──────────────────────────────────────────────────── agents

    def spawn(
        self,
        agent_id: str,
        *,
        start_xy: Tuple[int, int],
        target_tag: str = "water_source",
        direction: int = 0,
    ) -> AgentState:
        if agent_id in self._agents:
            raise ValueError(f"agent already spawned: {agent_id}")
        x, y = start_xy
        assert 0 <= x < self.width and 0 <= y < self.height
        # Don't allow spawn on wall or on another agent
        if self.cell(x, y) == WALL:
            raise ValueError(f"cannot spawn on wall: {start_xy}")
        for other in self._agents.values():
            if (other.x, other.y) == (x, y):
                raise ValueError(f"cell occupied by {other.agent_id}")
        ag = AgentState(
            agent_id=agent_id, x=x, y=y, direction=direction, target_tag=target_tag,
        )
        self._agents[agent_id] = ag
        return ag

    @property
    def agents(self) -> Dict[str, AgentState]:
        return self._agents

    @property
    def episode_step(self) -> int:
        return self._episode_step

    # ──────────────────────────────────────────────────── step

    def step(self, actions: Dict[str, int]) -> StepResult:
        """Apply one joint action.  ``actions`` maps agent_id → action int."""
        self._episode_step += 1

        # Phase 1: resolve turns and decide proposed moves
        proposals: Dict[str, Tuple[int, int]] = {}  # agent_id -> proposed_xy
        infos: Dict[str, StepInfo] = {}

        for aid, ag in self._agents.items():
            if ag.success:
                infos[aid] = StepInfo(aid, moved=False, new_xy=(ag.x, ag.y),
                                      cell_tag=self.cell_tag(ag.x, ag.y))
                continue
            a = int(actions.get(aid, NOOP))
            if a == TURN_LEFT:
                ag.direction = (ag.direction - 1) % 4
                infos[aid] = StepInfo(aid, moved=False, new_xy=(ag.x, ag.y),
                                      cell_tag=self.cell_tag(ag.x, ag.y))
            elif a == TURN_RIGHT:
                ag.direction = (ag.direction + 1) % 4
                infos[aid] = StepInfo(aid, moved=False, new_xy=(ag.x, ag.y),
                                      cell_tag=self.cell_tag(ag.x, ag.y))
            elif a == FORWARD:
                dx, dy = DIR_DELTAS[ag.direction]
                nx, ny = ag.x + dx, ag.y + dy
                proposals[aid] = (nx, ny)
            else:
                infos[aid] = StepInfo(aid, moved=False, new_xy=(ag.x, ag.y),
                                      cell_tag=self.cell_tag(ag.x, ag.y))

        # Phase 2: validate proposals (walls, boundaries, other agents)
        occupied_now = {(ag.x, ag.y): aid for aid, ag in self._agents.items()
                        if aid not in proposals}
        accepted: Dict[str, Tuple[int, int]] = {}

        # Detect conflicts: two agents proposing same cell → neither moves
        target_counts: Dict[Tuple[int, int], int] = {}
        for xy in proposals.values():
            target_counts[xy] = target_counts.get(xy, 0) + 1

        for aid, (nx, ny) in proposals.items():
            ag = self._agents[aid]
            # Boundary or wall
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                infos[aid] = StepInfo(aid, moved=False, new_xy=(ag.x, ag.y),
                                      cell_tag=self.cell_tag(ag.x, ag.y),
                                      blocked_by="boundary")
                continue
            if self.cell(nx, ny) == WALL:
                infos[aid] = StepInfo(aid, moved=False, new_xy=(ag.x, ag.y),
                                      cell_tag=self.cell_tag(ag.x, ag.y),
                                      blocked_by="wall")
                continue
            # Currently occupied by stationary agent
            if (nx, ny) in occupied_now:
                infos[aid] = StepInfo(aid, moved=False, new_xy=(ag.x, ag.y),
                                      cell_tag=self.cell_tag(ag.x, ag.y),
                                      blocked_by=f"agent:{occupied_now[(nx, ny)]}")
                continue
            # Conflict: another agent proposes the same cell
            if target_counts[(nx, ny)] > 1:
                infos[aid] = StepInfo(aid, moved=False, new_xy=(ag.x, ag.y),
                                      cell_tag=self.cell_tag(ag.x, ag.y),
                                      blocked_by="agent_conflict")
                continue
            accepted[aid] = (nx, ny)

        for aid, (nx, ny) in accepted.items():
            ag = self._agents[aid]
            ag.x, ag.y = nx, ny
            tag = self.cell_tag(nx, ny)
            infos[aid] = StepInfo(aid, moved=True, new_xy=(nx, ny), cell_tag=tag)

        # Phase 3: rewards / success tracking
        rewards: Dict[str, float] = {}
        for aid, ag in self._agents.items():
            if ag.success:
                rewards[aid] = 0.0
                continue
            ag.n_steps += 1
            r = -0.01  # step cost
            cell_tag = self.cell_tag(ag.x, ag.y)
            if cell_tag == "hazard_edge":
                r -= 0.5
                ag.n_hazard_hits += 1
            if cell_tag == ag.target_tag:
                r += 1.0
                ag.success = True
            rewards[aid] = r

        # Observation
        obs = {aid: self._observation(aid) for aid in self._agents}

        all_succeeded = all(ag.success for ag in self._agents.values())
        done = all_succeeded or self._episode_step >= self.step_limit

        return StepResult(obs=obs, info=infos, rewards=rewards,
                          done=done, all_succeeded=all_succeeded)

    # ──────────────────────────────────────────────────── observation

    def _observation(self, agent_id: str) -> Dict[str, Any]:
        ag = self._agents[agent_id]
        cells: List[Dict[str, Any]] = []
        r = self.observation_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) > r:
                    continue
                cx, cy = ag.x + dx, ag.y + dy
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    cv = int(self.grid[cy, cx])
                    cells.append({"xy": (cx, cy), "value": cv,
                                  "tag": CELL_TAGS.get(cv, "wall")})
        return {
            "agent_id": agent_id,
            "xy": (ag.x, ag.y),
            "direction": ag.direction,
            "target_tag": ag.target_tag,
            "success": ag.success,
            "cells": cells,
        }

    # ──────────────────────────────────────────────────── render

    def render_ascii(self) -> str:
        """Return an ASCII view of the grid + agent positions."""
        symbols = {EMPTY: ".", WALL: "#", WATER: "~", HAZARD: "X", GOAL: "G"}
        rows = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                placed = False
                for ag in self._agents.values():
                    if (ag.x, ag.y) == (x, y):
                        row += ag.agent_id[-1].upper()
                        placed = True
                        break
                if not placed:
                    row += symbols.get(int(self.grid[y, x]), "?")
            rows.append(row)
        return "\n".join(rows)
