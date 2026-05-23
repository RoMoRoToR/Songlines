"""Universal memory-driven planner (extracted from exp_4way_walk).

Same code for every architecture; only the memory_targets list differs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from multiagent_env import FORWARD, NOOP, TURN_LEFT, TURN_RIGHT, WALL
from multiagent_env.grid_world import DIR_DELTAS


@dataclass
class PlannerState:
    agent_id: str
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    locked_target: Optional[Tuple[int, int]] = None


def _stable_rng(agent_id: str, tick: int, variant: str) -> np.random.Generator:
    key = f"{agent_id}|{tick}|{variant}".encode()
    return np.random.default_rng(int(hashlib.md5(key).hexdigest()[:8], 16))


def _direction_toward(from_xy, to_xy) -> int:
    dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
    if abs(dx) >= abs(dy):
        return 0 if dx > 0 else 2
    return 1 if dy > 0 else 3


def _can_step(env, ag, target_xy: Tuple[int, int]) -> bool:
    nx, ny = target_xy
    if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height:
        return False
    if env.cell(nx, ny) == WALL:
        return False
    for other in env.agents.values():
        if other.agent_id != ag.agent_id and (other.x, other.y) == (nx, ny):
            return False
    return True


def _turn_or_forward(ag_direction: int, target_dir: int) -> int:
    if ag_direction == target_dir:
        return FORWARD
    diff = (target_dir - ag_direction) % 4
    if diff == 1:
        return TURN_RIGHT
    if diff == 3:
        return TURN_LEFT
    return TURN_LEFT


def plan_action(
    state: PlannerState,
    env,
    memory_targets: List[Tuple[float, float]],
    tick: int,
    variant: str,
) -> int:
    ag = env.agents[state.agent_id]
    state.visited.add((ag.x, ag.y))

    if ag.success:
        return NOOP

    # ── Tier 1: navigate toward a known, unoccupied water target
    if memory_targets:
        occupied = set()
        for other_aid, other in env.agents.items():
            if other_aid != state.agent_id and other.success:
                occupied.add((other.x, other.y))

        if state.locked_target in occupied:
            state.locked_target = None

        def dist(xy):
            return abs(xy[0] - ag.x) + abs(xy[1] - ag.y)

        candidates = sorted(memory_targets, key=dist)
        chosen: Optional[Tuple[int, int]] = None
        for cand in candidates:
            tx, ty = int(round(cand[0])), int(round(cand[1]))
            if (tx, ty) in occupied:
                continue
            chosen = (tx, ty)
            break

        if chosen is not None:
            tx, ty = chosen
            if state.locked_target is not None and state.locked_target not in occupied:
                lock_d = abs(state.locked_target[0] - ag.x) + abs(state.locked_target[1] - ag.y)
                new_d = abs(tx - ag.x) + abs(ty - ag.y)
                if lock_d - new_d < 2:
                    tx, ty = state.locked_target
            state.locked_target = (tx, ty)

            if (tx, ty) == (ag.x, ag.y):
                return NOOP
            target_dir = _direction_toward((ag.x, ag.y), (tx, ty))
            if ag.direction == target_dir:
                dx, dy = DIR_DELTAS[target_dir]
                if _can_step(env, ag, (ag.x + dx, ag.y + dy)):
                    return FORWARD
            else:
                return _turn_or_forward(ag.direction, target_dir)

    # ── Tier 2: explore — prefer unvisited adjacent cells
    state.locked_target = None
    fwd_dir = ag.direction
    fwd_dx, fwd_dy = DIR_DELTAS[fwd_dir]
    fwd_xy = (ag.x + fwd_dx, ag.y + fwd_dy)
    if _can_step(env, ag, fwd_xy) and fwd_xy not in state.visited:
        return FORWARD

    for try_dir in [(ag.direction + 1) % 4, (ag.direction + 3) % 4,
                    (ag.direction + 2) % 4]:
        dx, dy = DIR_DELTAS[try_dir]
        check_xy = (ag.x + dx, ag.y + dy)
        if _can_step(env, ag, check_xy) and check_xy not in state.visited:
            return _turn_or_forward(ag.direction, try_dir)

    # ── Tier 3: fallback
    if _can_step(env, ag, fwd_xy):
        return FORWARD
    rng = _stable_rng(state.agent_id, tick, variant)
    return int(rng.integers(0, 2))
