"""Planners — drive each agent toward its target tag using collective memory.

Three planners are provided, intended for ablation comparison:

  BaselineRandomPlanner
      Random valid actions; no memory consulted.

  GreedyMemoryPlanner
      Queries collective memory for the agent's target_tag.  Picks the
      top-1 concept's location and moves toward it.  Falls back to random
      when memory is empty.

  CoordinatedFieldPlanner
      Like GreedyMemoryPlanner, but uses a ``FieldAdapter`` in
      COORDINATED mode: before committing to a target, the planner
      reserves it.  Other agents querying the field then see the
      target's activation drop and select a different concept.

All planners run on top of a ``CollectiveMemory`` + ``ConceptRecallLayer``
(+ ``FieldAdapter``) shared across all agents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from songline_drive.collective_field_types import FieldMode
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.field_adapter import FieldAdapter
from songline_drive.semantic_field import SemanticField

from multiagent_env.grid_world import (
    DIR_DELTAS,
    FORWARD,
    NOOP,
    TURN_LEFT,
    TURN_RIGHT,
    MultiAgentGridWorld,
)


# ─────────────────────────────────────────────────────── helpers


def _dir_toward(from_xy: Tuple[int, int], to_xy: Tuple[int, int]) -> int:
    """Return the direction (0=E,1=S,2=W,3=N) of the dominant step toward target."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) >= abs(dy):
        return 0 if dx > 0 else 2  # east or west
    return 1 if dy > 0 else 3       # south or north


def _action_to_face(current_dir: int, target_dir: int, can_step_forward: bool) -> int:
    """Pick action to turn toward target_dir, then step.

    Returns one of TURN_LEFT, TURN_RIGHT, FORWARD.
    """
    if current_dir == target_dir:
        return FORWARD if can_step_forward else NOOP
    diff = (target_dir - current_dir) % 4
    if diff == 1:
        return TURN_RIGHT
    if diff == 3:
        return TURN_LEFT
    # 180° turn — choose left arbitrarily
    return TURN_LEFT


# ─────────────────────────────────────────────────────── observation publishing


def publish_observation_to_memory(
    collective: CollectiveMemory,
    agent_id: str,
    env_id: str,
    episode_id: int,
    step_idx: int,
    obs: Dict[str, Any],
    *,
    confidence: float = 1.0,
) -> int:
    """Publish all observed cells (with semantic tags) to the bus.

    Each visible cell becomes one ``place_observed`` event.  Returns the
    number of events published.
    """
    n = 0
    for cell in obs.get("cells", []):
        tag = cell["tag"]
        if tag in ("wall",):
            continue
        collective.publish_event(
            "place_observed", agent_id,
            episode_id=episode_id, step_idx=step_idx,
            env_id=env_id,
            payload={
                "place_key": list(cell["xy"]),
                "semantic_tags": {tag: 0.95},
                "node_freshness": 1.0,
            },
            confidence=confidence,
        )
        n += 1
    return n


# ─────────────────────────────────────────────────────── planners


class BaselineRandomPlanner:
    """Uniform random action.  No memory consulted."""

    name = "baseline_random"

    def __init__(self, rng_seed: int = 0) -> None:
        self.rng = np.random.default_rng(rng_seed)

    def choose(
        self,
        env: MultiAgentGridWorld,
        agent_id: str,
        collective: CollectiveMemory,
        adapter: Optional[FieldAdapter],
        *,
        current_seq: int,
        episode_id: int,
        step_idx: int,
    ) -> int:
        return int(self.rng.integers(0, 4))


class GreedyMemoryPlanner:
    """Picks the best memory candidate for target_tag and walks toward it."""

    name = "greedy_memory"

    def __init__(self, rng_seed: int = 0) -> None:
        self.rng = np.random.default_rng(rng_seed)

    def _query_target_xy(
        self,
        adapter: FieldAdapter,
        collective: CollectiveMemory,
        target_tag: str,
        agent_id: str,
        env_id: str,
        current_seq: int,
    ) -> Optional[Tuple[Optional[str], Tuple[float, float]]]:
        """Return (concept_id, centroid_xy) of best target, or None if no candidate."""
        items = adapter.field.top_k_for_channel(target_tag, k=5)
        for cid, _act in items:
            cell = adapter.field.cells.get(cid)
            if cell is None or cell.centroid_xy is None:
                continue
            return cid, (float(cell.centroid_xy[0]), float(cell.centroid_xy[1]))
        return None

    def choose(
        self,
        env: MultiAgentGridWorld,
        agent_id: str,
        collective: CollectiveMemory,
        adapter: Optional[FieldAdapter],
        *,
        current_seq: int,
        episode_id: int,
        step_idx: int,
    ) -> int:
        if adapter is None:
            return int(self.rng.integers(0, 4))
        ag = env.agents[agent_id]
        res = self._query_target_xy(
            adapter, collective, ag.target_tag, agent_id, "grid", current_seq,
        )
        if res is None:
            return int(self.rng.integers(0, 4))
        _cid, target_xy = res
        tx, ty = round(target_xy[0]), round(target_xy[1])
        if (tx, ty) == (ag.x, ag.y):
            return int(self.rng.integers(0, 4))

        target_dir = _dir_toward((ag.x, ag.y), (tx, ty))
        if ag.direction == target_dir:
            dx, dy = DIR_DELTAS[target_dir]
            nx, ny = ag.x + dx, ag.y + dy
            cell = env.cell(nx, ny)
            occupied = any(
                (other.x, other.y) == (nx, ny) and other.agent_id != agent_id
                for other in env.agents.values()
            )
            can_step = cell != 1 and not occupied  # 1 = WALL
            return _action_to_face(ag.direction, target_dir, can_step)
        return _action_to_face(ag.direction, target_dir, can_step_forward=True)


class CoordinatedFieldPlanner(GreedyMemoryPlanner):
    """Greedy + reserves the chosen concept so other agents are deflected."""

    name = "coordinated_field"

    def __init__(self, *args: Any, reservation_duration: int = 30, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reservation_duration = int(reservation_duration)
        self._committed_target: Dict[str, str] = {}  # agent_id -> concept_id

    def choose(
        self,
        env: MultiAgentGridWorld,
        agent_id: str,
        collective: CollectiveMemory,
        adapter: Optional[FieldAdapter],
        *,
        current_seq: int,
        episode_id: int,
        step_idx: int,
    ) -> int:
        if adapter is None or adapter.mode != FieldMode.COORDINATED:
            return super().choose(
                env, agent_id, collective, adapter,
                current_seq=current_seq, episode_id=episode_id, step_idx=step_idx,
            )

        ag = env.agents[agent_id]
        target_tag = ag.target_tag

        # Reserve our target if we haven't yet (or expired)
        if self._committed_target.get(agent_id) is None:
            items = adapter.field.top_k_for_channel(target_tag, k=3)
            if items:
                cid = items[0][0]
                adapter.commit_reservation(
                    agent_id=agent_id, concept_id=cid, channel=target_tag,
                    duration=self.reservation_duration, current_seq=current_seq,
                )
                self._committed_target[agent_id] = cid

        # Walk toward our committed concept (not the global top-1, which
        # may be different after our own reservation penalised it).
        committed_cid = self._committed_target.get(agent_id)
        if committed_cid is not None:
            cell = adapter.field.cells.get(committed_cid)
            if cell is not None and cell.centroid_xy is not None:
                tx, ty = round(cell.centroid_xy[0]), round(cell.centroid_xy[1])
                if (tx, ty) == (ag.x, ag.y):
                    return int(self.rng.integers(0, 4))
                target_dir = _dir_toward((ag.x, ag.y), (tx, ty))
                if ag.direction == target_dir:
                    dx, dy = DIR_DELTAS[target_dir]
                    nx, ny = ag.x + dx, ag.y + dy
                    occupied = any(
                        (o.x, o.y) == (nx, ny) and o.agent_id != agent_id
                        for o in env.agents.values()
                    )
                    can_step = env.cell(nx, ny) != 1 and not occupied
                    return _action_to_face(ag.direction, target_dir, can_step)
                return _action_to_face(ag.direction, target_dir, can_step_forward=True)

        return super().choose(
            env, agent_id, collective, adapter,
            current_seq=current_seq, episode_id=episode_id, step_idx=step_idx,
        )

    def release_all(self, adapter: FieldAdapter) -> None:
        for aid, cid in list(self._committed_target.items()):
            adapter.release_reservation(cid, aid)
        self._committed_target.clear()
