"""Phase 1 scout/consumer smoke-bench for the collective memory substrate.

Two synthetic agents share a single ``CollectiveMemory``:

* ``scout`` explores a small grid world and publishes ``place_observed``
  events for water- and hazard-like places.
* ``consumer`` arrives later, thirsty, and queries the substrate for a
  water target. Whether it ever uses a place that only ``scout`` saw
  is the headline signal that the collective layer is doing its job.

The script is deliberately detached from MiniGrid / MiniWorld / BabyAI:
no environment imports, no rendering, fully deterministic from
``--seed``. It exercises every public surface of the new package and
emits a full eight-metric report.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_collective.py \\
        --seed 0 \\
        --out_dir tmp/multiagent_collective_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

# Importing only the new package keeps this script independent of the
# existing single-agent stack.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_metrics import all_metrics
from songline_drive.collective_types import AgentSignature, CollectiveQuery
from songline_drive.multiagent_runtime import MultiAgentRuntime


GRID_W = 8
GRID_H = 6
ENV_ID = "synthetic-grid-8x6"
WATER_CONCEPT = "water_source"
HAZARD_CONCEPT = "hazard_edge"


@dataclass
class GroundTruthWorld:
    water_places: Set[Tuple[int, int]] = field(default_factory=set)
    hazard_places: Set[Tuple[int, int]] = field(default_factory=set)
    safe_places: Set[Tuple[int, int]] = field(default_factory=set)

    def tag_for(self, place: Tuple[int, int]) -> str:
        if place in self.water_places:
            return WATER_CONCEPT
        if place in self.hazard_places:
            return HAZARD_CONCEPT
        return "safe_neutral"


def build_world(rng: random.Random) -> GroundTruthWorld:
    cells = [(x, y) for x in range(GRID_W) for y in range(GRID_H)]
    rng.shuffle(cells)
    water = set(cells[:3])
    hazard = set(cells[3:6])
    safe = set(cells[6:]) - water - hazard
    return GroundTruthWorld(water_places=water, hazard_places=hazard, safe_places=safe)


def scout_walk(rng: random.Random, world: GroundTruthWorld, n_steps: int = 24) -> List[Tuple[int, int]]:
    """Scout meanders biased toward unvisited cells. Pure synthetic walk."""
    visited: List[Tuple[int, int]] = []
    pos = (rng.randrange(GRID_W), rng.randrange(GRID_H))
    visited.append(pos)
    for _ in range(n_steps - 1):
        candidates = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                candidates.append((nx, ny))
        rng.shuffle(candidates)
        candidates.sort(key=lambda p: 1 if p in visited else 0)
        pos = candidates[0] if candidates else pos
        visited.append(pos)
    return visited


def noisy_semantic_tags(
    rng: random.Random,
    world: GroundTruthWorld,
    place: Tuple[int, int],
    *,
    sensor_noise: float = 0.1,
) -> Dict[str, float]:
    """Return a noisy semantic-tag profile resembling what ``scene_encoder``
    would emit on a real grid cell."""
    tags: Dict[str, float] = {}
    if place in world.water_places:
        # water_source is the authoritative place-tag (matches scene_encoder
        # semantics); water_visible / near_water are weaker local cues.
        tags[WATER_CONCEPT] = max(0.0, 0.95 - rng.uniform(0, sensor_noise))
        tags["water_candidate"] = max(0.0, 0.7 - rng.uniform(0, sensor_noise))
        tags["water_visible"] = max(0.0, 0.6 - rng.uniform(0, sensor_noise))
        tags["near_water"] = max(0.0, 0.5 - rng.uniform(0, sensor_noise))
    elif place in world.hazard_places:
        tags[HAZARD_CONCEPT] = max(0.0, 0.9 - rng.uniform(0, sensor_noise))
        tags["adjacent_hazard"] = max(0.0, 0.55 - rng.uniform(0, sensor_noise))
    else:
        tags["safe_neutral"] = max(0.0, 0.6 - rng.uniform(0, sensor_noise))
        tags["corridor"] = max(0.0, 0.3 - rng.uniform(0, sensor_noise))
    # very small false-positive signal so belief fusion has something to chew on
    if rng.random() < 0.05:
        tags[WATER_CONCEPT] = tags.get(WATER_CONCEPT, 0.0) + rng.uniform(0.05, 0.15)
    return tags


def neighbors_of(place: Tuple[int, int]) -> List[Tuple[int, int]]:
    nbs = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = place[0] + dx, place[1] + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            nbs.append((nx, ny))
    return nbs


def run_scout_phase(
    runtime: MultiAgentRuntime,
    world: GroundTruthWorld,
    rng: random.Random,
    *,
    scout_id: str,
    episodes: int,
    steps_per_episode: int,
) -> Dict[str, object]:
    visited_total: List[Tuple[int, int]] = []
    successes = 0
    for episode_id in range(episodes):
        path = scout_walk(rng, world, n_steps=steps_per_episode)
        ep_visited: List[Tuple[int, int]] = []
        prev = None
        found_water_step = None
        for step_idx, place in enumerate(path):
            tags = noisy_semantic_tags(rng, world, place)
            runtime.publish_observation(
                agent_id=scout_id,
                episode_id=episode_id,
                step_idx=step_idx,
                env_id=ENV_ID,
                place_key=place,
                semantic_tags=tags,
                node_freshness=1.0,
                confidence=1.0,
            )
            if prev is not None and prev != place:
                runtime.publish_transition(
                    agent_id=scout_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    env_id=ENV_ID,
                    src_key=prev,
                    dst_key=place,
                    confidence=0.9,
                )
            if place in world.hazard_places:
                runtime.report_hazard_change(
                    agent_id=scout_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    env_id=ENV_ID,
                    place_key=place,
                    confidence=0.9,
                )
            if place in world.water_places and found_water_step is None:
                runtime.confirm_concept(
                    agent_id=scout_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    env_id=ENV_ID,
                    place_key=place,
                    concept_tag=WATER_CONCEPT,
                    confidence=1.0,
                )
                found_water_step = step_idx
            ep_visited.append(place)
            prev = place
        terminal = ep_visited[-1] if ep_visited else None
        success = bool(found_water_step is not None)
        successes += int(success)
        runtime.consolidate_episode(
            agent_id=scout_id,
            episode_id=episode_id,
            env_id=ENV_ID,
            outcome={
                "success": float(success),
                "target_concept_tag": WATER_CONCEPT,
                "terminal_place_key": list(terminal) if terminal else [],
                "found_water_at_step": found_water_step,
            },
            visited_place_keys=ep_visited,
        )
        visited_total.extend(ep_visited)
    return {
        "scout_episodes": episodes,
        "scout_success_count": successes,
        "scout_visited_unique_cells": len({tuple(p) for p in visited_total}),
    }


def run_consumer_phase(
    runtime: MultiAgentRuntime,
    world: GroundTruthWorld,
    rng: random.Random,
    *,
    consumer_id: str,
    scout_id: str,
    episodes: int,
    steps_per_episode: int,
) -> List[Dict[str, object]]:
    episode_outcomes: List[Dict[str, object]] = []
    for ep_offset in range(episodes):
        episode_id = 1000 + ep_offset
        pos = (rng.randrange(GRID_W), rng.randrange(GRID_H))
        visited: List[Tuple[int, int]] = [pos]
        query = CollectiveQuery(
            intent_type="find_water_source",
            target_tag=WATER_CONCEPT,
            score_weights={WATER_CONCEPT: 0.65, "water_candidate": 0.25, "near_water": 0.15},
            penalty_weights={HAZARD_CONCEPT: 0.3, "adjacent_hazard": 0.2},
            requesting_agent_id=consumer_id,
            env_id=ENV_ID,
            exclude_self=False,
            min_supporting_agents=1,
            min_fused_score=0.05,
        )
        results = runtime.query(consumer_id, query, top_k=3)
        target = None
        used_other = False
        if results:
            target = tuple(results[0].place_key)
            used_other = results[0].used_other_agent_knowledge
            runtime.commit_intent(
                agent_id=consumer_id,
                episode_id=episode_id,
                step_idx=0,
                env_id=ENV_ID,
                intent_type="find_water_source",
                target_place_key=target,
            )
        else:
            # cold-start fallback: random walk
            target = None

        found_water_step = None
        prev = pos
        for step_idx in range(1, steps_per_episode):
            if target is None:
                nbs = neighbors_of(prev)
                rng.shuffle(nbs)
                pos = nbs[0] if nbs else prev
            else:
                pos = _step_toward(prev, target)
            tags = noisy_semantic_tags(rng, world, pos)
            runtime.publish_observation(
                agent_id=consumer_id,
                episode_id=episode_id,
                step_idx=step_idx,
                env_id=ENV_ID,
                place_key=pos,
                semantic_tags=tags,
                node_freshness=1.0,
                confidence=1.0,
            )
            runtime.publish_transition(
                agent_id=consumer_id,
                episode_id=episode_id,
                step_idx=step_idx,
                env_id=ENV_ID,
                src_key=prev,
                dst_key=pos,
                confidence=0.9,
            )
            if pos in world.water_places and found_water_step is None:
                found_water_step = step_idx
                runtime.release_intent(
                    agent_id=consumer_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    env_id=ENV_ID,
                    intent_type="find_water_source",
                    success=True,
                    target_place_key=pos,
                    target_tag=WATER_CONCEPT,
                )
                visited.append(pos)
                prev = pos
                break
            visited.append(pos)
            prev = pos
        if found_water_step is None:
            runtime.release_intent(
                agent_id=consumer_id,
                episode_id=episode_id,
                step_idx=steps_per_episode,
                env_id=ENV_ID,
                intent_type="find_water_source",
                success=False,
                target_place_key=target or (),
                target_tag=WATER_CONCEPT,
            )
        outcome = {
            "agent_id": consumer_id,
            "episode_id": episode_id,
            "env_id": ENV_ID,
            "success": float(found_water_step is not None),
            "found_water_at_step": found_water_step,
            "target_concept_tag": WATER_CONCEPT,
            "terminal_place_key": list(prev),
            "visited_place_keys": [list(p) for p in visited],
            "used_other_agent_knowledge_on_query": used_other,
            "scout_id": scout_id,
        }
        runtime.consolidate_episode(
            agent_id=consumer_id,
            episode_id=episode_id,
            env_id=ENV_ID,
            outcome={k: v for k, v in outcome.items() if k != "visited_place_keys"},
            visited_place_keys=visited,
        )
        episode_outcomes.append(outcome)
    return episode_outcomes


def _step_toward(pos: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
    if pos == target:
        return pos
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return (pos[0] + (1 if dx > 0 else -1), pos[1])
    if dy != 0:
        return (pos[0], pos[1] + (1 if dy > 0 else -1))
    return pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scout_episodes", type=int, default=2)
    parser.add_argument("--scout_steps", type=int, default=24)
    parser.add_argument("--consumer_episodes", type=int, default=6)
    parser.add_argument("--consumer_steps", type=int, default=30)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="tmp/multiagent_collective_smoke",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    world = build_world(rng)

    collective = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    runtime = MultiAgentRuntime(collective)

    scout = AgentSignature(agent_id="scout-A", role="scout", trust=1.0)
    consumer = AgentSignature(agent_id="consumer-B", role="consumer", trust=1.0)
    runtime.register(scout)
    runtime.register(consumer)

    scout_summary = run_scout_phase(
        runtime,
        world,
        rng,
        scout_id=scout.agent_id,
        episodes=args.scout_episodes,
        steps_per_episode=args.scout_steps,
    )

    consumer_outcomes = run_consumer_phase(
        runtime,
        world,
        rng,
        consumer_id=consumer.agent_id,
        scout_id=scout.agent_id,
        episodes=args.consumer_episodes,
        steps_per_episode=args.consumer_steps,
    )

    ground_truth_places_by_tag = {
        WATER_CONCEPT: {tuple(p) for p in world.water_places},
        HAZARD_CONCEPT: {tuple(p) for p in world.hazard_places},
    }
    ground_truth_tag_per_place = {tuple(p): world.tag_for(p) for p in world.water_places | world.hazard_places}

    report = all_metrics(
        collective,
        episode_outcomes=[
            {
                "env_id": o["env_id"],
                "success": o["success"],
                "target_concept_tag": o["target_concept_tag"],
                "terminal_place_key": o["terminal_place_key"],
                "visited_place_keys": o["visited_place_keys"],
            }
            for o in consumer_outcomes
        ],
        ground_truth_places_by_tag=ground_truth_places_by_tag,
        ground_truth_tag_per_place=ground_truth_tag_per_place,
        target_concept_tag=WATER_CONCEPT,
        requesting_agent_id=consumer.agent_id,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    paths = collective.export(args.out_dir)

    consumer_success = sum(o["success"] for o in consumer_outcomes) / max(1, len(consumer_outcomes))
    summary = {
        "seed": args.seed,
        "env_id": ENV_ID,
        "world": {
            "water_places": [list(p) for p in sorted(world.water_places)],
            "hazard_places": [list(p) for p in sorted(world.hazard_places)],
            "n_safe_places": len(world.safe_places),
        },
        "scout": scout_summary,
        "consumer": {
            "episodes": len(consumer_outcomes),
            "success_rate": consumer_success,
            "queries_using_other_agent_knowledge": sum(
                int(o.get("used_other_agent_knowledge_on_query", False)) for o in consumer_outcomes
            ),
        },
        "metrics": report,
        "artifacts": paths,
    }
    summary_path = os.path.join(args.out_dir, "collective_smoke_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
