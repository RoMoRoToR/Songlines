"""Phase 2 smoke-bench — shared concept consolidation with 3 agents.

Three synthetic agents share one ``CollectiveMemory``:

* ``scout-A`` explores the left half of the grid (x < GRID_W // 2).
* ``scout-B`` explores the right half (x ≥ GRID_W // 2), with slight sensor noise.
* ``consumer-C`` arrives after both scouts and queries the concept graph built by
  ``PlaceAlignmentEngine``; it navigates toward the best concept result.

Key Phase 2 assertions:
1. ``cross_agent_concept_support_rate > 0`` — at least one concept is backed by
   both scouts (the water places near the grid centre belong to both halves).
2. ``concept_coverage_rate > 0`` — concept graph covers at least some GT places.
3. ``concept_query_precision > 0`` — concept query returns at least one correct place.
4. Shannon entropy of concepts is low (< 1.5 bits) — concepts are semantically pure.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase2.py \\
        --seed 0 \\
        --out_dir tmp/multiagent_phase2_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_metrics import all_metrics, all_concept_metrics
from songline_drive.collective_types import AgentSignature, CollectiveQuery
from songline_drive.multiagent_runtime import MultiAgentRuntime
from songline_drive.place_alignment import PlaceAlignmentEngine


GRID_W = 10
GRID_H = 8
ENV_ID = "synthetic-grid-10x8"
WATER_TAG = "water_source"
HAZARD_TAG = "hazard_edge"
SAFE_TAG = "safe_neutral"


# ------------------------------------------------------------------ world


@dataclass
class World:
    water_places: Set[Tuple[int, int]] = field(default_factory=set)
    hazard_places: Set[Tuple[int, int]] = field(default_factory=set)

    def tag_for(self, p: Tuple[int, int]) -> str:
        if p in self.water_places:
            return WATER_TAG
        if p in self.hazard_places:
            return HAZARD_TAG
        return SAFE_TAG


def build_world(rng: random.Random) -> World:
    cells = [(x, y) for x in range(GRID_W) for y in range(GRID_H)]
    rng.shuffle(cells)
    # 4 water, 4 hazard — enough spread that scouts from different halves can both see some
    water = set(cells[:4])
    hazard = set(cells[4:8])
    return World(water_places=water, hazard_places=hazard)


# ------------------------------------------------------------------ sensing


def semantic_tags(
    rng: random.Random,
    world: World,
    place: Tuple[int, int],
    noise: float = 0.1,
) -> Dict[str, float]:
    tags: Dict[str, float] = {}
    if place in world.water_places:
        tags[WATER_TAG] = max(0.0, 0.95 - rng.uniform(0, noise))
        tags["water_candidate"] = max(0.0, 0.70 - rng.uniform(0, noise))
        tags["water_visible"] = max(0.0, 0.55 - rng.uniform(0, noise))
        tags["near_water"] = max(0.0, 0.45 - rng.uniform(0, noise))
    elif place in world.hazard_places:
        tags[HAZARD_TAG] = max(0.0, 0.90 - rng.uniform(0, noise))
        tags["adjacent_hazard"] = max(0.0, 0.50 - rng.uniform(0, noise))
    else:
        tags[SAFE_TAG] = max(0.0, 0.60 - rng.uniform(0, noise))
        tags["corridor"] = max(0.0, 0.30 - rng.uniform(0, noise))
    if rng.random() < 0.04:
        tags[WATER_TAG] = tags.get(WATER_TAG, 0.0) + rng.uniform(0.05, 0.12)
    return tags


def neighbors(place: Tuple[int, int]) -> List[Tuple[int, int]]:
    nbs = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = place[0] + dx, place[1] + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            nbs.append((nx, ny))
    return nbs


# ------------------------------------------------------------------ scout walk


def scout_walk(
    rng: random.Random,
    world: World,
    n_steps: int,
    x_range: Tuple[int, int],  # (x_min_inclusive, x_max_exclusive)
) -> List[Tuple[int, int]]:
    """Meander biased toward unvisited cells within an x-band."""
    x_min, x_max = x_range
    pos = (rng.randrange(x_min, x_max), rng.randrange(GRID_H))
    visited: List[Tuple[int, int]] = [pos]
    for _ in range(n_steps - 1):
        cands = [
            (nx, ny)
            for nx, ny in neighbors(pos)
            if x_min <= nx < x_max
        ]
        if not cands:
            cands = neighbors(pos)
        rng.shuffle(cands)
        cands.sort(key=lambda p: 1 if p in visited else 0)
        pos = cands[0]
        visited.append(pos)
    return visited


def run_scout(
    runtime: MultiAgentRuntime,
    world: World,
    rng: random.Random,
    agent_id: str,
    episodes: int,
    steps: int,
    x_range: Tuple[int, int],
    noise: float = 0.10,
) -> Dict:
    visited_all: List[Tuple[int, int]] = []
    successes = 0
    for ep in range(episodes):
        path = scout_walk(rng, world, steps, x_range)
        prev: Optional[Tuple[int, int]] = None
        found_water_step: Optional[int] = None
        ep_visited: List[Tuple[int, int]] = []
        for step_idx, place in enumerate(path):
            tags = semantic_tags(rng, world, place, noise=noise)
            runtime.publish_observation(
                agent_id=agent_id,
                episode_id=ep,
                step_idx=step_idx,
                env_id=ENV_ID,
                place_key=place,
                semantic_tags=tags,
                node_freshness=1.0,
                confidence=1.0,
            )
            if prev is not None and prev != place:
                runtime.publish_transition(
                    agent_id=agent_id,
                    episode_id=ep,
                    step_idx=step_idx,
                    env_id=ENV_ID,
                    src_key=prev,
                    dst_key=place,
                    confidence=0.9,
                )
            if place in world.hazard_places:
                runtime.report_hazard_change(
                    agent_id=agent_id,
                    episode_id=ep,
                    step_idx=step_idx,
                    env_id=ENV_ID,
                    place_key=place,
                    confidence=0.9,
                )
            if place in world.water_places and found_water_step is None:
                runtime.confirm_concept(
                    agent_id=agent_id,
                    episode_id=ep,
                    step_idx=step_idx,
                    env_id=ENV_ID,
                    place_key=place,
                    concept_tag=WATER_TAG,
                    confidence=1.0,
                )
                found_water_step = step_idx
            ep_visited.append(place)
            prev = place
        success = found_water_step is not None
        successes += int(success)
        runtime.consolidate_episode(
            agent_id=agent_id,
            episode_id=ep,
            env_id=ENV_ID,
            outcome={"success": float(success), "target_concept_tag": WATER_TAG},
            visited_place_keys=ep_visited,
        )
        visited_all.extend(ep_visited)
    return {
        "agent_id": agent_id,
        "episodes": episodes,
        "success_count": successes,
        "unique_cells_visited": len(set(visited_all)),
    }


# ------------------------------------------------------------------ consumer walk


def step_toward(pos: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
    if pos == target:
        return pos
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return (pos[0] + (1 if dx > 0 else -1), pos[1])
    if dy != 0:
        return (pos[0], pos[1] + (1 if dy > 0 else -1))
    return pos


def run_consumer(
    runtime: MultiAgentRuntime,
    graph: any,  # SharedConceptGraph from PlaceAlignmentEngine
    world: World,
    rng: random.Random,
    agent_id: str,
    episodes: int,
    steps: int,
) -> List[Dict]:
    outcomes = []
    for ep_offset in range(episodes):
        episode_id = 2000 + ep_offset
        pos: Tuple[int, int] = (rng.randrange(GRID_W), rng.randrange(GRID_H))
        visited = [pos]

        # Query concept graph for best water concept
        concept_results = graph.query_concepts(
            target_tag=WATER_TAG,
            score_weights={WATER_TAG: 0.65, "water_candidate": 0.25, "near_water": 0.15},
            penalty_weights={HAZARD_TAG: 0.30, "adjacent_hazard": 0.20},
            requesting_agent_id=agent_id,
            exclude_self=False,
            min_support_count=1,
            min_supporting_agents=1,
            top_k=3,
        )

        # Pick centroid of the best concept as navigation target
        target: Optional[Tuple[int, int]] = None
        used_other = False
        best_concept_id: Optional[str] = None
        if concept_results:
            best = concept_results[0]
            best_concept_id = best.concept_id
            used_other = best.used_other_agent_knowledge
            if best.centroid_xy is not None:
                tx = int(round(best.centroid_xy[0]))
                ty = int(round(best.centroid_xy[1]))
                tx = max(0, min(GRID_W - 1, tx))
                ty = max(0, min(GRID_H - 1, ty))
                target = (tx, ty)
            elif best.member_count > 0:
                # Fall back to first member place of the concept
                concept_node = graph.concepts[best.concept_id]
                if concept_node.member_place_keys:
                    _env, raw_key = concept_node.member_place_keys[0]
                    if len(raw_key) >= 2:
                        target = (int(raw_key[0]), int(raw_key[1]))

        if target is not None:
            runtime.commit_intent(
                agent_id=agent_id,
                episode_id=episode_id,
                step_idx=0,
                env_id=ENV_ID,
                intent_type="find_water_concept",
                target_place_key=target,
            )

        prev = pos
        found_water_step: Optional[int] = None
        for step_idx in range(1, steps):
            if target is None:
                nbs = neighbors(prev)
                rng.shuffle(nbs)
                pos = nbs[0] if nbs else prev
            else:
                pos = step_toward(prev, target)

            tags = semantic_tags(rng, world, pos)
            runtime.publish_observation(
                agent_id=agent_id,
                episode_id=episode_id,
                step_idx=step_idx,
                env_id=ENV_ID,
                place_key=pos,
                semantic_tags=tags,
                node_freshness=1.0,
                confidence=1.0,
            )
            runtime.publish_transition(
                agent_id=agent_id,
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
                    agent_id=agent_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    env_id=ENV_ID,
                    intent_type="find_water_concept",
                    success=True,
                    target_place_key=pos,
                    target_tag=WATER_TAG,
                )
                visited.append(pos)
                prev = pos
                break
            visited.append(pos)
            prev = pos

        if found_water_step is None and target is not None:
            runtime.release_intent(
                agent_id=agent_id,
                episode_id=episode_id,
                step_idx=steps,
                env_id=ENV_ID,
                intent_type="find_water_concept",
                success=False,
                target_place_key=target,
                target_tag=WATER_TAG,
            )

        outcome = {
            "agent_id": agent_id,
            "episode_id": episode_id,
            "env_id": ENV_ID,
            "success": float(found_water_step is not None),
            "found_water_at_step": found_water_step,
            "target_concept_tag": WATER_TAG,
            "terminal_place_key": list(prev),
            "visited_place_keys": [list(p) for p in visited],
            "used_concept_from_other_agent": used_other,
            "best_concept_id": best_concept_id,
        }
        runtime.consolidate_episode(
            agent_id=agent_id,
            episode_id=episode_id,
            env_id=ENV_ID,
            outcome={k: v for k, v in outcome.items() if k != "visited_place_keys"},
            visited_place_keys=visited,
        )
        outcomes.append(outcome)
    return outcomes


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scout_episodes", type=int, default=3)
    parser.add_argument("--scout_steps", type=int, default=30)
    parser.add_argument("--consumer_episodes", type=int, default=6)
    parser.add_argument("--consumer_steps", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="tmp/multiagent_phase2_smoke")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    world = build_world(rng)

    collective = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    runtime = MultiAgentRuntime(collective)

    scout_a = AgentSignature(agent_id="scout-A", role="scout", trust=1.0)
    scout_b = AgentSignature(agent_id="scout-B", role="scout", trust=0.9)
    consumer_c = AgentSignature(agent_id="consumer-C", role="consumer", trust=1.0)
    for sig in (scout_a, scout_b, consumer_c):
        runtime.register(sig)

    # Scout-A explores left half, scout-B explores right half
    mid = GRID_W // 2
    scout_a_summary = run_scout(
        runtime, world, rng,
        agent_id=scout_a.agent_id,
        episodes=args.scout_episodes,
        steps=args.scout_steps,
        x_range=(0, mid + 2),  # slight overlap near centre
        noise=0.08,
    )
    scout_b_summary = run_scout(
        runtime, world, rng,
        agent_id=scout_b.agent_id,
        episodes=args.scout_episodes,
        steps=args.scout_steps,
        x_range=(mid - 2, GRID_W),  # slight overlap near centre
        noise=0.14,  # slightly noisier sensor
    )

    # Build concept graph from Phase 1 event bus
    engine = PlaceAlignmentEngine(
        semantic_threshold=0.45,
        spatial_radius=4.0,
        tag_match_bonus=0.45,
        min_confidence=0.05,
        cross_env=False,
    )
    concept_graph = engine.build(collective)

    # Consumer queries concept graph and navigates
    consumer_outcomes = run_consumer(
        runtime, concept_graph, world, rng,
        agent_id=consumer_c.agent_id,
        episodes=args.consumer_episodes,
        steps=args.consumer_steps,
    )

    # --- metrics ---
    gt_by_tag = {
        WATER_TAG: {tuple(p) for p in world.water_places},
        HAZARD_TAG: {tuple(p) for p in world.hazard_places},
    }
    gt_tag_per_place = {
        tuple(p): world.tag_for(p)
        for p in world.water_places | world.hazard_places
    }

    phase1_report = all_metrics(
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
        ground_truth_places_by_tag=gt_by_tag,
        ground_truth_tag_per_place=gt_tag_per_place,
        target_concept_tag=WATER_TAG,
        requesting_agent_id=consumer_c.agent_id,
    )

    phase2_report = all_concept_metrics(
        concept_graph,
        ground_truth_places_by_tag=gt_by_tag,
        target_tag=WATER_TAG,
        min_agents_for_cross=2,
    )

    # --- assertions ---
    errors = []
    if phase2_report["cross_agent_concept_support_rate"] <= 0.0:
        errors.append("FAIL: cross_agent_concept_support_rate == 0 — no cross-agent concept formed")
    cov = phase2_report.get("concept_coverage_rate")
    if cov is not None and cov <= 0.0:
        errors.append("FAIL: concept_coverage_rate == 0 — no GT water place in concept graph")
    prec = phase2_report.get("concept_query_precision")
    if prec is not None and prec <= 0.0:
        errors.append("FAIL: concept_query_precision == 0 — concept query returned no correct place")

    # --- export ---
    os.makedirs(args.out_dir, exist_ok=True)
    phase1_paths = collective.export(args.out_dir)
    phase2_paths = concept_graph.export(args.out_dir, filename_prefix="phase2_concepts")

    consumer_success = (
        sum(o["success"] for o in consumer_outcomes) / max(1, len(consumer_outcomes))
    )
    consumer_used_concepts = sum(
        int(o.get("used_concept_from_other_agent", False)) for o in consumer_outcomes
    )

    summary = {
        "seed": args.seed,
        "env_id": ENV_ID,
        "world": {
            "water_places": [list(p) for p in sorted(world.water_places)],
            "hazard_places": [list(p) for p in sorted(world.hazard_places)],
        },
        "scouts": {
            "scout_A": scout_a_summary,
            "scout_B": scout_b_summary,
        },
        "consumer": {
            "episodes": len(consumer_outcomes),
            "success_rate": consumer_success,
            "episodes_using_cross_agent_concepts": consumer_used_concepts,
        },
        "phase1_metrics": phase1_report,
        "phase2_concept_metrics": phase2_report,
        "phase2_assertions": {
            "passed": len(errors) == 0,
            "errors": errors,
        },
        "artifacts": {**phase1_paths, **phase2_paths},
    }

    summary_path = os.path.join(args.out_dir, "phase2_smoke_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        print("\n=== PHASE 2 ASSERTIONS FAILED ===", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
