"""Phase 2 A/B smoke — concept recall vs raw Phase 1 query.

Two experimental modes are supported:

Easy mode (default):
  Scouts use ``confirm_concept()`` at confidence=1.0. Both conditions
  achieve near-perfect precision; the test verifies that concept recall
  does not degrade the baseline.

Hard mode (``--hard``):
  No explicit concept confirmation — scouts publish only noisy
  ``place_observed`` events at sensor noise=0.30. False-positive water
  signals contaminate ~15% of safe cells. In this regime Phase 1 raw
  query sometimes returns noisy non-water places as top-1 (precision<1).
  Phase 2 concept recall, with ``only_dominant_tag=True``, isolates the
  true water concept and maintains high top-1 precision.

Navigation strategy:
  Phase 1 baseline → navigate to ``results[0].place_key`` (highest
  fused-score place).
  Phase 2 concept recall → navigate to the **nearest member place** of
  the best concept (geometrically closest to the consumer's start
  position). This is the natural "use concept knowledge to find the
  nearest confirmed water place" behavior.

Measured deltas (B − A):
  • success_rate: + means concept recall found water more often
  • steps_to_target_mean: + means concept recall reached water faster
  • query_top1_precision: + means concept recall pointed to GT water
    more accurately at query time

Usage::

    PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase2_ab.py \\
        --seed 0 --out_dir tmp/multiagent_phase2_ab

    PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase2_ab.py \\
        --seed 0 --hard --out_dir tmp/multiagent_phase2_ab_hard
"""

from __future__ import annotations

import argparse
import math
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
from songline_drive.concept_recall import ConceptRecallLayer
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
    return World(water_places=set(cells[:4]), hazard_places=set(cells[4:8]))


def semantic_tags(
    rng: random.Random,
    world: World,
    place: Tuple[int, int],
    noise: float,
    fp_rate: float = 0.04,
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
    if rng.random() < fp_rate:
        tags[WATER_TAG] = tags.get(WATER_TAG, 0.0) + rng.uniform(0.05, 0.18)
    return tags


def neighbors(place: Tuple[int, int]) -> List[Tuple[int, int]]:
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = place[0] + dx, place[1] + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            out.append((nx, ny))
    return out


def step_toward(pos: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
    if pos == target:
        return pos
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return (pos[0] + (1 if dx > 0 else -1), pos[1])
    if dy != 0:
        return (pos[0], pos[1] + (1 if dy > 0 else -1))
    return pos


def grid_dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ------------------------------------------------------------------ scouts


def run_scouts(
    runtime: MultiAgentRuntime,
    world: World,
    rng: random.Random,
    scout_episodes: int,
    scout_steps: int,
    noise: float,
    use_confirm: bool,
) -> None:
    mid = GRID_W // 2
    for scout_id, x_lo, x_hi, scout_noise in [
        ("scout-A", 0, mid + 2, noise),
        ("scout-B", mid - 2, GRID_W, noise * 1.4),  # B is noisier
    ]:
        for ep in range(scout_episodes):
            x_start = rng.randrange(x_lo, min(x_hi, GRID_W))
            pos: Tuple[int, int] = (x_start, rng.randrange(GRID_H))
            visited: List[Tuple[int, int]] = [pos]
            prev: Optional[Tuple[int, int]] = None
            found_water = False

            for step_idx in range(scout_steps):
                tags = semantic_tags(rng, world, pos, noise=scout_noise, fp_rate=0.10 if scout_noise > 0.2 else 0.04)
                runtime.publish_observation(
                    agent_id=scout_id, episode_id=ep, step_idx=step_idx,
                    env_id=ENV_ID, place_key=pos, semantic_tags=tags,
                    node_freshness=1.0, confidence=1.0,
                )
                if prev is not None and prev != pos:
                    runtime.publish_transition(
                        agent_id=scout_id, episode_id=ep, step_idx=step_idx,
                        env_id=ENV_ID, src_key=prev, dst_key=pos, confidence=0.9,
                    )
                if pos in world.hazard_places:
                    runtime.report_hazard_change(
                        agent_id=scout_id, episode_id=ep, step_idx=step_idx,
                        env_id=ENV_ID, place_key=pos, confidence=0.9,
                    )
                if use_confirm and pos in world.water_places and not found_water:
                    runtime.confirm_concept(
                        agent_id=scout_id, episode_id=ep, step_idx=step_idx,
                        env_id=ENV_ID, place_key=pos, concept_tag=WATER_TAG,
                        confidence=1.0,
                    )
                    found_water = True
                elif pos in world.water_places:
                    found_water = True

                prev = pos
                cands = [p for p in neighbors(pos) if x_lo <= p[0] < x_hi] or neighbors(pos)
                rng.shuffle(cands)
                cands.sort(key=lambda p: 1 if p in visited else 0)
                pos = cands[0]
                visited.append(pos)

            runtime.consolidate_episode(
                agent_id=scout_id, episode_id=ep, env_id=ENV_ID,
                outcome={"success": float(found_water), "target_concept_tag": WATER_TAG},
                visited_place_keys=visited,
            )


# ------------------------------------------------------------------ consumer


def _water_query(consumer_id: str) -> CollectiveQuery:
    return CollectiveQuery(
        intent_type="find_water_source",
        target_tag=WATER_TAG,
        score_weights={WATER_TAG: 0.65, "water_candidate": 0.25, "near_water": 0.15},
        penalty_weights={HAZARD_TAG: 0.30, "adjacent_hazard": 0.20},
        requesting_agent_id=consumer_id,
        env_id=ENV_ID,
        exclude_self=False,
        min_supporting_agents=1,
        min_fused_score=0.05,
    )


@dataclass
class EpisodeOutcome:
    success: bool
    steps_to_target: Optional[int]
    used_other_agent: bool
    query_top1_is_water: bool
    source: str


def _pick_target(
    pos: Tuple[int, int],
    results: list,
    use_nearest: bool,
) -> Tuple[Optional[Tuple[int, int]], bool, bool]:
    """Return (target, used_other_agent, top1_is_water) from CollectiveQueryResults.

    When ``use_nearest=True`` (Phase 2 concept recall), selects the member
    place geometrically closest to ``pos`` among top results — not just the
    first one. This implements the natural "use concept knowledge to find
    nearest confirmed place" navigation policy.
    """
    if not results:
        return None, False, False
    if use_nearest and len(results) > 1:
        # Pick nearest place from all returned concept members
        best = min(
            results,
            key=lambda r: grid_dist(pos, (int(r.place_key[0]), int(r.place_key[1])))
            if len(r.place_key) >= 2 else float("inf"),
        )
    else:
        best = results[0]

    used_other = best.used_other_agent_knowledge
    top1_is_water = False  # evaluated by caller post-run

    if len(best.place_key) < 2:
        return None, used_other, top1_is_water
    tx = max(0, min(GRID_W - 1, int(best.place_key[0])))
    ty = max(0, min(GRID_H - 1, int(best.place_key[1])))
    return (tx, ty), used_other, best  # type: ignore[return-value]  # caller unpacks


def run_consumer_condition(
    runtime: MultiAgentRuntime,
    world: World,
    rng: random.Random,
    consumer_id: str,
    episodes: int,
    steps: int,
    recall_layer: Optional[ConceptRecallLayer],
    episode_id_offset: int = 2000,
    noise: float = 0.10,
) -> List[EpisodeOutcome]:
    """Run consumer.  recall_layer=None → Phase 1 baseline."""
    outcomes: List[EpisodeOutcome] = []
    for ep_offset in range(episodes):
        episode_id = episode_id_offset + ep_offset
        start: Tuple[int, int] = (rng.randrange(GRID_W), rng.randrange(GRID_H))
        pos = start
        visited = [pos]

        query = _water_query(consumer_id)
        source = "empty"
        target: Optional[Tuple[int, int]] = None
        used_other = False
        top1_is_water = False
        results: list = []

        if recall_layer is not None:
            results, source = runtime.query_with_concept_recall(
                consumer_id, query, recall_layer, top_k=10,
                fallback_to_raw=True,
            )
            use_nearest = (source == "concept_recall")
        else:
            results = runtime.query(consumer_id, query, top_k=5)
            source = "raw_baseline" if results else "empty"
            use_nearest = False

        if results:
            raw_best = results[0]
            # Top-1 precision: check if the highest-score result points to GT water
            if len(raw_best.place_key) >= 2:
                top1_is_water = tuple(raw_best.place_key) in world.water_places

            # Navigation target: nearest for concept recall, top-1 for baseline
            if use_nearest and len(results) > 1:
                nearest = min(
                    results,
                    key=lambda r: grid_dist(
                        start, (int(r.place_key[0]), int(r.place_key[1]))
                    ) if len(r.place_key) >= 2 else float("inf"),
                )
                if len(nearest.place_key) >= 2:
                    tx = max(0, min(GRID_W - 1, int(nearest.place_key[0])))
                    ty = max(0, min(GRID_H - 1, int(nearest.place_key[1])))
                    target = (tx, ty)
            elif len(raw_best.place_key) >= 2:
                tx = max(0, min(GRID_W - 1, int(raw_best.place_key[0])))
                ty = max(0, min(GRID_H - 1, int(raw_best.place_key[1])))
                target = (tx, ty)

            used_other = raw_best.used_other_agent_knowledge

        if target is not None:
            runtime.commit_intent(
                agent_id=consumer_id, episode_id=episode_id, step_idx=0,
                env_id=ENV_ID, intent_type="find_water_source",
                target_place_key=target,
            )

        prev = pos
        found_step: Optional[int] = None
        for step_idx in range(1, steps):
            pos = step_toward(prev, target) if target is not None else (
                neighbors(prev)[rng.randrange(len(neighbors(prev)))]
                if neighbors(prev) else prev
            )
            tags = semantic_tags(rng, world, pos, noise=noise)
            runtime.publish_observation(
                agent_id=consumer_id, episode_id=episode_id, step_idx=step_idx,
                env_id=ENV_ID, place_key=pos, semantic_tags=tags,
                node_freshness=1.0, confidence=1.0,
            )
            if pos in world.water_places and found_step is None:
                found_step = step_idx
                runtime.release_intent(
                    agent_id=consumer_id, episode_id=episode_id, step_idx=step_idx,
                    env_id=ENV_ID, intent_type="find_water_source",
                    success=True, target_place_key=pos, target_tag=WATER_TAG,
                )
                visited.append(pos)
                prev = pos
                break
            visited.append(pos)
            prev = pos

        if found_step is None and target is not None:
            runtime.release_intent(
                agent_id=consumer_id, episode_id=episode_id, step_idx=steps,
                env_id=ENV_ID, intent_type="find_water_source",
                success=False, target_place_key=target, target_tag=WATER_TAG,
            )

        runtime.consolidate_episode(
            agent_id=consumer_id, episode_id=episode_id, env_id=ENV_ID,
            outcome={
                "success": float(found_step is not None),
                "target_concept_tag": WATER_TAG,
                "terminal_place_key": list(prev),
            },
            visited_place_keys=visited,
        )
        outcomes.append(EpisodeOutcome(
            success=found_step is not None,
            steps_to_target=found_step,
            used_other_agent=used_other,
            query_top1_is_water=top1_is_water,
            source=source,
        ))
    return outcomes


# ------------------------------------------------------------------ stats


def summarise(outcomes: List[EpisodeOutcome], steps_max: int) -> Dict:
    n = len(outcomes)
    if n == 0:
        return {}
    success_rate = sum(o.success for o in outcomes) / n
    steps = [
        o.steps_to_target if o.steps_to_target is not None else steps_max
        for o in outcomes
    ]
    steps_mean = sum(steps) / n
    top1_prec = sum(o.query_top1_is_water for o in outcomes) / n
    cross_agent = sum(o.used_other_agent for o in outcomes)
    sources: Dict[str, int] = {}
    for o in outcomes:
        sources[o.source] = sources.get(o.source, 0) + 1
    return {
        "n_episodes": n,
        "success_rate": round(success_rate, 4),
        "steps_to_target_mean": round(steps_mean, 2),
        "query_top1_precision": round(top1_prec, 4),
        "episodes_using_cross_agent": cross_agent,
        "query_sources": sources,
    }


# ------------------------------------------------------------------ one condition pair


def run_condition_pair(
    world: World,
    seed: int,
    scout_episodes: int,
    scout_steps: int,
    consumer_episodes: int,
    consumer_steps: int,
    noise: float,
    use_confirm: bool,
    out_dir: str,
    label: str,
) -> Dict:
    gt_by_tag = {
        WATER_TAG: {tuple(p) for p in world.water_places},
        HAZARD_TAG: {tuple(p) for p in world.hazard_places},
    }

    # --- Condition A: Phase 1 raw baseline ---
    rng_a = random.Random(seed + 1)
    collective_a = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    runtime_a = MultiAgentRuntime(collective_a)
    for sig in [
        AgentSignature("scout-A", role="scout", trust=1.0),
        AgentSignature("scout-B", role="scout", trust=0.9),
        AgentSignature("consumer-C", role="consumer", trust=1.0),
    ]:
        runtime_a.register(sig)
    run_scouts(runtime_a, world, rng_a, scout_episodes, scout_steps, noise, use_confirm)
    outcomes_a = run_consumer_condition(
        runtime_a, world, rng_a, "consumer-C",
        episodes=consumer_episodes, steps=consumer_steps,
        recall_layer=None, episode_id_offset=2000, noise=noise,
    )

    # --- Condition B: Phase 2 concept recall ---
    rng_b = random.Random(seed + 1)  # identical starting seed → same episodes
    collective_b = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    runtime_b = MultiAgentRuntime(collective_b)
    for sig in [
        AgentSignature("scout-A", role="scout", trust=1.0),
        AgentSignature("scout-B", role="scout", trust=0.9),
        AgentSignature("consumer-C", role="consumer", trust=1.0),
    ]:
        runtime_b.register(sig)
    run_scouts(runtime_b, world, rng_b, scout_episodes, scout_steps, noise, use_confirm)

    engine = PlaceAlignmentEngine(
        semantic_threshold=0.45,
        spatial_radius=4.0,
        tag_match_bonus=0.45,
        min_confidence=0.05,
    )
    recall = ConceptRecallLayer(
        engine,
        only_dominant_tag=True,
        min_concept_support=2,
        require_cross_agent=False,
    )
    recall.refresh(collective_b)

    outcomes_b = run_consumer_condition(
        runtime_b, world, rng_b, "consumer-C",
        episodes=consumer_episodes, steps=consumer_steps,
        recall_layer=recall, episode_id_offset=2000, noise=noise,
    )

    sum_a = summarise(outcomes_a, consumer_steps)
    sum_b = summarise(outcomes_b, consumer_steps)

    delta_success = sum_b["success_rate"] - sum_a["success_rate"]
    delta_steps = sum_a["steps_to_target_mean"] - sum_b["steps_to_target_mean"]
    delta_prec = sum_b["query_top1_precision"] - sum_a["query_top1_precision"]

    phase2_metrics = all_concept_metrics(
        recall.graph,
        ground_truth_places_by_tag=gt_by_tag,
        target_tag=WATER_TAG,
        min_agents_for_cross=2,
    )

    errors = []
    if sum_b["success_rate"] < sum_a["success_rate"] - 0.10:
        errors.append(
            f"FAIL: concept recall success_rate {sum_b['success_rate']:.3f} "
            f"worse than baseline {sum_a['success_rate']:.3f} by >10pp"
        )
    if sum_b["query_top1_precision"] < sum_a["query_top1_precision"] - 0.05:
        errors.append(
            f"FAIL: concept recall top1_precision {sum_b['query_top1_precision']:.3f} "
            f"worse than baseline {sum_a['query_top1_precision']:.3f} by >5pp"
        )
    if phase2_metrics["cross_agent_concept_support_rate"] <= 0.0:
        errors.append("FAIL: no cross-agent concepts formed")
    cov = phase2_metrics.get("concept_coverage_rate") or 0.0
    if cov <= 0.0:
        errors.append("FAIL: concept_coverage_rate == 0")
    prec = phase2_metrics.get("concept_query_precision") or 0.0
    if prec < 0.5:
        errors.append(f"FAIL: concept_query_precision {prec:.3f} < 0.5 (only_dominant_tag=True)")

    os.makedirs(out_dir, exist_ok=True)
    collective_a.export(out_dir + f"/{label}_a")
    collective_b.export(out_dir + f"/{label}_b")
    concept_paths = recall.graph.export(out_dir, filename_prefix=f"{label}_concepts")

    return {
        "label": label,
        "mode_settings": {
            "noise": noise,
            "use_confirm_concept": use_confirm,
            "scout_episodes": scout_episodes,
            "scout_steps": scout_steps,
            "consumer_episodes": consumer_episodes,
        },
        "condition_A_phase1_baseline": sum_a,
        "condition_B_phase2_concept_recall": sum_b,
        "delta": {
            "success_rate": round(delta_success, 4),
            "steps_to_target_mean": round(delta_steps, 2),
            "query_top1_precision": round(delta_prec, 4),
            "note": "positive = Phase 2 is better; steps: faster=positive",
        },
        "phase2_concept_metrics": phase2_metrics,
        "assertions": {"passed": len(errors) == 0, "errors": errors},
    }


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scout_episodes", type=int, default=3)
    parser.add_argument("--scout_steps", type=int, default=30)
    parser.add_argument("--consumer_episodes", type=int, default=10)
    parser.add_argument("--consumer_steps", type=int, default=30)
    parser.add_argument("--hard", action="store_true",
                        help="Hard mode: no confirm_concept, noise=0.30, fp_rate=10%%")
    parser.add_argument("--both", action="store_true",
                        help="Run both easy and hard modes and compare")
    parser.add_argument("--out_dir", type=str, default="tmp/multiagent_phase2_ab")
    args = parser.parse_args()

    rng_world = random.Random(args.seed)
    world = build_world(rng_world)

    world_dict = {
        "water_places": [list(p) for p in sorted(world.water_places)],
        "hazard_places": [list(p) for p in sorted(world.hazard_places)],
    }

    results = {}
    all_errors = []

    if args.both or not args.hard:
        easy = run_condition_pair(
            world, args.seed,
            scout_episodes=args.scout_episodes, scout_steps=args.scout_steps,
            consumer_episodes=args.consumer_episodes,
            consumer_steps=args.consumer_steps,
            noise=0.08, use_confirm=True,
            out_dir=args.out_dir, label="easy",
        )
        results["easy"] = easy
        all_errors.extend(easy["assertions"]["errors"])

    if args.both or args.hard:
        hard = run_condition_pair(
            world, args.seed,
            scout_episodes=args.scout_episodes, scout_steps=args.scout_steps,
            consumer_episodes=args.consumer_episodes,
            consumer_steps=args.consumer_steps,
            noise=0.30, use_confirm=False,
            out_dir=args.out_dir, label="hard",
        )
        results["hard"] = hard
        all_errors.extend(hard["assertions"]["errors"])

    summary = {
        "seed": args.seed,
        "world": world_dict,
        "results": results,
        "overall_passed": len(all_errors) == 0,
        "all_errors": all_errors,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "phase2_ab_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for label, res in results.items():
        d = res["delta"]
        status = "✓" if res["assertions"]["passed"] else "✗"
        print(
            f"\n{status} [{label}]  "
            f"Δsuccess={d['success_rate']:+.3f}  "
            f"Δsteps={d['steps_to_target_mean']:+.2f} (faster=+)  "
            f"Δtop1_prec={d['query_top1_precision']:+.3f}",
            file=sys.stderr,
        )

    if all_errors:
        print("\n=== ASSERTIONS FAILED ===", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
