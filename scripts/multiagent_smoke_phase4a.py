"""Phase 4a smoke — descriptive semantic field validation.

Three scenarios in one run:

A. Water / hazard channel separation (real scouting, Phase 3a setup)
   Verifies that after normal scouts observe water and hazard places the
   field's water_source channel is systematically higher for water concepts
   than for hazard concepts, and vice versa for hazard_edge channel.

B. Conflict suppression (Phase 3b isolated-cell setup)
   Verifies that a contested water+hazard concept has lower water_source
   activation than the two clean water reference concepts.

C. Decay
   Applies ``field.decay(steps=100)`` and verifies that top-1 activation
   drops by at least 40 % (λ^100 = 0.95^100 ≈ 0.006 → almost zero, so
   this is a soft check).

All assertions use FieldMode.DESCRIPTIVE — the field does not affect
any retrieval result.  All Phase 1/2/3 passes still produce the same output.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase4a.py \\
        --seed 0 --out_dir tmp/multiagent_phase4a_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_field_types import FieldMode
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import AgentSignature
from songline_drive.concept_recall import ConceptRecallLayer
from songline_drive.field_adapter import FieldAdapter
from songline_drive.field_metrics import (
    all_field_metrics_4a,
    field_conflict_suppression_rate,
    field_cross_channel_separation,
    field_decay_half_life,
    field_top1_stability,
)
from songline_drive.field_visualization import activation_table, save_snapshot
from songline_drive.multiagent_runtime import MultiAgentRuntime
from songline_drive.place_alignment import PlaceAlignmentEngine
from songline_drive.semantic_field import SemanticField

GRID_W = 10
GRID_H = 8
ENV_ID = "synthetic-grid-10x8"
WATER_TAG = "water_source"
HAZARD_TAG = "hazard_edge"
SAFE_TAG = "safe_neutral"


# ─────────────────────────────────────────────── world helpers (same as Phase 3)


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
    rng: random.Random, world: World, place: Tuple[int, int],
    noise: float = 0.08, fp_rate: float = 0.04,
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
        tags[WATER_TAG] = tags.get(WATER_TAG, 0.0) + rng.uniform(0.05, 0.12)
    return tags


def neighbors(p: Tuple[int, int]) -> List[Tuple[int, int]]:
    return [
        (p[0] + dx, p[1] + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if 0 <= p[0] + dx < GRID_W and 0 <= p[1] + dy < GRID_H
    ]


def run_scout(
    runtime: MultiAgentRuntime, world: World, rng: random.Random,
    agent_id: str, episodes: int, steps: int,
    x_range: Tuple[int, int], noise: float = 0.08,
) -> None:
    for ep in range(episodes):
        x_lo, x_hi = x_range
        pos: Tuple[int, int] = (rng.randrange(x_lo, x_hi), rng.randrange(GRID_H))
        visited = [pos]
        prev: Optional[Tuple[int, int]] = None
        found_water = False
        for step_idx in range(steps):
            tags = semantic_tags(rng, world, pos, noise=noise)
            runtime.publish_observation(
                agent_id=agent_id, episode_id=ep, step_idx=step_idx,
                env_id=ENV_ID, place_key=pos, semantic_tags=tags,
                node_freshness=1.0, confidence=1.0,
            )
            if prev and prev != pos:
                runtime.publish_transition(
                    agent_id=agent_id, episode_id=ep, step_idx=step_idx,
                    env_id=ENV_ID, src_key=prev, dst_key=pos, confidence=0.9,
                )
            if pos in world.hazard_places:
                runtime.report_hazard_change(
                    agent_id=agent_id, episode_id=ep, step_idx=step_idx,
                    env_id=ENV_ID, place_key=pos,
                )
            if pos in world.water_places and not found_water:
                runtime.confirm_concept(
                    agent_id=agent_id, episode_id=ep, step_idx=step_idx,
                    env_id=ENV_ID, place_key=pos,
                    concept_tag=WATER_TAG, confidence=1.0,
                )
                found_water = True
            prev = pos
            cands = [p for p in neighbors(pos) if x_lo <= p[0] < x_hi] or neighbors(pos)
            rng.shuffle(cands)
            cands.sort(key=lambda p: 1 if p in visited else 0)
            pos = cands[0]
            visited.append(pos)
        runtime.consolidate_episode(
            agent_id=agent_id, episode_id=ep, env_id=ENV_ID,
            outcome={"success": float(found_water), "target_concept_tag": WATER_TAG},
            visited_place_keys=visited,
        )


def make_engine() -> PlaceAlignmentEngine:
    return PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=4.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )


def make_recall(engine: PlaceAlignmentEngine) -> ConceptRecallLayer:
    return ConceptRecallLayer(
        engine,
        only_dominant_tag=True,
        min_concept_support=2,
        decay_engine=TemporalDecayEngine(),
        conflict_rules=ConflictRuleSet.songlines_default(),
    )


def make_field(mode: str = FieldMode.DESCRIPTIVE) -> SemanticField:
    return SemanticField(
        channels=[WATER_TAG, HAZARD_TAG, SAFE_TAG],
        mode=mode,
        lambda_decay=0.95,
        alpha_belief=0.60,
        eta_conflict=0.30,
        diffusion_steps=1,
    )


# ═══════════════════════════════════════════════════════════ Scenario A


def run_scenario_a(world: World, rng: random.Random, out_dir: str) -> Dict:
    """Water/hazard channel separation via standard scouting."""
    collective = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    runtime = MultiAgentRuntime(collective)
    for sig in [
        AgentSignature("scout-A", role="scout", trust=1.0),
        AgentSignature("scout-B", role="scout", trust=0.9),
    ]:
        runtime.register(sig)

    mid = GRID_W // 2
    run_scout(runtime, world, rng, "scout-A", episodes=3, steps=30, x_range=(0, mid + 2))
    run_scout(runtime, world, rng, "scout-B", episodes=3, steps=30, x_range=(mid - 2, GRID_W))

    recall = make_recall(make_engine())
    field = make_field(FieldMode.DESCRIPTIVE)
    adapter = FieldAdapter(field, recall, mode=FieldMode.DESCRIPTIVE)
    graph, _ = adapter.refresh(collective)

    # Build concept→dominant_tag map
    concept_tag_map = {cid: c.dominant_tag for cid, c in graph.concepts.items()}

    water_metrics = all_field_metrics_4a(
        field, WATER_TAG, concept_tag_map,
        reference_channel=HAZARD_TAG,
    )
    hazard_metrics = all_field_metrics_4a(
        field, HAZARD_TAG, concept_tag_map,
        reference_channel=WATER_TAG,
    )

    snap_path = os.path.join(out_dir, "phase4a_scenario_a_snapshot.json")
    save_snapshot(field, snap_path, label="scenario_a")

    table = activation_table(field, channels=[WATER_TAG, HAZARD_TAG, SAFE_TAG],
                             concept_tag_map=concept_tag_map)

    # Key numbers for assertions
    split_w = water_metrics.get("activation_split_by_tag", {})
    split_h = hazard_metrics.get("activation_split_by_tag", {})
    sep_wh = water_metrics.get("field_cross_channel_separation", 0.0)

    return {
        "n_concepts": len(graph.concepts),
        "water_channel_metrics": water_metrics,
        "hazard_channel_metrics": hazard_metrics,
        "water_tag_mean_water_activation": split_w.get(WATER_TAG),
        "hazard_tag_mean_water_activation": split_w.get(HAZARD_TAG),
        "water_tag_mean_hazard_activation": split_h.get(WATER_TAG),
        "hazard_tag_mean_hazard_activation": split_h.get(HAZARD_TAG),
        "water_hazard_channel_separation": sep_wh,
        "decay_half_life_steps": field_decay_half_life(field.lambda_decay),
        "activation_table": table,
    }


# ═══════════════════════════════════════════════════════════ Scenario B


def run_scenario_b(rng: random.Random, out_dir: str) -> Dict:
    """Conflict suppression: clean water vs contested water+hazard cell.

    Uses the Phase 3b fixed-cell design:
      PURE_A = (0, 0) — 8 water observations → isolated clean water concept
      PURE_B = (9, 7) — 8 water observations → isolated clean water concept
      contested = (4, 3) — 8 water + 8 hazard → mixed concept

    All three cells are > 4.0 apart → each forms its own concept.
    """
    PURE_A: Tuple[int, int] = (0, 0)
    PURE_B: Tuple[int, int] = (9, 7)
    CONTESTED: Tuple[int, int] = (4, 3)

    collective = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    for sig in [
        AgentSignature("scout-A", role="scout", trust=1.0),
        AgentSignature("scout-B", role="scout", trust=0.9),
    ]:
        collective.register_agent(sig)

    for ep_off, wp in enumerate([PURE_A, PURE_B]):
        for obs_i in range(8):
            collective.publish_event(
                "place_observed", "scout-A", episode_id=60 + ep_off, step_idx=obs_i,
                env_id=ENV_ID,
                payload={
                    "place_key": list(wp),
                    "semantic_tags": {WATER_TAG: 0.92, "water_candidate": 0.72, "water_visible": 0.58},
                    "node_freshness": 1.0,
                },
                confidence=1.0,
            )

    for obs_i in range(8):
        collective.publish_event(
            "place_observed", "scout-A", episode_id=70, step_idx=obs_i,
            env_id=ENV_ID,
            payload={
                "place_key": list(CONTESTED),
                "semantic_tags": {WATER_TAG: 0.88, "water_visible": 0.62},
                "node_freshness": 1.0,
            },
            confidence=1.0,
        )
    for obs_i in range(8):
        collective.publish_event(
            "place_observed", "scout-B", episode_id=71, step_idx=obs_i,
            env_id=ENV_ID,
            payload={
                "place_key": list(CONTESTED),
                "semantic_tags": {HAZARD_TAG: 0.85, "adjacent_hazard": 0.62},
                "node_freshness": 1.0,
            },
            confidence=0.9,
        )

    engine = PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=4.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )
    recall = ConceptRecallLayer(
        engine,
        only_dominant_tag=True,
        min_concept_support=1,
        decay_engine=TemporalDecayEngine(),
        conflict_rules=ConflictRuleSet.songlines_default(),
    )
    field = make_field(FieldMode.DESCRIPTIVE)
    adapter = FieldAdapter(field, recall, mode=FieldMode.DESCRIPTIVE)
    graph, _ = adapter.refresh(collective)

    concept_tag_map = {cid: c.dominant_tag for cid, c in graph.concepts.items()}

    # Identify concepts
    contested_graph_key = (ENV_ID, tuple(CONTESTED))
    contested_cid = graph.place_to_concept.get(contested_graph_key)
    pure_a_cid = graph.place_to_concept.get((ENV_ID, tuple(PURE_A)))
    pure_b_cid = graph.place_to_concept.get((ENV_ID, tuple(PURE_B)))

    contested_act = field.activation_for(contested_cid or "", WATER_TAG) if contested_cid else None
    pure_a_act = field.activation_for(pure_a_cid or "", WATER_TAG) if pure_a_cid else None
    pure_b_act = field.activation_for(pure_b_cid or "", WATER_TAG) if pure_b_cid else None

    conflict_suppress = field_conflict_suppression_rate(field, WATER_TAG, conflict_threshold=0.15)
    top1_stable = field_top1_stability(field, WATER_TAG)

    water_metrics = all_field_metrics_4a(
        field, WATER_TAG, concept_tag_map,
        conflict_threshold=0.15,
        reference_channel=HAZARD_TAG,
    )

    snap_path = os.path.join(out_dir, "phase4a_scenario_b_snapshot.json")
    save_snapshot(field, snap_path, label="scenario_b")

    return {
        "contested_place": list(CONTESTED),
        "contested_cid": contested_cid,
        "pure_a_cid": pure_a_cid,
        "pure_b_cid": pure_b_cid,
        "contested_water_activation": contested_act,
        "pure_a_water_activation": pure_a_act,
        "pure_b_water_activation": pure_b_act,
        "contested_conflict_score": (
            round(graph.concepts[contested_cid].conflict_score, 4)
            if contested_cid and contested_cid in graph.concepts else None
        ),
        "field_conflict_suppression_rate": conflict_suppress,
        "field_top1_stability": top1_stable,
        "water_metrics": water_metrics,
    }


# ═══════════════════════════════════════════════════════════ Scenario C


def run_scenario_c(world: World, rng: random.Random, out_dir: str) -> Dict:
    """Decay: top-1 activation drops after simulating 100 elapsed steps."""
    collective = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    runtime = MultiAgentRuntime(collective)
    for sig in [
        AgentSignature("scout-A", role="scout", trust=1.0),
        AgentSignature("scout-B", role="scout", trust=0.9),
    ]:
        runtime.register(sig)

    mid = GRID_W // 2
    run_scout(runtime, world, rng, "scout-A", episodes=3, steps=30, x_range=(0, mid + 2))
    run_scout(runtime, world, rng, "scout-B", episodes=3, steps=30, x_range=(mid - 2, GRID_W))

    recall = make_recall(make_engine())
    field = make_field(FieldMode.DESCRIPTIVE)
    adapter = FieldAdapter(field, recall, mode=FieldMode.DESCRIPTIVE)
    adapter.refresh(collective)

    top1_before = field.top_k_for_channel(WATER_TAG, k=1)
    act_before = top1_before[0][1] if top1_before else 0.0

    DECAY_STEPS = 100
    field.decay(current_seq=0, steps=DECAY_STEPS)

    top1_after = field.top_k_for_channel(WATER_TAG, k=1)
    act_after = top1_after[0][1] if top1_after else 0.0

    expected_factor = field.lambda_decay ** DECAY_STEPS
    actual_factor = act_after / act_before if act_before > 0 else 0.0

    return {
        "decay_steps": DECAY_STEPS,
        "activation_before": round(act_before, 5),
        "activation_after": round(act_after, 5),
        "actual_decay_factor": round(actual_factor, 5),
        "expected_decay_factor": round(expected_factor, 5),
        "decay_half_life_steps": field_decay_half_life(field.lambda_decay),
    }


# ═══════════════════════════════════════════════════════════ assertions


def check_assertions(ra: Dict, rb: Dict, rc: Dict) -> List[str]:
    errors: List[str] = []

    # Scenario A: channel separation
    w_in_water = ra.get("water_tag_mean_water_activation")
    w_in_hazard = ra.get("hazard_tag_mean_water_activation")
    h_in_hazard = ra.get("hazard_tag_mean_hazard_activation")
    h_in_water = ra.get("water_tag_mean_hazard_activation")

    if w_in_water is not None and w_in_hazard is not None:
        if w_in_water <= w_in_hazard:
            errors.append(
                f"4a-A FAIL: water concepts water_source activation ({w_in_water:.3f}) "
                f"≤ hazard concepts ({w_in_hazard:.3f})"
            )
    else:
        errors.append("4a-A FAIL: missing water/hazard mean activations for water_source channel")

    if h_in_hazard is not None and h_in_water is not None:
        if h_in_hazard <= h_in_water:
            errors.append(
                f"4a-A FAIL: hazard concepts hazard_edge activation ({h_in_hazard:.3f}) "
                f"≤ water concepts ({h_in_water:.3f})"
            )
    else:
        errors.append("4a-A FAIL: missing water/hazard mean activations for hazard_edge channel")

    sep = ra.get("water_hazard_channel_separation", 0)
    if sep < 0.02:
        errors.append(f"4a-A FAIL: water/hazard channel separation too low ({sep:.4f} < 0.02)")

    if not ra.get("water_channel_metrics", {}).get("field_top1_stability", False):
        errors.append("4a-A FAIL: field_top1_stability = False for water_source channel")

    # Scenario B: conflict suppression
    pure_a = rb.get("pure_a_water_activation")
    pure_b = rb.get("pure_b_water_activation")
    contested = rb.get("contested_water_activation")

    if pure_a is not None and contested is not None:
        if contested >= pure_a:
            errors.append(
                f"4a-B FAIL: contested activation ({contested:.4f}) "
                f">= pure_a activation ({pure_a:.4f}); conflict not suppressed"
            )
    if pure_b is not None and contested is not None:
        if contested >= pure_b:
            errors.append(
                f"4a-B FAIL: contested activation ({contested:.4f}) "
                f">= pure_b activation ({pure_b:.4f}); conflict not suppressed"
            )

    suppress = rb.get("field_conflict_suppression_rate")
    if suppress is not None and suppress < 0.5:
        errors.append(
            f"4a-B FAIL: conflict_suppression_rate={suppress:.3f} < 0.5"
        )

    if not rb.get("field_top1_stability"):
        errors.append("4a-B FAIL: field_top1_stability = False for water_source channel")

    # Scenario C: decay
    act_b = rc.get("activation_before", 0.0)
    act_a = rc.get("activation_after", 0.0)
    if act_b > 0 and act_a >= act_b * 0.6:
        errors.append(
            f"4a-C FAIL: activation did not decay sufficiently "
            f"(before={act_b:.5f}, after={act_a:.5f})"
        )

    return errors


# ═══════════════════════════════════════════════════════════ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="tmp/multiagent_phase4a_smoke")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    world = build_world(rng)

    ra = run_scenario_a(world, random.Random(args.seed + 1), args.out_dir)
    rb = run_scenario_b(random.Random(args.seed + 2), args.out_dir)
    rc = run_scenario_c(world, random.Random(args.seed + 3), args.out_dir)

    errors = check_assertions(ra, rb, rc)
    passed = len(errors) == 0

    if passed:
        print(
            f"✓ Phase 4a passed  "
            f"sep={ra.get('water_hazard_channel_separation', 0):.3f}  "
            f"suppress={rb.get('field_conflict_suppression_rate', 'n/a')}  "
            f"decay_factor={rc.get('actual_decay_factor', 0):.5f}"
        )
    else:
        print("\n=== PHASE 4a ASSERTIONS FAILED ===")
        for e in errors:
            print(" ", e)

    summary = {
        "seed": args.seed,
        "world": {
            "water_places": [list(p) for p in sorted(world.water_places)],
            "hazard_places": [list(p) for p in sorted(world.hazard_places)],
        },
        "scenario_a_separation": ra,
        "scenario_b_conflict": rb,
        "scenario_c_decay": rc,
        "assertions": {"passed": passed, "errors": errors},
    }

    summary_path = os.path.join(args.out_dir, "phase4a_smoke_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
