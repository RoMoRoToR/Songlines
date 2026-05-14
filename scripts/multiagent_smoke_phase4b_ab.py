"""Phase 4b A/B smoke — read_only field reranking vs concept recall.

Three modes are compared on two scenarios:

  Mode A — raw Phase 1 query (CollectiveMemory.query_collective_nodes)
  Mode B — Phase 2/3 concept recall (ConceptRecallLayer, no field)
  Mode C — Phase 4b field reranking (FieldAdapter, mode=read_only)

Scenario I — clean water (standard scouts, no contested cell)
  All three modes should find the water concept.  The question is whether
  field reranking maintains or improves precision@1.

Scenario II — contested field (Phase 3b isolated-cell setup)
  PURE_A (0,0) and PURE_B (9,7) are clean water; CONTESTED (4,3) has
  both water and hazard signal.  Mode B (pure concept recall) returns all
  three; Mode C (field reranking) should deprioritise the contested one,
  raising precision@1 and precision@3 for the clean water concepts.

Assertions:
  - Mode C precision@1 >= Mode B precision@1 in scenario II
  - Mode C does not degrade precision@1 vs Mode B in scenario I
  - field_top1_gain >= 0 across both scenarios

Usage::

    PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase4b_ab.py \\
        --seed 0 --out_dir tmp/multiagent_phase4b_smoke
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

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_field_types import FieldMode
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import AgentSignature, CollectiveQuery
from songline_drive.concept_recall import ConceptRecallLayer
from songline_drive.field_adapter import FieldAdapter
from songline_drive.field_metrics import (
    all_field_metrics_4b,
    field_rerank_precision_at_k,
    field_top1_gain,
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


# ────────────────────────────────────────── reuse world helpers from phase4a


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
    agent_id: str, episodes: int, steps: int, x_range: Tuple[int, int],
    noise: float = 0.08,
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


def make_recall_layer(
    engine: PlaceAlignmentEngine,
    min_concept_support: int = 2,
) -> ConceptRecallLayer:
    return ConceptRecallLayer(
        engine,
        only_dominant_tag=True,
        min_concept_support=min_concept_support,
        decay_engine=TemporalDecayEngine(),
        conflict_rules=ConflictRuleSet.songlines_default(),
    )


def make_field_adapter(
    recall: ConceptRecallLayer,
    mode: str = FieldMode.READ_ONLY,
    field_weight: float = 0.35,
) -> FieldAdapter:
    field = SemanticField(
        channels=[WATER_TAG, HAZARD_TAG, SAFE_TAG],
        mode=mode,
        lambda_decay=0.95,
        alpha_belief=0.60,
        eta_conflict=0.30,
        diffusion_steps=1,
    )
    return FieldAdapter(field, recall, field_weight=field_weight, mode=mode)


# ═══════════════════════════════════════════════════════════ Scenario I


def run_scenario_i(world: World, rng: random.Random, out_dir: str) -> Dict:
    """Clean water: all three modes compared on a standard scouting setup."""
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

    current_seq = collective._next_seq  # noqa: SLF001
    q = CollectiveQuery(
        requesting_agent_id="consumer-I",
        intent_type="seek_resource",
        target_tag=WATER_TAG,
        env_id=ENV_ID,
    )

    # Ground-truth: all water_source concepts (dominant_tag matches)
    recall_a = make_recall_layer(make_engine())
    graph_a = recall_a.refresh(collective)
    gt_cids: Set[str] = {
        cid for cid, c in graph_a.concepts.items() if c.dominant_tag == WATER_TAG
    }

    # Mode A: raw Phase 1
    raw_results = collective.query_collective_nodes(q, top_k=5)
    # Mode A doesn't return concept_ids directly; use place_keys as proxy
    mode_a_places = [(r.env_id, r.place_key) for r in raw_results]

    # Mode B: concept recall only
    recall_b = make_recall_layer(make_engine())
    recall_b.refresh(collective)
    b_results = recall_b.query(
        target_tag=WATER_TAG, requesting_agent_id="consumer-I",
        env_id=ENV_ID, top_k=5, current_seq=current_seq,
    )
    b_cids = [r.concept_id for r in b_results]

    # Mode C: field reranked
    adapter_c = make_field_adapter(make_recall_layer(make_engine()))
    adapter_c.refresh(collective, current_seq=current_seq)
    c_results_raw = adapter_c.recall_layer.query(
        target_tag=WATER_TAG, requesting_agent_id="consumer-I",
        env_id=ENV_ID, top_k=15, current_seq=current_seq,
    )
    c_reranked = adapter_c.field.rerank(c_results_raw, channel=WATER_TAG)
    c_cids = [r.concept_id for r in c_reranked]

    snap_path = os.path.join(out_dir, "phase4b_scenario_i_field.json")
    save_snapshot(adapter_c.field, snap_path, label="scenario_i")

    concept_tag_map = {cid: c.dominant_tag for cid, c in graph_a.concepts.items()}
    table = activation_table(adapter_c.field, channels=[WATER_TAG, HAZARD_TAG],
                             concept_tag_map=concept_tag_map)

    b_prec1 = field_rerank_precision_at_k(b_cids, gt_cids, k=1)
    c_prec1 = field_rerank_precision_at_k(c_cids, gt_cids, k=1)
    b_prec3 = field_rerank_precision_at_k(b_cids, gt_cids, k=3)
    c_prec3 = field_rerank_precision_at_k(c_cids, gt_cids, k=3)
    top1_gain = field_top1_gain(
        b_cids[0] if b_cids else None,
        c_cids[0] if c_cids else None,
        gt_cids,
    )

    metrics_4b = all_field_metrics_4b(
        adapter_c.field, WATER_TAG,
        baseline_recall=b_cids,
        field_recall=c_cids,
        gt_concept_ids=gt_cids,
        k=3,
    )

    return {
        "gt_water_concept_ids": sorted(gt_cids),
        "mode_b_top3_concept_ids": b_cids[:3],
        "mode_c_top3_concept_ids": c_cids[:3],
        "mode_b_precision_at_1": b_prec1,
        "mode_c_precision_at_1": c_prec1,
        "mode_b_precision_at_3": b_prec3,
        "mode_c_precision_at_3": c_prec3,
        "field_top1_gain": top1_gain,
        "4b_metrics": metrics_4b,
        "activation_table": table,
    }


# ═══════════════════════════════════════════════════════════ Scenario II


def run_scenario_ii(rng: random.Random, out_dir: str) -> Dict:
    """Contested field: field reranking should deprioritise water+hazard concept."""
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

    current_seq = collective._next_seq  # noqa: SLF001

    engine_b = PlaceAlignmentEngine(semantic_threshold=0.45, spatial_radius=4.0, tag_match_bonus=0.45, min_confidence=0.05)
    recall_b = ConceptRecallLayer(engine_b, only_dominant_tag=True, min_concept_support=1,
                                  decay_engine=TemporalDecayEngine(),
                                  conflict_rules=ConflictRuleSet.songlines_default())
    graph_b = recall_b.refresh(collective)

    engine_c = PlaceAlignmentEngine(semantic_threshold=0.45, spatial_radius=4.0, tag_match_bonus=0.45, min_confidence=0.05)
    recall_c = ConceptRecallLayer(engine_c, only_dominant_tag=True, min_concept_support=1,
                                  decay_engine=TemporalDecayEngine(),
                                  conflict_rules=ConflictRuleSet.songlines_default())
    adapter_c = make_field_adapter(recall_c, mode=FieldMode.READ_ONLY, field_weight=0.35)
    adapter_c.refresh(collective, current_seq=current_seq)

    # Identify concepts
    contested_cid = graph_b.place_to_concept.get((ENV_ID, tuple(CONTESTED)))
    pure_a_cid = graph_b.place_to_concept.get((ENV_ID, tuple(PURE_A)))
    pure_b_cid = graph_b.place_to_concept.get((ENV_ID, tuple(PURE_B)))
    gt_cids: Set[str] = {
        cid for cid in [pure_a_cid, pure_b_cid] if cid is not None
    }

    # Mode B: concept recall (no field)
    b_results = recall_b.query(
        target_tag=WATER_TAG, requesting_agent_id="consumer-II",
        env_id=ENV_ID, top_k=5, current_seq=current_seq,
    )
    b_cids = [r.concept_id for r in b_results]

    # Mode C: field reranked
    c_results_raw = adapter_c.recall_layer.query(
        target_tag=WATER_TAG, requesting_agent_id="consumer-II",
        env_id=ENV_ID, top_k=15, current_seq=current_seq,
    )
    c_reranked = adapter_c.field.rerank(c_results_raw, channel=WATER_TAG)
    c_cids = [r.concept_id for r in c_reranked]

    # Field activations
    pure_a_act = adapter_c.field.activation_for(pure_a_cid or "", WATER_TAG) if pure_a_cid else None
    pure_b_act = adapter_c.field.activation_for(pure_b_cid or "", WATER_TAG) if pure_b_cid else None
    contested_act = adapter_c.field.activation_for(contested_cid or "", WATER_TAG) if contested_cid else None

    # Where does contested rank in each mode?
    b_contested_rank = b_cids.index(contested_cid) if contested_cid in b_cids else -1
    c_contested_rank = c_cids.index(contested_cid) if contested_cid in c_cids else -1

    b_prec1 = field_rerank_precision_at_k(b_cids, gt_cids, k=1)
    c_prec1 = field_rerank_precision_at_k(c_cids, gt_cids, k=1)
    b_prec3 = field_rerank_precision_at_k(b_cids, gt_cids, k=3)
    c_prec3 = field_rerank_precision_at_k(c_cids, gt_cids, k=3)
    top1_gain = field_top1_gain(
        b_cids[0] if b_cids else None,
        c_cids[0] if c_cids else None,
        gt_cids,
    )

    metrics_4b = all_field_metrics_4b(
        adapter_c.field, WATER_TAG,
        baseline_recall=b_cids,
        field_recall=c_cids,
        gt_concept_ids=gt_cids,
        k=1,
    )

    concept_tag_map = {cid: c.dominant_tag for cid, c in graph_b.concepts.items()}
    snap_path = os.path.join(out_dir, "phase4b_scenario_ii_field.json")
    save_snapshot(adapter_c.field, snap_path, label="scenario_ii")
    table = activation_table(adapter_c.field, channels=[WATER_TAG, HAZARD_TAG],
                             concept_tag_map=concept_tag_map)

    return {
        "contested_cid": contested_cid,
        "pure_a_cid": pure_a_cid,
        "pure_b_cid": pure_b_cid,
        "gt_clean_water_cids": sorted(gt_cids),
        "mode_b_ranked_cids": b_cids,
        "mode_c_ranked_cids": c_cids,
        "b_contested_rank": b_contested_rank,
        "c_contested_rank": c_contested_rank,
        "pure_a_water_activation": pure_a_act,
        "pure_b_water_activation": pure_b_act,
        "contested_water_activation": contested_act,
        "mode_b_precision_at_1": b_prec1,
        "mode_c_precision_at_1": c_prec1,
        "mode_b_precision_at_3": b_prec3,
        "mode_c_precision_at_3": c_prec3,
        "field_top1_gain": top1_gain,
        "4b_metrics": metrics_4b,
        "activation_table": table,
    }


# ═══════════════════════════════════════════════════════════ assertions


def check_assertions(ri: Dict, rii: Dict) -> List[str]:
    errors: List[str] = []

    # Scenario I: mode C must not degrade vs mode B
    b1 = ri.get("mode_b_precision_at_1", 0)
    c1 = ri.get("mode_c_precision_at_1", 0)
    if c1 < b1:
        errors.append(
            f"4b-I FAIL: field reranking degraded precision@1 "
            f"({c1:.2f} < baseline {b1:.2f})"
        )

    gain_i = ri.get("field_top1_gain", 0)
    if gain_i < 0:
        errors.append(
            f"4b-I FAIL: field_top1_gain={gain_i} (mode C worse than baseline for GT top-1)"
        )

    # Scenario II: field should deprioritise the contested concept
    c_rank_ii = rii.get("c_contested_rank", -1)
    b_rank_ii = rii.get("b_contested_rank", -1)

    # Contested concept should rank lower (higher index) in mode C vs mode B
    if (
        contested_cid := rii.get("contested_cid")
    ) and b_rank_ii >= 0 and c_rank_ii >= 0:
        if c_rank_ii < b_rank_ii:
            errors.append(
                f"4b-II FAIL: field moved contested concept UP "
                f"(mode_b_rank={b_rank_ii}, mode_c_rank={c_rank_ii})"
            )

    c1_ii = rii.get("mode_c_precision_at_1", 0)
    b1_ii = rii.get("mode_b_precision_at_1", 0)
    if c1_ii < b1_ii:
        errors.append(
            f"4b-II FAIL: field reranking degraded precision@1 in contested scenario "
            f"({c1_ii:.2f} < {b1_ii:.2f})"
        )

    # Field activation: pure water > contested
    pure_a = rii.get("pure_a_water_activation")
    pure_b = rii.get("pure_b_water_activation")
    contested = rii.get("contested_water_activation")
    if pure_a is not None and contested is not None and pure_a <= contested:
        errors.append(
            f"4b-II FAIL: pure_a activation ({pure_a:.4f}) <= contested ({contested:.4f})"
        )
    if pure_b is not None and contested is not None and pure_b <= contested:
        errors.append(
            f"4b-II FAIL: pure_b activation ({pure_b:.4f}) <= contested ({contested:.4f})"
        )

    return errors


# ═══════════════════════════════════════════════════════════ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="tmp/multiagent_phase4b_smoke")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    world = build_world(rng)

    ri = run_scenario_i(world, random.Random(args.seed + 10), args.out_dir)
    rii = run_scenario_ii(random.Random(args.seed + 20), args.out_dir)

    errors = check_assertions(ri, rii)
    passed = len(errors) == 0

    if passed:
        print(
            f"✓ Phase 4b passed  "
            f"I(prec@1 B={ri['mode_b_precision_at_1']:.2f}→C={ri['mode_c_precision_at_1']:.2f})  "
            f"II(contested_rank B={rii['b_contested_rank']}→C={rii['c_contested_rank']}  "
            f"prec@1 B={rii['mode_b_precision_at_1']:.2f}→C={rii['mode_c_precision_at_1']:.2f})"
        )
    else:
        print("\n=== PHASE 4b ASSERTIONS FAILED ===")
        for e in errors:
            print(" ", e)

    summary = {
        "seed": args.seed,
        "world": {
            "water_places": [list(p) for p in sorted(world.water_places)],
            "hazard_places": [list(p) for p in sorted(world.hazard_places)],
        },
        "scenario_i_clean_water": ri,
        "scenario_ii_contested": rii,
        "assertions": {"passed": passed, "errors": errors},
    }

    summary_path = os.path.join(args.out_dir, "phase4b_smoke_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
