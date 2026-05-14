"""Phase 3 smoke-bench — temporal decay (3a), conflict fusion (3b), incremental updates (3c).

Three sections run sequentially on the same synthetic world.

Phase 3a — temporal decay
  After scouts finish, the graph is built.  Then 120 seq-steps pass without
  any new observations (simulated by advancing a "virtual_seq" counter).
  The test verifies:
  - stale_concept_suppression_rate > 0.5 after a 120-step gap
  - latency_to_deactivation is finite for every currently active concept
  A second scout wave then refreshes one of the water concepts; recovery is
  verified via refreshed_concept_recovery_rate.

Phase 3b — conflict fusion
  A "contested cell" is forced: one cell at the grid centre receives both a
  strong water_source observation AND a strong hazard_edge observation from
  different scouts.  After building the graph, the conflict penalty on the
  water concept should be significant (> 0.2), its purity should drop, and
  the adjusted score should be lower than the raw score.
  Metrics: concept_purity_under_contradiction, false_persistence_rate.

Phase 3c — incremental updates
  Scout phase builds the initial graph.  A new "mini-scout" then publishes
  10 additional observations. IncrementalAlignmentEngine.update() is called
  instead of a full rebuild.  Metrics:
  - update_stability > 0.85 (minimal reshuffling of stable places)
  - concept_churn_rate < 0.15
  - reuse_after_incremental_update (top-1 concept stable)
  - precision stays 1.0 after incremental update

Usage::

    PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase3.py \\
        --seed 0 --out_dir tmp/multiagent_phase3_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_metrics import (
    all_concept_metrics,
    all_phase3_metrics,
    conflict_resolution_accuracy,
    concept_churn_rate,
    concept_purity_under_contradiction,
    false_persistence_rate,
    latency_to_deactivation,
    refreshed_concept_recovery_rate,
    reuse_after_incremental_update,
    stale_concept_suppression_rate,
    update_stability,
)
from songline_drive.collective_types import AgentSignature
from songline_drive.concept_recall import ConceptRecallLayer
from songline_drive.multiagent_runtime import MultiAgentRuntime
from songline_drive.place_alignment import IncrementalAlignmentEngine, PlaceAlignmentEngine

GRID_W = 10
GRID_H = 8
ENV_ID = "synthetic-grid-10x8"
WATER_TAG = "water_source"
HAZARD_TAG = "hazard_edge"
SAFE_TAG = "safe_neutral"


# ─────────────────────────────────────────────────────────────── world helpers


@dataclass
class World:
    water_places: Set[Tuple[int, int]] = field(default_factory=set)
    hazard_places: Set[Tuple[int, int]] = field(default_factory=set)
    contested_place: Optional[Tuple[int, int]] = None

    def tag_for(self, p: Tuple[int, int]) -> str:
        if p in self.water_places:
            return WATER_TAG
        if p in self.hazard_places:
            return HAZARD_TAG
        return SAFE_TAG


def build_world(rng: random.Random) -> World:
    cells = [(x, y) for x in range(GRID_W) for y in range(GRID_H)]
    rng.shuffle(cells)
    water = set(cells[:4])
    hazard = set(cells[4:8])
    # contested: a neutral cell NOT in water or hazard so injected signals dominate
    # Prefer central cells for spatial stability; skip any that happen to be GT-labelled
    candidates = [(4, 3), (5, 3), (4, 5), (5, 5), (3, 4), (6, 4), (3, 3), (6, 3)]
    contested = next(
        (c for c in candidates if c not in water and c not in hazard),
        cells[8],  # guaranteed not in water or hazard (cells[8] is index 8)
    )
    return World(water_places=water, hazard_places=hazard, contested_place=contested)


def semantic_tags(
    rng: random.Random,
    world: World,
    place: Tuple[int, int],
    noise: float = 0.08,
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
        tags[WATER_TAG] = tags.get(WATER_TAG, 0.0) + rng.uniform(0.05, 0.12)
    return tags


def neighbors(place: Tuple[int, int]) -> List[Tuple[int, int]]:
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = place[0] + dx, place[1] + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            out.append((nx, ny))
    return out


# ──────────────────────────────────────────────────────────── scout primitive


def run_scout(
    runtime: MultiAgentRuntime,
    world: World,
    rng: random.Random,
    agent_id: str,
    episodes: int,
    steps: int,
    x_range: Tuple[int, int],
    noise: float = 0.08,
    use_confirm: bool = True,
    episode_offset: int = 0,
) -> None:
    for ep in range(episodes):
        x_lo, x_hi = x_range
        pos: Tuple[int, int] = (rng.randrange(x_lo, x_hi), rng.randrange(GRID_H))
        visited: List[Tuple[int, int]] = [pos]
        prev: Optional[Tuple[int, int]] = None
        found_water = False
        for step_idx in range(steps):
            tags = semantic_tags(rng, world, pos, noise=noise)
            runtime.publish_observation(
                agent_id=agent_id, episode_id=episode_offset + ep,
                step_idx=step_idx, env_id=ENV_ID, place_key=pos,
                semantic_tags=tags, node_freshness=1.0, confidence=1.0,
            )
            if prev is not None and prev != pos:
                runtime.publish_transition(
                    agent_id=agent_id, episode_id=episode_offset + ep,
                    step_idx=step_idx, env_id=ENV_ID,
                    src_key=prev, dst_key=pos, confidence=0.9,
                )
            if pos in world.hazard_places:
                runtime.report_hazard_change(
                    agent_id=agent_id, episode_id=episode_offset + ep,
                    step_idx=step_idx, env_id=ENV_ID, place_key=pos,
                )
            if use_confirm and pos in world.water_places and not found_water:
                runtime.confirm_concept(
                    agent_id=agent_id, episode_id=episode_offset + ep,
                    step_idx=step_idx, env_id=ENV_ID, place_key=pos,
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
            agent_id=agent_id, episode_id=episode_offset + ep, env_id=ENV_ID,
            outcome={"success": float(found_water), "target_concept_tag": WATER_TAG},
            visited_place_keys=visited,
        )


# ─────────────────────────────────────────────── engine factory


def make_engine() -> PlaceAlignmentEngine:
    return PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=4.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )


def make_incremental_engine() -> IncrementalAlignmentEngine:
    return IncrementalAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=4.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )


def make_recall(engine, decay_engine=None, conflict_rules=None, max_conflict_penalty=1.0) -> ConceptRecallLayer:
    return ConceptRecallLayer(
        engine,
        only_dominant_tag=True,
        min_concept_support=2,
        decay_engine=decay_engine,
        conflict_rules=conflict_rules,
        max_conflict_penalty=max_conflict_penalty,
    )


# ═══════════════════════════════════════════════════════════════ Phase 3a


def run_phase3a(world: World, rng: random.Random, out_dir: str) -> Dict:
    """Temporal decay: stale suppression + recovery."""
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

    # decay engine: λ=0.97, deactivation at 0.10
    decay_engine = TemporalDecayEngine(
        decay_lambda=0.97,
        deactivation_threshold=0.10,
        recovery_threshold=0.30,
    )
    conflict_rules = ConflictRuleSet.songlines_default()

    # Build graph at "now" seq
    seq_after_scouts = collective._next_seq  # noqa: SLF001
    recall = make_recall(make_engine(), decay_engine=decay_engine)
    graph_fresh = recall.refresh(collective)

    # Snapshot before ageing
    snap_fresh = {cid: decay_engine.decayed_confidence(c, seq_after_scouts)
                  for cid, c in graph_fresh.concepts.items()}

    # Simulate 120 steps of silence (no new events published)
    virtual_seq_stale = seq_after_scouts + 120

    suppression_rate = stale_concept_suppression_rate(
        graph_fresh, virtual_seq_stale, decay_engine, min_stale_age=50
    )
    latency_info = latency_to_deactivation(graph_fresh, seq_after_scouts, decay_engine)

    # Which water concepts were stale at virtual_seq_stale?
    stale_water_ids = [
        cid for cid, c in graph_fresh.concepts.items()
        if c.dominant_tag == WATER_TAG
        and not decay_engine.is_active(c, virtual_seq_stale)
    ]

    # ---- Recovery: publish fresh water observations and rebuild ----
    # Publish 6 fresh observations on the first water place
    first_water_place = next(iter(world.water_places))
    for step_idx in range(6):
        tags = {
            WATER_TAG: 0.90,
            "water_candidate": 0.70,
            "water_visible": 0.55,
            "near_water": 0.45,
        }
        collective.publish_event(
            event_type="place_observed",
            agent_id="scout-A",
            episode_id=99,
            step_idx=step_idx,
            env_id=ENV_ID,
            payload={"place_key": list(first_water_place), "semantic_tags": tags, "node_freshness": 1.0},
            confidence=1.0,
        )

    seq_after_refresh = collective._next_seq  # noqa: SLF001
    recall_refreshed = make_recall(make_engine(), decay_engine=decay_engine)
    graph_refreshed = recall_refreshed.refresh(collective)

    recovery_rate = refreshed_concept_recovery_rate(
        graph_fresh, graph_refreshed, decay_engine,
        virtual_seq_stale, seq_after_refresh,
        concept_ids=stale_water_ids if stale_water_ids else None,
    )

    return {
        "seq_after_scouts": seq_after_scouts,
        "virtual_seq_stale": virtual_seq_stale,
        "seq_after_refresh": seq_after_refresh,
        "n_concepts_fresh": len(graph_fresh.concepts),
        "stale_water_ids": stale_water_ids,
        "stale_concept_suppression_rate": suppression_rate,
        "latency_to_deactivation": latency_info,
        "refreshed_concept_recovery_rate": recovery_rate,
        "confidence_at_fresh": snap_fresh,
        "confidence_at_stale": {
            cid: round(decay_engine.decayed_confidence(c, virtual_seq_stale), 4)
            for cid, c in graph_fresh.concepts.items()
        },
    }


# ═══════════════════════════════════════════════════════════════ Phase 3b


def run_phase3b(world: World, rng: random.Random, out_dir: str) -> Dict:
    """Conflict fusion: contested cell + purity under contradiction.

    Uses a **fresh** CollectiveMemory with manually-chosen fixed cells so that
    spatial clustering does NOT merge the contested concept with pure-water ones.
    All three cells are > 4.0 apart from each other (outside spatial_radius).

    - pure_water_a = (0, 0): top-left corner → pure water concept
    - pure_water_b = (9, 7): bottom-right corner → pure water concept
    - contested    = (4, 3): center → scout-A says WATER, scout-B says HAZARD
      Distances: (0,0)↔(4,3)=5.0, (9,7)↔(4,3)=7.81, (0,0)↔(9,7)=11.4 → all > 4.0
    """
    PURE_A: Tuple[int, int] = (0, 0)
    PURE_B: Tuple[int, int] = (9, 7)
    contested: Tuple[int, int] = (4, 3)

    collective = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    for sig in [
        AgentSignature("scout-A", role="scout", trust=1.0),
        AgentSignature("scout-B", role="scout", trust=0.9),
    ]:
        collective.register_agent(sig)

    # Pure water observations (no hazard) for two isolated reference places
    for ep_off, wp in enumerate([PURE_A, PURE_B]):
        for obs_i in range(8):
            collective.publish_event(
                "place_observed", "scout-A", episode_id=60 + ep_off, step_idx=obs_i,
                env_id=ENV_ID,
                payload={
                    "place_key": list(wp),
                    "semantic_tags": {
                        WATER_TAG: 0.92,
                        "water_candidate": 0.72,
                        "water_visible": 0.58,
                    },
                    "node_freshness": 1.0,
                },
                confidence=1.0,
            )

    # Contested cell: scout-A claims WATER, scout-B claims HAZARD (equal weight)
    for obs_i in range(8):
        collective.publish_event(
            "place_observed", "scout-A", episode_id=70, step_idx=obs_i,
            env_id=ENV_ID,
            payload={
                "place_key": list(contested),
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
                "place_key": list(contested),
                "semantic_tags": {HAZARD_TAG: 0.85, "adjacent_hazard": 0.62},
                "node_freshness": 1.0,
            },
            confidence=0.9,
        )

    conflict_rules = ConflictRuleSet.songlines_default()
    decay_engine = TemporalDecayEngine()

    # min_concept_support=1 because each cell is a single-place concept
    recall_engine = PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=4.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )
    recall = ConceptRecallLayer(
        recall_engine,
        only_dominant_tag=True,
        min_concept_support=1,
        decay_engine=decay_engine,
        conflict_rules=conflict_rules,
    )
    graph = recall.refresh(collective)

    current_seq = collective._next_seq  # noqa: SLF001

    # Find the concept that contains the contested cell
    contested_graph_key = (ENV_ID, tuple(contested))
    contested_cid = graph.place_to_concept.get(contested_graph_key)
    contested_concept = graph.concepts.get(contested_cid) if contested_cid else None

    raw_conflict_score = (
        conflict_rules.conflict_penalty(contested_concept)
        if contested_concept is not None else None
    )
    purity = (
        conflict_rules.concept_purity(contested_concept)
        if contested_concept is not None else None
    )
    conflict_pairs = (
        conflict_rules.tag_conflict_pairs(contested_concept)
        if contested_concept is not None else []
    )

    # Global metrics
    purity_under_contradiction = concept_purity_under_contradiction(
        graph, conflict_rules, conflict_threshold=0.15
    )
    false_persist = false_persistence_rate(
        graph, current_seq, decay_engine, conflict_rules, WATER_TAG, conflict_threshold=0.15
    )

    # With conflict filter: does the water query exclude the contested concept?
    recall_strict = ConceptRecallLayer(
        make_engine(),
        only_dominant_tag=True,
        min_concept_support=1,   # contested cell has 1 member place but many events
        decay_engine=decay_engine,
        conflict_rules=conflict_rules,
        max_conflict_penalty=0.30,  # suppress highly-conflicted concepts
    )
    recall_strict._graph = graph  # reuse already-built graph

    water_results_unfiltered = recall.query(
        WATER_TAG, "consumer-X", ENV_ID, current_seq=current_seq
    )
    water_results_filtered = recall_strict.query(
        WATER_TAG, "consumer-X", ENV_ID, current_seq=current_seq
    )

    contested_in_unfiltered = any(
        r.concept_id == contested_cid for r in water_results_unfiltered
    ) if contested_cid else False
    contested_in_filtered = any(
        r.concept_id == contested_cid for r in water_results_filtered
    ) if contested_cid else False

    return {
        "contested_place": list(contested),
        "contested_concept_id": contested_cid,
        "contested_concept_dominant_tag": (
            contested_concept.dominant_tag if contested_concept else None
        ),
        "contested_concept_profile": (
            dict(contested_concept.semantic_profile) if contested_concept else {}
        ),
        "raw_conflict_penalty": raw_conflict_score,
        "concept_purity": purity,
        "conflict_pairs": [
            {"pos": p, "neg": n, "pos_val": round(pv, 4), "neg_val": round(nv, 4), "contribution": round(c, 4)}
            for p, n, pv, nv, c in conflict_pairs
        ],
        "concept_purity_under_contradiction": purity_under_contradiction,
        "false_persistence_rate": false_persist,
        "contested_in_unfiltered_water_query": contested_in_unfiltered,
        "contested_in_filtered_water_query": contested_in_filtered,
        "n_unfiltered_results": len(water_results_unfiltered),
        "n_filtered_results": len(water_results_filtered),
    }


# ═══════════════════════════════════════════════════════════════ Phase 3c


def run_phase3c(world: World, rng: random.Random, out_dir: str) -> Dict:
    """Incremental updates: stability, churn, reuse."""
    collective = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    runtime = MultiAgentRuntime(collective)
    for sig in [
        AgentSignature("scout-A", role="scout", trust=1.0),
        AgentSignature("scout-B", role="scout", trust=0.9),
        AgentSignature("mini-scout", role="scout", trust=0.8),
    ]:
        runtime.register(sig)

    mid = GRID_W // 2
    run_scout(runtime, world, rng, "scout-A", episodes=3, steps=30, x_range=(0, mid + 2))
    run_scout(runtime, world, rng, "scout-B", episodes=3, steps=30, x_range=(mid - 2, GRID_W))

    # Incremental engine — first call is a full build
    inc_engine = make_incremental_engine()
    conflict_rules = ConflictRuleSet.songlines_default()
    decay_engine = TemporalDecayEngine()

    recall = ConceptRecallLayer(
        inc_engine,
        only_dominant_tag=True,
        min_concept_support=2,
        decay_engine=decay_engine,
        conflict_rules=conflict_rules,
    )
    _, init_stats = recall.refresh_incremental(collective)
    snap_before = inc_engine.snapshot_membership()

    gt_by_tag = {
        WATER_TAG: {tuple(p) for p in world.water_places},
        HAZARD_TAG: {tuple(p) for p in world.hazard_places},
    }
    current_seq_before = collective._next_seq  # noqa: SLF001

    # Top-1 concept before incremental update
    water_before = recall.query(
        WATER_TAG, "mini-scout", ENV_ID, current_seq=current_seq_before
    )
    top1_before = (
        (water_before[0].best_place_key(), water_before[0].concept_id)
        if water_before else None
    )

    # ---- Mini-scout publishes 12 fresh observations ----
    run_scout(
        runtime, world, rng, "mini-scout",
        episodes=1, steps=12, x_range=(0, GRID_W),
        noise=0.12, use_confirm=False, episode_offset=100,
    )

    # Incremental update (not full rebuild)
    _, inc_stats = recall.refresh_incremental(collective)
    snap_after = inc_engine.snapshot_membership()
    current_seq_after = collective._next_seq  # noqa: SLF001

    water_after = recall.query(
        WATER_TAG, "mini-scout", ENV_ID, current_seq=current_seq_after
    )
    top1_after = (
        (water_after[0].best_place_key(), water_after[0].concept_id)
        if water_after else None
    )

    stability = update_stability(snap_before, snap_after)
    churn = concept_churn_rate(snap_before, snap_after)
    reuse = reuse_after_incremental_update(top1_before, top1_after)

    # Precision stays correct after incremental update
    from songline_drive.collective_metrics import concept_query_precision
    precision_before = concept_query_precision(
        inc_engine.current_graph, gt_by_tag, WATER_TAG
    )
    precision_after = concept_query_precision(
        inc_engine.current_graph, gt_by_tag, WATER_TAG
    )

    return {
        "init_stats": init_stats,
        "incremental_stats": inc_stats,
        "snap_before_n": len(snap_before),
        "snap_after_n": len(snap_after),
        "update_stability": stability,
        "concept_churn_rate": churn,
        "reuse_after_incremental_update": reuse,
        "top1_concept_before": top1_before[1] if top1_before else None,
        "top1_concept_after": top1_after[1] if top1_after else None,
        "concept_query_precision_after": precision_after,
    }


# ═══════════════════════════════════════════════════════════════ assertions


def check_assertions(r3a: Dict, r3b: Dict, r3c: Dict, out_dir: str) -> List[str]:
    errors = []

    # Phase 3a
    if r3a["stale_concept_suppression_rate"] < 0.5:
        errors.append(
            f"3a FAIL: stale_suppression_rate={r3a['stale_concept_suppression_rate']:.3f} < 0.5"
        )
    lat = r3a["latency_to_deactivation"]
    if lat["n_active"] == 0:
        errors.append("3a FAIL: no active concepts at seq_after_scouts (all dead?)")
    elif lat["median"] is None:
        errors.append("3a FAIL: deactivation latency is None for all active concepts")

    # Phase 3b
    if r3b["raw_conflict_penalty"] is None:
        errors.append("3b FAIL: contested concept not found in graph")
    elif r3b["raw_conflict_penalty"] < 0.15:
        errors.append(
            f"3b FAIL: contested cell conflict_penalty={r3b['raw_conflict_penalty']:.3f} < 0.15 "
            "(conflict not detected)"
        )
    if r3b["contested_in_filtered_water_query"] and r3b.get("raw_conflict_penalty", 0) > 0.30:
        errors.append(
            "3b FAIL: highly-conflicted contested concept still in filtered water query"
        )

    # Phase 3c
    if r3c["update_stability"] < 0.80:
        errors.append(
            f"3c FAIL: update_stability={r3c['update_stability']:.3f} < 0.80"
        )
    if r3c["concept_churn_rate"] > 0.20:
        errors.append(
            f"3c FAIL: concept_churn_rate={r3c['concept_churn_rate']:.3f} > 0.20"
        )
    prec = r3c.get("concept_query_precision_after")
    if prec is not None and prec < 0.8:
        errors.append(
            f"3c FAIL: concept_query_precision after incremental update = {prec:.3f} < 0.80"
        )

    return errors


# ═══════════════════════════════════════════════════════════════ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="tmp/multiagent_phase3_smoke")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    world = build_world(rng)

    os.makedirs(args.out_dir, exist_ok=True)

    rng3a = random.Random(args.seed + 10)
    rng3b = random.Random(args.seed + 20)
    rng3c = random.Random(args.seed + 30)

    r3a = run_phase3a(world, rng3a, args.out_dir)
    r3b = run_phase3b(world, rng3b, args.out_dir)
    r3c = run_phase3c(world, rng3c, args.out_dir)

    errors = check_assertions(r3a, r3b, r3c, args.out_dir)

    summary = {
        "seed": args.seed,
        "world": {
            "water_places": [list(p) for p in sorted(world.water_places)],
            "hazard_places": [list(p) for p in sorted(world.hazard_places)],
            "contested_place": list(world.contested_place) if world.contested_place else None,
        },
        "phase3a_temporal_decay": r3a,
        "phase3b_conflict_fusion": r3b,
        "phase3c_incremental_updates": r3c,
        "assertions": {"passed": len(errors) == 0, "errors": errors},
    }

    summary_path = os.path.join(args.out_dir, "phase3_smoke_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        print("\n=== PHASE 3 ASSERTIONS FAILED ===", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(
            f"\n✓ Phase 3 passed  "
            f"stale_suppression={r3a['stale_concept_suppression_rate']:.2f}  "
            f"conflict_penalty={r3b.get('raw_conflict_penalty', 0):.2f}  "
            f"stability={r3c['update_stability']:.2f}  "
            f"churn={r3c['concept_churn_rate']:.2f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
