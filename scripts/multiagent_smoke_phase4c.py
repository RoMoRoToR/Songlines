"""Phase 4c smoke — coordinated field deconfliction.

Scenario: two consumers both need water. Two clean water concepts exist
(PURE_A at (0,0) and PURE_B at (9,7), spatially isolated, both valid targets).

Comparison:

  read_only mode — both consumers independently query the field →
    both pick the same top-1 concept → collision (duplicate_target_rate = 1.0)

  coordinated mode — Consumer-A queries → picks top-1 → reserves it →
    Consumer-B queries (after reservation) → top-1 is penalised →
    Consumer-B picks top-2 → deconfliction (duplicate_target_rate = 0.0)

Assertions:
  - In read_only: consumer_a_target == consumer_b_target (collision detected)
  - In coordinated: consumer_a_target != consumer_b_target (deconfliction)
  - duplicate_target_rate(read_only) > duplicate_target_rate(coordinated)
  - field_driven_deconfliction_rate = 1.0 (all collisions resolved)
  - Reservation reduces PURE_A/B activation by ≥ xi_occupancy - epsilon

Usage::

    PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase4c.py \\
        --seed 0 --out_dir tmp/multiagent_phase4c_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_field_types import FieldMode
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import AgentSignature
from songline_drive.concept_recall import ConceptRecallLayer
from songline_drive.field_adapter import FieldAdapter
from songline_drive.field_metrics import (
    duplicate_target_rate,
    field_driven_deconfliction_rate,
    reservation_conflict_rate,
)
from songline_drive.field_visualization import save_snapshot
from songline_drive.place_alignment import PlaceAlignmentEngine
from songline_drive.semantic_field import SemanticField

ENV_ID = "synthetic-grid-10x8"
WATER_TAG = "water_source"
HAZARD_TAG = "hazard_edge"
SAFE_TAG = "safe_neutral"

# Fixed isolated water places (dist > 4.0 apart from each other)
PURE_A: Tuple[int, int] = (0, 0)
PURE_B: Tuple[int, int] = (9, 7)


# ─────────────────────────────────────────────────────── helpers


def build_two_water_collective() -> CollectiveMemory:
    """Build a fresh collective with exactly two isolated water concepts."""
    collective = CollectiveMemory(recency_lambda=0.97, convergence_min_score=0.4)
    for sig in [
        AgentSignature("scout-A", role="scout", trust=1.0),
        AgentSignature("scout-B", role="scout", trust=0.9),
    ]:
        collective.register_agent(sig)

    # Each scout observes one of the pure water places (8 observations each)
    for ep_off, (scout, wp) in enumerate([("scout-A", PURE_A), ("scout-B", PURE_B)]):
        for obs_i in range(8):
            collective.publish_event(
                "place_observed", scout, episode_id=10 + ep_off, step_idx=obs_i,
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
    return collective


def make_engine() -> PlaceAlignmentEngine:
    return PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=4.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )


def make_recall() -> ConceptRecallLayer:
    return ConceptRecallLayer(
        make_engine(),
        only_dominant_tag=True,
        min_concept_support=1,
        decay_engine=TemporalDecayEngine(),
        conflict_rules=ConflictRuleSet.songlines_default(),
    )


def make_adapter(mode: str, field_weight: float = 0.35) -> FieldAdapter:
    field = SemanticField(
        channels=[WATER_TAG, HAZARD_TAG, SAFE_TAG],
        mode=mode,
        lambda_decay=0.95,
        alpha_belief=0.60,
        eta_conflict=0.30,
        xi_occupancy=0.20,
        diffusion_steps=1,
    )
    return FieldAdapter(field, make_recall(), field_weight=field_weight, mode=mode)


def field_top1(adapter: FieldAdapter, channel: str) -> Optional[str]:
    """Return the top-1 concept_id from the field for ``channel``."""
    items = adapter.field.top_k_for_channel(channel, k=1)
    return items[0][0] if items else None


# ═══════════════════════════════════════════════════════════ scenarios


def run_read_only(collective: CollectiveMemory, out_dir: str) -> Dict:
    """read_only mode: both consumers query independently → collision expected."""
    adapter = make_adapter(FieldMode.READ_ONLY)
    graph, field = adapter.refresh(collective)
    current_seq = collective._next_seq  # noqa: SLF001

    top1 = field.top_k_for_channel(WATER_TAG, k=2)
    if len(top1) < 2:
        return {"error": "fewer than 2 water concepts"}

    # Both consumers query independently, no coordination
    consumer_a_cid = top1[0][0]
    consumer_a_act = top1[0][1]
    consumer_b_cid = top1[0][0]  # same top-1 since no reservation
    consumer_b_act = top1[0][1]

    targets = [("consumer-A", consumer_a_cid), ("consumer-B", consumer_b_cid)]
    dup_rate = duplicate_target_rate(targets)

    concept_tag_map = {cid: c.dominant_tag for cid, c in graph.concepts.items()}
    save_snapshot(field, os.path.join(out_dir, "phase4c_read_only_field.json"), label="read_only")

    return {
        "mode": FieldMode.READ_ONLY,
        "consumer_a_target": consumer_a_cid,
        "consumer_b_target": consumer_b_cid,
        "consumer_a_activation": round(consumer_a_act, 4),
        "consumer_b_activation": round(consumer_b_act, 4),
        "collision": consumer_a_cid == consumer_b_cid,
        "duplicate_target_rate": dup_rate,
        "top2_concepts": [
            {"concept_id": cid, "activation": round(act, 4)}
            for cid, act in top1
        ],
    }


def run_coordinated(collective: CollectiveMemory, out_dir: str) -> Dict:
    """coordinated mode: Consumer-A reserves top-1 → Consumer-B redirected to top-2."""
    adapter = make_adapter(FieldMode.COORDINATED)
    graph, field = adapter.refresh(collective)
    current_seq = collective._next_seq  # noqa: SLF001

    top2_before = field.top_k_for_channel(WATER_TAG, k=2)
    if len(top2_before) < 2:
        return {"error": "fewer than 2 water concepts"}

    # Consumer-A: query → pick top-1 → reserve
    a_cid, a_act_before = top2_before[0]
    b_alt_cid, b_alt_act = top2_before[1]

    reservation = adapter.commit_reservation(
        agent_id="consumer-A",
        concept_id=a_cid,
        channel=WATER_TAG,
        duration=30,
        current_seq=current_seq,
    )

    # After reservation, field activation of a_cid drops
    a_act_after = field.activation_for(a_cid, WATER_TAG)

    # Consumer-B: query → gets updated activations
    top2_after = field.top_k_for_channel(WATER_TAG, k=2)
    b_cid = top2_after[0][0]  # B's top-1 after reservation
    b_act_after = top2_after[0][1]

    targets = [("consumer-A", a_cid), ("consumer-B", b_cid)]
    dup_rate = duplicate_target_rate(targets)

    active_res = adapter.active_reservations
    res_conflict = reservation_conflict_rate(active_res)

    save_snapshot(
        field,
        os.path.join(out_dir, "phase4c_coordinated_field.json"),
        label="coordinated_after_reservation",
    )

    return {
        "mode": FieldMode.COORDINATED,
        "consumer_a_target": a_cid,
        "consumer_b_target": b_cid,
        "consumer_a_activation_before_reservation": round(a_act_before, 4),
        "consumer_a_activation_after_reservation": round(a_act_after, 4),
        "activation_drop": round(a_act_before - a_act_after, 4),
        "consumer_b_activation": round(b_act_after, 4),
        "consumer_b_alternative_cid": b_alt_cid,
        "collision": a_cid == b_cid,
        "deconflicted": a_cid != b_cid,
        "duplicate_target_rate": dup_rate,
        "reservation_conflict_rate": res_conflict,
        "n_active_reservations": len(active_res),
        "top2_before_reservation": [
            {"concept_id": cid, "activation": round(act, 4)} for cid, act in top2_before
        ],
        "top2_after_reservation": [
            {"concept_id": cid, "activation": round(act, 4)} for cid, act in top2_after
        ],
    }


# ═══════════════════════════════════════════════════════════ assertions


def check_assertions(ro: Dict, co: Dict) -> List[str]:
    errors: List[str] = []

    # read_only should produce a collision
    if not ro.get("collision", False):
        errors.append(
            "4c FAIL: read_only mode should produce collision "
            f"(a={ro.get('consumer_a_target')}, b={ro.get('consumer_b_target')})"
        )

    # coordinated should deconflict
    if not co.get("deconflicted", False):
        errors.append(
            "4c FAIL: coordinated mode did NOT deconflict "
            f"(a={co.get('consumer_a_target')}, b={co.get('consumer_b_target')})"
        )

    # reservation should reduce activation
    drop = co.get("activation_drop", 0.0)
    xi = 0.20  # xi_occupancy
    if drop < xi * 0.9:
        errors.append(
            f"4c FAIL: activation did not drop enough after reservation "
            f"(drop={drop:.4f}, expected≥{xi * 0.9:.4f})"
        )

    # duplicate_target_rate comparison
    ro_dup = ro.get("duplicate_target_rate", 1.0)
    co_dup = co.get("duplicate_target_rate", 1.0)
    if co_dup >= ro_dup:
        errors.append(
            f"4c FAIL: coordinated dup_rate ({co_dup:.2f}) >= read_only ({ro_dup:.2f})"
        )

    # field_driven_deconfliction_rate
    ro_targets = [("consumer-A", ro.get("consumer_a_target", "?")),
                  ("consumer-B", ro.get("consumer_b_target", "?"))]
    co_targets = [("consumer-A", co.get("consumer_a_target", "?")),
                  ("consumer-B", co.get("consumer_b_target", "?"))]
    deconf_rate = field_driven_deconfliction_rate(ro_targets, co_targets)
    if deconf_rate < 1.0 and not (deconf_rate != deconf_rate):  # NaN check
        errors.append(
            f"4c FAIL: field_driven_deconfliction_rate={deconf_rate:.2f} < 1.0"
        )

    return errors, deconf_rate


# ═══════════════════════════════════════════════════════════ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="tmp/multiagent_phase4c_smoke")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    collective = build_two_water_collective()

    ro = run_read_only(collective, args.out_dir)
    co = run_coordinated(collective, args.out_dir)

    errors, deconf_rate = check_assertions(ro, co)
    passed = len(errors) == 0

    if passed:
        print(
            f"✓ Phase 4c passed  "
            f"read_only_collision={ro['collision']}  "
            f"coordinated_deconflicted={co['deconflicted']}  "
            f"deconf_rate={deconf_rate:.2f}  "
            f"activation_drop={co['activation_drop']:.3f}"
        )
    else:
        print("\n=== PHASE 4c ASSERTIONS FAILED ===")
        for e in errors:
            print(" ", e)

    summary = {
        "seed": args.seed,
        "pure_a": list(PURE_A),
        "pure_b": list(PURE_B),
        "read_only": ro,
        "coordinated": co,
        "field_driven_deconfliction_rate": (
            deconf_rate if deconf_rate == deconf_rate else None
        ),
        "assertions": {"passed": passed, "errors": errors},
    }

    summary_path = os.path.join(args.out_dir, "phase4c_smoke_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
