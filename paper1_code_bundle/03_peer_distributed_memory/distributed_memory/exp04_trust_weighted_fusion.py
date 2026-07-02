"""Experiment 04 — trust-weighted fusion.

Three agents observe the same place but disagree on tag.  Two
high-trust agents agree on water_source; one low-trust agent claims
hazard_edge.  The consensus should favour the majority + high trust.

Setup
-----
scout-A (trust=1.0): SHARED → water_source (8 obs, conf=0.95)
scout-B (trust=1.0): SHARED → water_source (8 obs, conf=0.95)
scout-C (trust=0.2): SHARED → hazard_edge  (8 obs, conf=0.95)

Expectations
------------
  - Aligned distributed concept at SHARED
  - consensus_dominant_tag == "water_source" (majority of trust mass)
  - water_source mass in consensus_profile > hazard_edge mass
  - scout-C still appears in contributions (preserves provenance)
  - Disagreement is flagged but doesn't override the majority

Comparison
----------
Then re-run with reversed trusts (A=0.2, B=0.2, C=1.0) — consensus
should flip to hazard_edge.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/distributed_memory/exp04_trust_weighted_fusion.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distributed_memory import DistributedRuntime

ENV_ID = "exp04-grid"
WATER_TAG = "water_source"
HAZARD_TAG = "hazard_edge"
SHARED: Tuple[int, int] = (4, 3)


def run_scenario(trusts: Dict[str, float]) -> Dict:
    rt = DistributedRuntime(
        env_id=ENV_ID,
        consensus_radius=4.0,
        consensus_profile_threshold=0.0,
    )
    rt.spawn_agent("scout-A", trust=trusts["scout-A"])
    rt.spawn_agent("scout-B", trust=trusts["scout-B"])
    rt.spawn_agent("scout-C", trust=trusts["scout-C"])

    for i in range(8):
        rt.observe("scout-A", SHARED, {WATER_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)
        rt.observe("scout-B", SHARED, {WATER_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)
        rt.observe("scout-C", SHARED, {HAZARD_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)

    report = rt.tick()
    aligned = [c for c in report.distributed_concepts if c.n_agents == 3]
    if not aligned:
        # Fall back: there could be 2-agent + 1-agent clusters if alignment failed
        aligned = sorted(
            report.distributed_concepts, key=lambda c: c.n_agents, reverse=True,
        )[:1]
    dc = aligned[0]

    return {
        "trusts": trusts,
        "n_agents_in_consensus": dc.n_agents,
        "consensus_dominant_tag": dc.consensus_dominant_tag,
        "consensus_profile": dc.consensus_profile,
        "consensus_confidence": dc.consensus_confidence,
        "inter_agent_agreement": dc.inter_agent_agreement,
        "contributions": [
            {
                "agent": c.agent_id,
                "trust": round(c.trust, 4),
                "local_tag": c.local_dominant_tag,
            }
            for c in dc.contributions
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/distributed_exp04")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenario_a = run_scenario({"scout-A": 1.0, "scout-B": 1.0, "scout-C": 0.2})
    scenario_b = run_scenario({"scout-A": 0.2, "scout-B": 0.2, "scout-C": 1.0})

    out_path = os.path.join(args.out_dir, "exp04_summary.json")
    with open(out_path, "w") as f:
        json.dump({"scenario_a_majority_water": scenario_a,
                   "scenario_b_majority_hazard": scenario_b}, f, indent=2)

    # Scenario A: high-trust majority claims water
    assert scenario_a["consensus_dominant_tag"] == WATER_TAG, (
        f"Scenario A: expected {WATER_TAG}, got {scenario_a['consensus_dominant_tag']}"
    )
    water_a = scenario_a["consensus_profile"].get(WATER_TAG, 0)
    hazard_a = scenario_a["consensus_profile"].get(HAZARD_TAG, 0)
    assert water_a > hazard_a, (
        f"Scenario A: water mass {water_a:.3f} should dominate over hazard {hazard_a:.3f}"
    )

    # Scenario B: trust flipped → hazard wins
    assert scenario_b["consensus_dominant_tag"] == HAZARD_TAG, (
        f"Scenario B: expected {HAZARD_TAG}, got {scenario_b['consensus_dominant_tag']}"
    )

    # Both scenarios: all three agents must be in contributions (provenance preserved)
    assert scenario_a["n_agents_in_consensus"] == 3
    assert scenario_b["n_agents_in_consensus"] == 3

    print(
        f"✓ Exp04 passed  "
        f"scenario_a: tag={scenario_a['consensus_dominant_tag']} "
        f"(water={water_a:.3f}, hazard={hazard_a:.3f})  |  "
        f"scenario_b: tag={scenario_b['consensus_dominant_tag']} (trusts flipped)"
    )
    print(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
