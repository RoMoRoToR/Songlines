"""Experiment 05 — partial observability + knowledge transfer via consensus.

Each agent observes only its own region.  Locally, each only knows
about places it visited.  After consensus tick(), each agent can query
the collective view and discover places observed by others.

Setup
-----
scout-A observes PLACE_A=(0, 0) — water.
scout-B observes PLACE_B=(9, 7) — water.
scout-C observes PLACE_C=(4, 0) — water.

Expectations
------------
  - scout-A.local_query(water) returns 1 concept (own).
  - scout-B.local_query(water) returns 1 concept (own).
  - scout-C.local_query(water) returns 1 concept (own).
  - After tick(): collective_query(water) returns 3 distributed concepts.
  - Each agent can access knowledge from all three regions via consensus.
  - n_isolated == 3 (no spatial overlap → all 3 concepts are isolated
    from the consensus perspective).

This demonstrates that the consensus layer acts as a **knowledge bus**:
locally private knowledge becomes globally accessible after fusion,
without changing any local graph.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/distributed_memory/exp05_partial_observability.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distributed_memory import DistributedRuntime

ENV_ID = "exp05-grid"
WATER_TAG = "water_source"
PLACE_A: Tuple[int, int] = (0, 0)
PLACE_B: Tuple[int, int] = (9, 7)
PLACE_C: Tuple[int, int] = (5, 0)  # distance from A=(0,0) is 5.0 > radius 4.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/distributed_exp05")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rt = DistributedRuntime(env_id=ENV_ID, consensus_radius=4.0)
    rt.spawn_agent("scout-A", trust=1.0)
    rt.spawn_agent("scout-B", trust=1.0)
    rt.spawn_agent("scout-C", trust=1.0)

    obs_specs = [
        ("scout-A", PLACE_A),
        ("scout-B", PLACE_B),
        ("scout-C", PLACE_C),
    ]
    for agent_id, place in obs_specs:
        for i in range(8):
            rt.observe(agent_id, place, {WATER_TAG: 0.93},
                       episode_id=1, step_idx=i, confidence=0.93)

    # Local queries BEFORE consensus tick — each agent only knows its own region
    pre_local: Dict[str, list] = {}  # type: ignore
    pre_local = {}
    for agent_id in ["scout-A", "scout-B", "scout-C"]:
        # Need to refresh local graph at least once for local_query to work
        rt.agent(agent_id).refresh_local()
        results = rt.local_query(agent_id, WATER_TAG, top_k=5)
        pre_local[agent_id] = [
            {"concept_id": r.concept_id, "score": round(r.score, 4)} for r in results
        ]

    # Now run consensus tick
    report = rt.tick()
    collective_top = rt.collective_query(WATER_TAG, top_k=5)

    out_path = os.path.join(args.out_dir, "exp05_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "pre_consensus_local_queries": pre_local,
            "post_consensus_report": report.to_dict(),
            "collective_top_k_water": [c.to_dict() for c in collective_top],
        }, f, indent=2)

    # Assertions: each agent locally sees exactly 1 concept
    for agent_id in ["scout-A", "scout-B", "scout-C"]:
        assert len(pre_local[agent_id]) == 1, (
            f"{agent_id} should locally see exactly 1 water concept, got {len(pre_local[agent_id])}"
        )

    # Consensus: 3 isolated concepts (no spatial overlap between regions)
    assert report.n_aligned == 0, (
        f"Expected 0 aligned concepts (regions are spatially disjoint), got {report.n_aligned}"
    )
    assert report.n_isolated == 3, (
        f"Expected 3 isolated concepts, got {report.n_isolated}"
    )

    # Collective query returns all 3 regions
    assert len(collective_top) == 3, (
        f"Collective query should expose 3 water concepts, got {len(collective_top)}"
    )

    # Check that all three centroids are present
    centroids = sorted([tuple(round(v, 1) for v in c.centroid_xy) for c in collective_top])
    expected = sorted([PLACE_A, PLACE_B, PLACE_C])
    assert centroids == [(float(x), float(y)) for x, y in expected], (
        f"Collective centroids {centroids} should cover all of {expected}"
    )

    print(
        f"✓ Exp05 passed  "
        f"each agent locally sees 1 concept  "
        f"collective exposes {len(collective_top)} regions  "
        f"(n_aligned={report.n_aligned}, n_isolated={report.n_isolated})"
    )
    print(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
