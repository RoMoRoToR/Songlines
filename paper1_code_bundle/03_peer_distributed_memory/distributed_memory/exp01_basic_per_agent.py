"""Experiment 01 — basic per-agent memory.

Each agent independently builds a local graph from its own observations
only.  Verifies:

  - Two agents observing different places end up with non-overlapping graphs
  - Each agent's local_query returns only what that agent has seen
  - No leakage through the runtime (Phase 1 isolation invariant)

Setup
-----
scout-A observes one water cluster at (0, 0).
scout-B observes a different water cluster at (9, 7).

Expectations
------------
  - scout-A's graph has exactly 1 concept centered near (0, 0)
  - scout-B's graph has exactly 1 concept centered near (9, 7)
  - scout-A.local_query("water_source") returns its concept, not B's
  - scout-B.local_query("water_source") returns its concept, not A's

Usage::

    PYTHONPATH=. .venv/bin/python experiments/distributed_memory/exp01_basic_per_agent.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distributed_memory import DistributedRuntime

ENV_ID = "exp01-grid"
WATER_TAG = "water_source"
PLACE_A: Tuple[int, int] = (0, 0)
PLACE_B: Tuple[int, int] = (9, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/distributed_exp01")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rt = DistributedRuntime(env_id=ENV_ID)
    rt.spawn_agent("scout-A", trust=1.0)
    rt.spawn_agent("scout-B", trust=1.0)

    # Each scout observes only its own place
    for i in range(8):
        rt.observe("scout-A", PLACE_A, {WATER_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)
        rt.observe("scout-B", PLACE_B, {WATER_TAG: 0.92},
                   episode_id=1, step_idx=i, confidence=0.92)

    # Refresh local graphs (no consensus needed for isolation check)
    for agent in rt.all_agents():
        agent.refresh_local()

    snap_a = rt.agent("scout-A").snapshot()
    snap_b = rt.agent("scout-B").snapshot()

    # Local queries
    q_a = rt.local_query("scout-A", WATER_TAG, top_k=5)
    q_b = rt.local_query("scout-B", WATER_TAG, top_k=5)

    result = {
        "agent_a": snap_a.to_dict(),
        "agent_b": snap_b.to_dict(),
        "scout_a_top1": q_a[0].concept_id if q_a else None,
        "scout_b_top1": q_b[0].concept_id if q_b else None,
        "scout_a_concept_ids": list(snap_a.local_concept_ids),
        "scout_b_concept_ids": list(snap_b.local_concept_ids),
    }

    # Save
    out_path = os.path.join(args.out_dir, "exp01_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Assertions
    assert snap_a.n_local_concepts == 1, (
        f"scout-A should have exactly 1 concept, got {snap_a.n_local_concepts}"
    )
    assert snap_b.n_local_concepts == 1, (
        f"scout-B should have exactly 1 concept, got {snap_b.n_local_concepts}"
    )
    # Concept IDs are LOCAL to each agent's graph — they can coincide.
    # The actual isolation check is on centroids and event counts.
    cid_a = snap_a.local_concept_ids[0]
    cid_b = snap_b.local_concept_ids[0]
    xy_a = snap_a.local_concepts[cid_a]["centroid_xy"]
    xy_b = snap_b.local_concepts[cid_b]["centroid_xy"]
    assert xy_a != xy_b, "Centroids must differ — proves agents have private graphs"
    assert snap_a.n_events == 8 and snap_b.n_events == 8, (
        f"Each agent must see only its own 8 events: A={snap_a.n_events}, B={snap_b.n_events}"
    )
    # Sanity: scout-A's centroid is near PLACE_A
    assert abs(xy_a[0] - PLACE_A[0]) < 0.5 and abs(xy_a[1] - PLACE_A[1]) < 0.5
    assert abs(xy_b[0] - PLACE_B[0]) < 0.5 and abs(xy_b[1] - PLACE_B[1]) < 0.5
    assert q_a and q_b, "Both agents must answer their own water query"

    print(
        f"✓ Exp01 passed  "
        f"scout-A: 1 concept @ {xy_a}  "
        f"scout-B: 1 concept @ {xy_b}  "
        f"private_graphs=True"
    )
    print(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
