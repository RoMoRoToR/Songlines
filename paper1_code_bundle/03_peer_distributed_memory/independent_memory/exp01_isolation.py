"""Experiment 01 — fully isolated agents have no cross-knowledge.

Three agents in three disjoint regions.  Each observes only its own
water cell.  After N ticks of local refresh:

  - Each agent's local_query returns exactly ONE water concept (own region)
  - No agent has any knowledge of the other two regions
  - The IndependentRuntime exposes NO collective_query method

Additionally we verify the "isolation contract" at the API level: calling
snapshot() / broadcast() / receive() on an IndependentAgent raises.

This experiment is the lower-bound baseline against which
``experiments/peer_memory/exp03_three_way_ablation.py`` compares
centralized and peer modes.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/independent_memory/exp01_isolation.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from independent_memory import IndependentRuntime

ENV_ID = "indep-exp01"
WATER_TAG = "water_source"
WATER_N: Tuple[int, int] = (1, 1)
WATER_E: Tuple[int, int] = (8, 1)
WATER_S: Tuple[int, int] = (4, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/indep_exp01")
    parser.add_argument("--n_ticks", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rt = IndependentRuntime(env_id=ENV_ID)
    rt.spawn_agent("scout-N")
    rt.spawn_agent("scout-E")
    rt.spawn_agent("scout-S")

    placements = {"scout-N": WATER_N, "scout-E": WATER_E, "scout-S": WATER_S}
    for aid, place in placements.items():
        for i in range(8):
            rt.observe(aid, place, {WATER_TAG: 0.93},
                       episode_id=1, step_idx=i, confidence=0.93)

    for _ in range(args.n_ticks):
        rt.tick()

    # Each agent should see exactly 1 water concept (its own)
    knowledge = {}
    for aid in placements:
        results = rt.local_query(aid, WATER_TAG, top_k=5)
        seen_xy = [r.centroid_xy for r in results if r.centroid_xy is not None]
        knowledge[aid] = seen_xy

    out_path = os.path.join(args.out_dir, "indep_exp01_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_ticks": args.n_ticks,
            "knowledge_per_agent": {
                aid: [list(xy) for xy in xys] for aid, xys in knowledge.items()
            },
            "stats": rt.stats(),
        }, f, indent=2)

    # Assertions
    for aid, xys in knowledge.items():
        assert len(xys) == 1, (
            f"{aid} should know exactly 1 concept (own region), got {len(xys)}"
        )
    # Each agent's centroid matches ITS placement
    expected = {
        "scout-N": WATER_N, "scout-E": WATER_E, "scout-S": WATER_S,
    }
    for aid, want in expected.items():
        got = knowledge[aid][0]
        assert abs(got[0] - want[0]) < 0.5 and abs(got[1] - want[1]) < 0.5, (
            f"{aid}: expected centroid near {want}, got {got}"
        )

    # Verify no collective query is available
    assert not hasattr(rt, "collective_query"), (
        "IndependentRuntime must not expose collective_query"
    )
    assert not hasattr(rt, "consensus"), (
        "IndependentRuntime must not have a ConsensusLayer"
    )
    assert not hasattr(rt, "bus"), (
        "IndependentRuntime must not have a BroadcastBus"
    )

    # Verify forbidden agent operations raise
    a = rt.agent("scout-N")
    for forbidden in ("snapshot", "broadcast", "receive"):
        try:
            getattr(a, forbidden)()
            raise AssertionError(f"{forbidden}() should have raised")
        except RuntimeError:
            pass  # expected

    print(
        f"✓ Independent Exp01 passed  "
        f"{len(knowledge)} agents, each knows exactly 1 region  "
        f"(N={knowledge['scout-N'][0]}, E={knowledge['scout-E'][0]}, S={knowledge['scout-S'][0]})  "
        f"no collective query exposed"
    )
    print(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
