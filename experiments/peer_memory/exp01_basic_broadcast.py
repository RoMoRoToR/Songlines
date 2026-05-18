"""Experiment 01 — basic periodic broadcast.

Two agents observe different places.  No central layer.  After K ticks,
each agent broadcasts.  Each agent's private peer_view should converge
to include both places.

Setup
-----
scout-A observes PLACE_A=(0, 0), water (8 obs).
scout-B observes PLACE_B=(9, 7), water (8 obs).
broadcast_every_k = 3.

Expectations after 3 ticks (one broadcast cycle done)
------------------------------------------------------
  - Both agents' peer_view contains 2 concepts (own + peer's)
  - Both peer_views show ``contributing_peer_ids`` non-empty
  - But each peer_view is a SEPARATE object (no sharing)

Sanity checks
-------------
  - Replacing the bus with a dummy that drops all messages → peer_views
    only contain the agent's own concept (proves no back-channel)

Usage::

    PYTHONPATH=. .venv/bin/python experiments/peer_memory/exp01_basic_broadcast.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from peer_memory import PeerRuntime

ENV_ID = "peer-exp01"
WATER_TAG = "water_source"
PLACE_A: Tuple[int, int] = (0, 0)
PLACE_B: Tuple[int, int] = (9, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/peer_exp01")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rt = PeerRuntime(env_id=ENV_ID, broadcast_every_k=3)
    rt.spawn_agent("scout-A")
    rt.spawn_agent("scout-B")

    for i in range(8):
        rt.observe("scout-A", PLACE_A, {WATER_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)
        rt.observe("scout-B", PLACE_B, {WATER_TAG: 0.92},
                   episode_id=1, step_idx=i, confidence=0.92)

    # 3 ticks: first broadcast at tick 3
    for _ in range(3):
        rt.tick()

    va = rt.agent("scout-A").peer_view
    vb = rt.agent("scout-B").peer_view

    out_path = os.path.join(args.out_dir, "peer_exp01_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "agent_a_view": va.to_dict(),
            "agent_b_view": vb.to_dict(),
            "broadcast_history": rt.broadcast_history,
            "stats": rt.stats(),
        }, f, indent=2)

    # Assertions
    assert len(va.distributed_concepts) == 2, (
        f"agent-A peer_view should have 2 concepts (own + peer), "
        f"got {len(va.distributed_concepts)}"
    )
    assert len(vb.distributed_concepts) == 2, (
        f"agent-B peer_view should have 2 concepts, "
        f"got {len(vb.distributed_concepts)}"
    )
    assert va.contributing_peer_ids == ["scout-B"], (
        f"A should have heard from B: {va.contributing_peer_ids}"
    )
    assert vb.contributing_peer_ids == ["scout-A"]
    # Peer views are SEPARATE objects
    assert va is not vb
    assert va.owner_id == "scout-A"
    assert vb.owner_id == "scout-B"

    print(
        f"✓ Peer Exp01 passed  "
        f"A heard from {va.contributing_peer_ids}, A sees {len(va.distributed_concepts)} concepts  |  "
        f"B heard from {vb.contributing_peer_ids}, B sees {len(vb.distributed_concepts)} concepts"
    )
    print(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
