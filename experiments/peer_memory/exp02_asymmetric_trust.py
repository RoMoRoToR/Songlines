"""Experiment 02 — asymmetric trust leads to different peer_views.

Two agents observe the SAME contested place but tag it differently.
agent-A trusts agent-B at 0.9; agent-B trusts agent-A at 0.2.
Both agents broadcast.

Expectations
------------
  - Each agent forms a private peer_view.
  - In agent-A's view, the contested concept's profile is biased toward
    B's tag (because A trusts B highly).
  - In agent-B's view, A's contribution is downweighted; B's view is
    biased toward B's OWN tag (because A is barely trusted).
  - The two peer_views are different — proving no central consensus.

This is the headline property of peer-to-peer: same world, different
beliefs depending on who trusts whom.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/peer_memory/exp02_asymmetric_trust.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from peer_memory import PeerRuntime

ENV_ID = "peer-exp02"
WATER_TAG = "water_source"
HAZARD_TAG = "hazard_edge"
CONTESTED: Tuple[int, int] = (4, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/peer_exp02")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rt = PeerRuntime(env_id=ENV_ID, broadcast_every_k=2,
                     consensus_radius=4.0, profile_similarity_threshold=0.0)
    a = rt.spawn_agent("agent-A")
    b = rt.spawn_agent("agent-B")

    # Asymmetric trust BEFORE any merge happens
    a.trust.set("agent-B", 0.90)
    b.trust.set("agent-A", 0.20)

    # agent-A sees CONTESTED as water; agent-B sees it as hazard
    for i in range(8):
        rt.observe("agent-A", CONTESTED, {WATER_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)
        rt.observe("agent-B", CONTESTED, {HAZARD_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)

    # 2 ticks: broadcast at tick 2
    for _ in range(2):
        rt.tick()

    va = a.peer_view
    vb = b.peer_view

    # Find the merged contested concept in each view
    def contested_concept(view):
        for c in view.distributed_concepts:
            if c.n_agents == 2:  # merged across both agents
                return c
        return None

    contested_a = contested_concept(va)
    contested_b = contested_concept(vb)

    summary = {
        "agent_a_trust": a.trust.snapshot(),
        "agent_b_trust": b.trust.snapshot(),
        "agent_a_view_contested": contested_a.to_dict() if contested_a else None,
        "agent_b_view_contested": contested_b.to_dict() if contested_b else None,
        "agent_a_view": va.to_dict(),
        "agent_b_view": vb.to_dict(),
    }
    out_path = os.path.join(args.out_dir, "peer_exp02_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    assert contested_a is not None, "agent-A should have merged contested concept"
    assert contested_b is not None, "agent-B should have merged contested concept"

    # Profile mass on each tag should differ between the two peer_views
    water_a = contested_a.consensus_profile.get(WATER_TAG, 0)
    hazard_a = contested_a.consensus_profile.get(HAZARD_TAG, 0)
    water_b = contested_b.consensus_profile.get(WATER_TAG, 0)
    hazard_b = contested_b.consensus_profile.get(HAZARD_TAG, 0)

    # agent-A trusts B highly → hazard mass (B's tag) significant in A's view
    # agent-B distrusts A → water mass (A's tag) downweighted in B's view
    a_hazard_share = hazard_a / max(1e-9, water_a + hazard_a)
    b_water_share = water_b / max(1e-9, water_b + hazard_b)

    print(
        f"agent-A profile (trusts B at 0.90): water={water_a:.3f}, hazard={hazard_a:.3f} "
        f"→ hazard share = {a_hazard_share:.3f}"
    )
    print(
        f"agent-B profile (trusts A at 0.20): water={water_b:.3f}, hazard={hazard_b:.3f} "
        f"→ water share = {b_water_share:.3f}"
    )

    # The key invariant: trust asymmetry produces DIFFERENT profiles
    assert a_hazard_share > b_water_share, (
        f"A (trusts B high) should let hazard mass dominate more than "
        f"B (trusts A low) lets water mass dominate: "
        f"A_hazard_share={a_hazard_share:.3f}, B_water_share={b_water_share:.3f}"
    )

    # Both peer_views exist as separate objects
    assert va is not vb
    assert va.owner_id != vb.owner_id

    print(
        f"\n✓ Peer Exp02 passed  "
        f"A_view hazard_share={a_hazard_share:.3f} > B_view water_share={b_water_share:.3f}  "
        f"(asymmetric trust → divergent beliefs)"
    )
    print(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
