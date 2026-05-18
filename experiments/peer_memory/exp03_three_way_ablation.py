"""Experiment 03 — full 3-way ablation: independent vs centralized vs peer.

This is the comparison the reviewers asked for.  Same scenario, three
communication regimes:

  (1) Independent   — agents NEVER exchange anything.  Each acts on
                       its own observations only.
  (2) Centralized   — one ConsensusLayer aggregates all snapshots; each
                       agent queries the same central merged view.
  (3) Peer-to-peer  — periodic broadcast; each agent maintains its own
                       merged view from received messages.

Scenario
--------
Three agents in three disjoint regions, each observes ONE water cell:
  - scout-N observes water at (1, 1)
  - scout-E observes water at (8, 1)
  - scout-S observes water at (4, 6)

Each region is spatially isolated.  An agent's *local* graph contains
exactly its own water cell.  To know about other regions an agent must
receive that information from peers (modes 2 or 3) — under independent
(mode 1) the agent never learns about the other two water cells.

Metrics
-------
  - ``knowledge_coverage[agent]``: how many of the 3 water cells does
    this agent see in its query result?
  - ``avg_knowledge_coverage``: mean across agents
  - ``view_divergence``: for centralized this is always 0 (all agents
    see the same report); for peer it measures how much agents disagree
  - ``n_messages_sent``: counts comms cost (independent=0, centralized
    counts as 1 merge per tick, peer counts each broadcast)

Usage::

    PYTHONPATH=. .venv/bin/python experiments/peer_memory/exp03_three_way_ablation.py \\
        --n_ticks 6 --out_dir tmp/peer_exp03_ablation
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distributed_memory import DistributedRuntime
from peer_memory import PeerRuntime

ENV_ID = "ablation-grid"
WATER_TAG = "water_source"

# Three isolated water cells (pairwise distance > 4.0)
WATER_N: Tuple[int, int] = (1, 1)
WATER_E: Tuple[int, int] = (8, 1)
WATER_S: Tuple[int, int] = (4, 6)
ALL_WATER: List[Tuple[int, int]] = [WATER_N, WATER_E, WATER_S]


def _xy_close(a: Tuple[float, float], b: Tuple[int, int], tol: float = 0.5) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


# ─────────────────────────────────────────────────────── independent mode


def run_independent(n_ticks: int) -> Dict[str, Any]:
    """Each agent has private memory + no inter-agent communication."""
    from distributed_memory.agent_memory import AgentMemory

    agents = {
        "scout-N": (AgentMemory("scout-N", env_id=ENV_ID), WATER_N),
        "scout-E": (AgentMemory("scout-E", env_id=ENV_ID), WATER_E),
        "scout-S": (AgentMemory("scout-S", env_id=ENV_ID), WATER_S),
    }

    for aid, (mem, place) in agents.items():
        for i in range(8):
            mem.observe(place, {WATER_TAG: 0.93},
                        episode_id=1, step_idx=i, confidence=0.93)

    # Refresh each agent's local graph (no exchange)
    for _ in range(n_ticks):
        for aid, (mem, _) in agents.items():
            mem.refresh_local()

    knowledge = {}
    for aid, (mem, _) in agents.items():
        results = mem.local_query(WATER_TAG, top_k=10)
        known: List[Tuple[int, int]] = []
        for r in results:
            xy = r.centroid_xy
            if xy is None:
                continue
            for w in ALL_WATER:
                if _xy_close((float(xy[0]), float(xy[1])), w) and w not in known:
                    known.append(w)
        knowledge[aid] = known

    coverage = {aid: len(k) for aid, k in knowledge.items()}
    return {
        "mode": "independent",
        "n_messages_total": 0,
        "knowledge_per_agent": {aid: list(k) for aid, k in knowledge.items()},
        "coverage_per_agent": coverage,
        "avg_coverage": statistics.mean(coverage.values()),
        "view_divergence": 0.0,  # not applicable
    }


# ─────────────────────────────────────────────────────── centralized mode


def run_centralized(n_ticks: int) -> Dict[str, Any]:
    """Each agent has private memory; central ConsensusLayer aggregates."""
    rt = DistributedRuntime(env_id=ENV_ID, consensus_radius=4.0)
    rt.spawn_agent("scout-N")
    rt.spawn_agent("scout-E")
    rt.spawn_agent("scout-S")

    placements = {"scout-N": WATER_N, "scout-E": WATER_E, "scout-S": WATER_S}
    for aid, place in placements.items():
        for i in range(8):
            rt.observe(aid, place, {WATER_TAG: 0.93},
                       episode_id=1, step_idx=i, confidence=0.93)

    for _ in range(n_ticks):
        rt.tick()

    # In centralized mode, every agent queries the SAME central report
    report = rt.last_report
    centroids = [c.centroid_xy for c in report.distributed_concepts
                 if c.consensus_dominant_tag == WATER_TAG]
    known: List[Tuple[int, int]] = []
    for xy in centroids:
        for w in ALL_WATER:
            if _xy_close((float(xy[0]), float(xy[1])), w) and w not in known:
                known.append(w)

    knowledge = {aid: list(known) for aid in placements}
    coverage = {aid: len(k) for aid, k in knowledge.items()}
    return {
        "mode": "centralized",
        "n_messages_total": n_ticks,  # one centralized merge per tick
        "knowledge_per_agent": {aid: list(k) for aid, k in knowledge.items()},
        "coverage_per_agent": coverage,
        "avg_coverage": statistics.mean(coverage.values()),
        "view_divergence": 0.0,  # all agents see the same central report
    }


# ─────────────────────────────────────────────────────── peer mode


def run_peer(n_ticks: int, broadcast_every_k: int = 3) -> Dict[str, Any]:
    """Each agent has private memory + own merged view via periodic broadcast."""
    rt = PeerRuntime(env_id=ENV_ID, broadcast_every_k=broadcast_every_k,
                     consensus_radius=4.0)
    rt.spawn_agent("scout-N")
    rt.spawn_agent("scout-E")
    rt.spawn_agent("scout-S")

    placements = {"scout-N": WATER_N, "scout-E": WATER_E, "scout-S": WATER_S}
    for aid, place in placements.items():
        for i in range(8):
            rt.observe(aid, place, {WATER_TAG: 0.93},
                       episode_id=1, step_idx=i, confidence=0.93)

    for _ in range(n_ticks):
        rt.tick()

    knowledge: Dict[str, List[Tuple[int, int]]] = {}
    for aid in placements:
        view = rt.agent(aid).peer_view
        known: List[Tuple[int, int]] = []
        for c in view.distributed_concepts:
            if c.consensus_dominant_tag != WATER_TAG:
                continue
            for w in ALL_WATER:
                if _xy_close(c.centroid_xy, w) and w not in known:
                    known.append(w)
        knowledge[aid] = known

    coverage = {aid: len(k) for aid, k in knowledge.items()}

    # view_divergence: pairwise difference in number of known concepts
    cov_list = list(coverage.values())
    if len(cov_list) > 1:
        divergence = statistics.pstdev(cov_list)
    else:
        divergence = 0.0

    return {
        "mode": "peer",
        "broadcast_every_k": broadcast_every_k,
        "n_messages_total": rt.bus.n_broadcasts,
        "knowledge_per_agent": {aid: list(k) for aid, k in knowledge.items()},
        "coverage_per_agent": coverage,
        "avg_coverage": statistics.mean(coverage.values()),
        "view_divergence": divergence,
    }


# ─────────────────────────────────────────────────────── main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ticks", type=int, default=6)
    parser.add_argument("--broadcast_every_k", type=int, default=3)
    parser.add_argument("--out_dir", default="tmp/peer_exp03_ablation")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    independent = run_independent(args.n_ticks)
    centralized = run_centralized(args.n_ticks)
    peer = run_peer(args.n_ticks, args.broadcast_every_k)

    out_path = os.path.join(args.out_dir, "peer_exp03_ablation_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "independent": independent,
            "centralized": centralized,
            "peer": peer,
        }, f, indent=2)

    print(f"3-way ablation ({args.n_ticks} ticks)")
    print("=" * 80)
    header = f"{'mode':<14}  {'avg_cov/3':>10}  {'msgs':>5}  {'divergence':>10}  per-agent"
    print(header)
    print("-" * 80)
    for label, res in [
        ("independent", independent),
        ("centralized", centralized),
        ("peer", peer),
    ]:
        cov = res["coverage_per_agent"]
        cov_str = "  ".join(f"{aid.split('-')[-1]}={v}" for aid, v in cov.items())
        print(
            f"{label:<14}  {res['avg_coverage']:>10.2f}  "
            f"{res['n_messages_total']:>5d}  "
            f"{res['view_divergence']:>10.3f}  {cov_str}"
        )

    print(f"\nSummary → {out_path}")

    # Assertions on the comparison
    assert independent["avg_coverage"] == 1.0, (
        f"Independent should have coverage=1 per agent (own region only): "
        f"got {independent['avg_coverage']}"
    )
    assert centralized["avg_coverage"] == 3.0, (
        f"Centralized should expose all 3 cells to every agent: "
        f"got {centralized['avg_coverage']}"
    )
    assert peer["avg_coverage"] >= 2.0, (
        f"Peer with periodic broadcast should expose most cells: "
        f"got {peer['avg_coverage']}"
    )
    # The headline: peer-to-peer reaches comparable coverage to centralized
    # without a central layer
    print(
        f"\n✓ Reviewers' triad complete: "
        f"independent={independent['avg_coverage']:.1f}/3, "
        f"centralized={centralized['avg_coverage']:.1f}/3, "
        f"peer={peer['avg_coverage']:.1f}/3"
    )


if __name__ == "__main__":
    main()
