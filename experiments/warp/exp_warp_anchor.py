"""W0 acceptance — the misled-A regression anchor.

Re-runs the exp_4way_walk scenario (12x10 grid, waters at (3,8), (8,7),
(10,2), three agents A/B/C, peer broadcast K=4) through the warp-annotated
runner and checks the two acceptance criteria of design §8/W0:

  1. agent-A's lock on (3,8) is annotated phi = 1.0, strict W*
     (A knows about (3,8) only from C's broadcast — §7.5 of the
     2026-05-18 memory report);
  2. the independent variant produces exactly zero W* events.

As a free bonus (design §12) the same run gives a first H-W5 datapoint:
co-locked pressure on warp targets.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_anchor.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.big_experiment.memory_factory import build_memory
from experiments.warp.warp_runner import run_warp_episode
from multiagent_env import HAZARD, MultiAgentGridWorld, WATER

ENV_ID = "warp-anchor"
GRID_W, GRID_H = 12, 10
WATER_CELLS = [(3, 8), (8, 7), (10, 2)]
HAZARD_CELLS = [(5, 4), (5, 5), (6, 4), (6, 6)]
AGENT_SPEC = [
    ("agent-A", (0, 0), 0),   # NW, faces east
    ("agent-B", (11, 0), 2),  # NE, faces west
    ("agent-C", (0, 9), 3),   # SW, faces north
]
N_TICKS = 40
BROADCAST_K = 4


def build_anchor_env() -> MultiAgentGridWorld:
    env = MultiAgentGridWorld(
        width=GRID_W, height=GRID_H, step_limit=200,
        observation_radius=2, rng_seed=0,
    )
    for x, y in WATER_CELLS:
        env.set_cell(x, y, WATER)
    for x, y in HAZARD_CELLS:
        env.set_cell(x, y, HAZARD)
    for aid, xy, d in AGENT_SPEC:
        env.spawn(aid, start_xy=xy, target_tag="water_source", direction=d)
    return env


def run_variant(architecture: str):
    env = build_anchor_env()
    agent_ids = [s[0] for s in AGENT_SPEC]
    memory = build_memory(architecture, agent_ids, f"{ENV_ID}-{architecture}",
                          broadcast_every_k=BROADCAST_K)
    return run_warp_episode(
        env, agent_ids, WATER_CELLS, memory,
        step_limit=N_TICKS, variant_tag=architecture,
    )


def main() -> None:
    out_dir = "tmp/warp/anchor"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for arch in ("peer", "independent", "shared", "centralized", "csm"):
        metrics, log = run_variant(arch)
        results[arch] = {"metrics": metrics,
                         "events": [e.to_dict() for e in log.events]}
        print(f"\n── {arch} ──────────────────────────────────────────")
        print(f"  success={metrics['n_succeeded']}/3  "
              f"mean_t_succ={metrics['mean_t_succ']:.1f}  "
              f"M*-locks={metrics['n_m_star']}  "
              f"W*strict={metrics['n_w_star_strict']}  "
              f"W*soft={metrics['n_w_star_soft']}")
        for e in log.m_star_events():
            print(f"    tick={e.tick:>3} {e.agent_id} → {e.target_xy}  "
                  f"phi={e.phi:.3f} strict={e.w_star_strict} "
                  f"self_seen={e.self_ever_observed} "
                  f"radius={e.warp_radius_cells} "
                  f"src_age={e.source_snapshot_age} "
                  f"co_locked={e.co_locked} "
                  f"completed={e.completed} sources={sorted(e.per_source_mass)}")

    with open(os.path.join(out_dir, "anchor_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── acceptance checks ─────────────────────────────────────────
    print("\n" + "=" * 64)
    checks = []

    peer_events = [e for r in [results["peer"]] for e in r["events"]]
    a_38 = [e for e in peer_events
            if e["agent_id"] == "agent-A"
            and tuple(e["target_xy"]) == (3, 8)]
    anchor_ok = any(e["phi"] >= 0.999 and e["w_star_strict"] for e in a_38)
    checks.append(("ANCHOR: agent-A lock on (3,8) is strict W* (phi=1.0)",
                   anchor_ok,
                   f"found {len(a_38)} A→(3,8) locks, "
                   f"phis={[round(e['phi'], 3) for e in a_38]}"))

    indep = results["independent"]["metrics"]
    indep_ok = (indep["n_w_star_strict"] == 0 and indep["n_w_star_soft"] == 0)
    checks.append(("INDEPENDENT: exactly zero W* events", indep_ok,
                   f"strict={indep['n_w_star_strict']} "
                   f"soft={indep['n_w_star_soft']}"))

    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}   ({detail})")
        all_ok &= ok
    print("=" * 64)
    print("W0 ACCEPTANCE:", "PASS" if all_ok else "FAIL")
    print(f"Results: {out_dir}/anchor_results.json")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
