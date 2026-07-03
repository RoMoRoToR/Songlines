"""R1 — the gain(D) ablation: route-warp value grows with world detour.

Design (FRONTIER_ROUTE_WARP §3): three arms on IDENTICAL mazes
(paired by seed) across detour-factor buckets:

    blind  — no foreign evidence, standard planner (exploration);
    place  — foreign WATER LOCATION only, standard planner;
    route  — foreign traversed EDGES too, follower commits the path
             (strict RW*: the whole route is someone else's song).

Detour factor D = BFS-path length / manhattan distance is MEASURED per
(maze, start, water) pair and bucketed:

    D1 bucket: open grid (D = 1.0 exactly)      — negative control;
    D2 bucket: sparse comb, pair with D in [1.6, 2.8);
    D4 bucket: dense comb,  pair with D in [3.0, 8.0).

Registered predictions (written before episodes):
  R1-neg: at D = 1 the route arm buys nothing:
          |mean gain(route vs place)| <= 3 ticks (turn-order noise).
  R2:     gain(route vs place) > 0 with bootstrap CI excluding 0 at
          both D >= 2 buckets, and mean gain grows from D2 to D4.
  (exploratory) place-vs-blind: the R0 pathology — place evidence
  without route evidence can be WORSE than blind exploration at high D.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_route_warp_r1.py \\
        [--seeds 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.warp.exp_route_warp_r0 import (
    GRID_H, GRID_W, RoutePeerMemory, WATER_TAG, bfs_path, blind_arm,
    detour_factor, follow_route_arm, place_arm, witness_walk,
)
from multiagent_env import MultiAgentGridWorld, WALL, WATER

GridXY = Tuple[int, int]
OUT_DIR = "tmp/warp/r1_route_gain"
STEP_LIMIT = 300
BUCKETS = {
    "D1": (1.0, 1.0),
    "D15": (1.05, 1.6),   # probe bucket: locates the place-collapse cliff
    "D2": (1.6, 2.8),
    "D4": (3.0, 8.0),
}


# ───────────────────────────────────── maze generation


KIND_OF = {"D1": "open", "D15": "sparse", "D2": "sparse", "D4": "dense"}


def build_maze(kind: str, seed: int) -> MultiAgentGridWorld:
    """kind: open | sparse (walls every 3rd col) | dense (every 2nd)."""
    env = MultiAgentGridWorld(width=GRID_W, height=GRID_H,
                              step_limit=STEP_LIMIT,
                              observation_radius=2, rng_seed=seed)
    rng = np.random.default_rng(seed)
    if kind == "open":
        return env
    walls_x = list(range(2, GRID_W - 1, 3)) if kind == "sparse" \
        else list(range(1, GRID_W - 1, 2))
    for wx in walls_x:
        gap_y = int(rng.integers(0, GRID_H))
        for y in range(GRID_H):
            if y != gap_y:
                env.set_cell(wx, y, WALL)
    return env


def pick_pair(env: MultiAgentGridWorld, bucket: Tuple[float, float],
              seed: int) -> Optional[Tuple[GridXY, GridXY, float]]:
    """Sample (start, water) pairs; return the first whose measured D
    lands in the bucket (manhattan >= 8 for stable D)."""
    rng = np.random.default_rng(90_000 + seed)
    passable = [(x, y) for x in range(GRID_W) for y in range(GRID_H)
                if env.cell(x, y) != WALL]
    lo, hi = bucket
    for _ in range(400):
        s = passable[int(rng.integers(0, len(passable)))]
        w = passable[int(rng.integers(0, len(passable)))]
        if abs(s[0] - w[0]) + abs(s[1] - w[1]) < 8:
            continue
        d = detour_factor(env, s, w)
        if d == float("inf"):
            continue
        if lo <= d <= hi:
            return s, w, d
    return None


# ───────────────────────────────────── runner


def run_cell(bucket_name: str, seed: int) -> Optional[Dict[str, Any]]:
    kind = KIND_OF[bucket_name]
    for sub in range(6):
        env = build_maze(kind, seed * 10 + sub)
        pair = pick_pair(env, BUCKETS[bucket_name], seed * 10 + sub)
        if pair is not None:
            break
    else:
        return None
    start, water, d_measured = pair

    def new_env() -> MultiAgentGridWorld:
        e = build_maze(kind, seed * 10 + sub)
        e.set_cell(*water, WATER)
        return e

    # witness experience + one broadcast wave (identical for all arms)
    memory = RoutePeerMemory(["witness", "traveler"], broadcast_every_k=4)
    witness_walk(new_env(), memory, start, water)
    memory.tick(4)

    route = follow_route_arm(new_env(), memory, start)
    place = place_arm(new_env(), water, start)
    blind = blind_arm(new_env(), start)

    def t_cens(r):
        return r["t_succ"] if r["t_succ"] is not None else STEP_LIMIT

    return {
        "bucket": bucket_name, "seed": seed, "D": round(d_measured, 2),
        "start": list(start), "water": list(water),
        "t_route": t_cens(route), "t_place": t_cens(place),
        "t_blind": t_cens(blind),
        "moves_route": route.get("n_moves"),
        "moves_place": place.get("n_moves"),
        "route_completed": route["completed"],
        "place_completed": place["completed"],
        "blind_completed": blind["completed"],
        "rw_strict": route.get("rw_star", {}).get("strict_RW", False),
        "phi_route": route.get("rw_star", {}).get("phi_route"),
        "gain_route_vs_place": t_cens(place) - t_cens(route),
        "gain_route_vs_blind": t_cens(blind) - t_cens(route),
        "gain_place_vs_blind": t_cens(blind) - t_cens(place),
    }


def mean_ci(vals: List[float], seed: int = 7,
            n_boot: int = 4000) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(vals, dtype=float)
    boots = [np.mean(rng.choice(arr, len(arr))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "r1_registered.json"), "w") as f:
        json.dump({
            "R1_neg": "|mean gain(route vs place)| <= 3 ticks at D=1",
            "R2": "gain(route vs place) > 0, bootstrap CI excl. 0 at "
                  "D2 and D4; mean(D4) > mean(D2)",
            "exploratory": "place can be worse than blind at high D",
        }, f, indent=2)

    rows: List[Dict[str, Any]] = []
    for bucket in BUCKETS:
        n_ok = 0
        for seed in range(a.seeds):
            r = run_cell(bucket, seed)
            if r is not None:
                rows.append(r)
                n_ok += 1
        print(f"{bucket}: {n_ok}/{a.seeds} cells "
              f"(mean D = {np.mean([r['D'] for r in rows if r['bucket'] == bucket]):.2f})")

    summary: Dict[str, Any] = {}
    print(f"\n{'bucket':<6} {'D':>5} {'gain r-vs-p [CI]':>26} "
          f"{'gain r-vs-b [CI]':>26} {'succ r/p/b':>12}")
    for bucket in BUCKETS:
        rs = [r for r in rows if r["bucket"] == bucket]
        if not rs:
            continue
        g_rp = mean_ci([r["gain_route_vs_place"] for r in rs])
        g_rb = mean_ci([r["gain_route_vs_blind"] for r in rs], seed=9)
        g_pb = mean_ci([r["gain_place_vs_blind"] for r in rs], seed=8)
        succ = (sum(r["route_completed"] for r in rs),
                sum(r["place_completed"] for r in rs),
                sum(r["blind_completed"] for r in rs))
        summary[bucket] = {
            "n": len(rs), "mean_D": float(np.mean([r["D"] for r in rs])),
            "gain_route_vs_place": g_rp, "gain_route_vs_blind": g_rb,
            "gain_place_vs_blind": g_pb,
            "succ_route": succ[0], "succ_place": succ[1],
            "succ_blind": succ[2],
            "place_success_rate": succ[1] / len(rs),
            "n_strict_RW": sum(r["rw_strict"] for r in rs),
        }
        print(f"{bucket:<6} {summary[bucket]['mean_D']:>5.2f} "
              f"{g_rp[0]:>+8.1f} [{g_rp[1]:>+6.1f},{g_rp[2]:>+6.1f}]   "
              f"{g_rb[0]:>+8.1f} [{g_rb[1]:>+6.1f},{g_rb[2]:>+6.1f}]   "
              f"{succ[0]}/{succ[1]}/{succ[2]} of {len(rs)}")

    # negative control in MOVES (post-hoc, clearly labelled): at D=1 the
    # tick gap is turn economy, not route knowledge — path lengths match
    d1 = [r for r in rows if r["bucket"] == "D1"
          if r["moves_route"] is not None and r["moves_place"] is not None]
    moves_gap = mean_ci([r["moves_place"] - r["moves_route"] for r in d1],
                        seed=11) if d1 else (float("nan"),) * 3
    summary["D1_moves_gap_posthoc"] = moves_gap

    r1_neg = abs(summary["D1"]["gain_route_vs_place"][0]) <= 3.0
    r2 = (summary["D2"]["gain_route_vs_place"][1] > 0
          and summary["D4"]["gain_route_vs_place"][1] > 0
          and summary["D4"]["gain_route_vs_place"][0]
          > summary["D2"]["gain_route_vs_place"][0])
    pathology = summary["D4"]["gain_place_vs_blind"][0] < 0

    verdict = {"R1_neg_control_at_D1": r1_neg,
               "R2_gain_grows_with_D": r2,
               "exploratory_place_worse_than_blind_at_D4": pathology,
               "posthoc_D1_moves_gap": [round(v, 2) for v in moves_gap],
               "place_success_by_bucket": {
                   b: summary[b]["place_success_rate"] for b in BUCKETS
                   if b in summary}}
    with open(os.path.join(OUT_DIR, "r1_results.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary, "verdict": verdict},
                  f, indent=1)

    print("=" * 68)
    for k, v in verdict.items():
        tag = "PASS" if v else ("FAIL" if not k.startswith("exploratory")
                                else "not observed")
        print(f"  [{tag}] {k}")
    print("=" * 68)
    print(f"Saved: {OUT_DIR}/r1_results.json")


if __name__ == "__main__":
    main()
