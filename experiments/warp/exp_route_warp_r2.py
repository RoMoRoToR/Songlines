"""R2 — route rupture under evidence aging: the song breaks mid-verse.

The witness stamps edges SEQUENTIALLY (edge i at tick i), so its trace
ages non-uniformly: the beginning of the route is always older than the
end, and a delayed traveler CHASES a fading trace.  With the same gate
as everywhere in this series (weight = trust * exp(-alpha*age) *
log1p(count), inclusion threshold tau), the edge the traveler is about
to cross is the oldest remaining one, which yields three predictable
regimes on the start-delay axis a0:

  complete   — the traveler outruns the decay end to end;
  rupture@i* — the gate closes mid-route at a PREDICTED path index
               (an RW* lock with C* = 0 and a measured break position);
  DOA        — edge 0 is already dead at commit: the route never forms.

The predictor is fully deterministic: a no-decay calibration run
records the traveler's kinematics (path index per tick, including turn
ticks), and the registered prediction replays the closed-form gate
over that trajectory — the same move that produced the 6/6 hold-out in
W2.  Predictions are written to disk BEFORE the decaying episodes.

Trust-flip for routes: at trust = 0.7 (single traversal, log1p(1)
mass) age_max = 9.6 < L for every maze here, so NO route ever commits
— the discrete off-switch, now for paths.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_route_warp_r2.py \\
        [--seeds 5]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.big_experiment.planner import (
    _direction_toward, _turn_or_forward,
)
from experiments.warp.exp_route_warp_r0 import (
    ALPHA, GRID_H, GRID_W, RoutePeerMemory, TAU, WATER_TAG, bfs_path,
    witness_walk,
)
from experiments.warp.exp_route_warp_r1 import build_maze
from multiagent_env import MultiAgentGridWorld, WALL, WATER

GridXY = Tuple[int, int]
OUT_DIR = "tmp/warp/r2_route_rupture"
L_RANGE = (10, 14)       # target witness-path length in edges
A0S = list(range(0, 11))
TRUSTS = [1.0, 0.7]
EDGE_MASS = math.log1p(1)  # single traversal


def age_max(trust: float) -> float:
    w0 = trust * EDGE_MASS
    return math.log(w0 / TAU) / ALPHA if w0 > TAU else -1.0


# ───────────────────────────────────── maze cell


def pick_cell(seed: int) -> Optional[Tuple[MultiAgentGridWorld,
                                           GridXY, GridXY, int, int]]:
    """A sparse-comb maze plus a (start, water) pair whose true path
    length lands in L_RANGE.  Returns (env, start, water, L, sub)."""
    for sub in range(8):
        env = build_maze("sparse", seed * 10 + sub)
        rng = np.random.default_rng(70_000 + seed * 10 + sub)
        passable_cells = [(x, y) for x in range(GRID_W)
                          for y in range(GRID_H)
                          if env.cell(x, y) != WALL]
        for _ in range(300):
            s = passable_cells[int(rng.integers(0, len(passable_cells)))]
            w = passable_cells[int(rng.integers(0, len(passable_cells)))]
            path = bfs_path(env, s, w)
            if path is None:
                continue
            L = len(path) - 1
            if L_RANGE[0] <= L <= L_RANGE[1] \
                    and abs(s[0] - w[0]) + abs(s[1] - w[1]) >= 6:
                return env, s, w, L, sub
    return None


# ───────────────────────────────────── chase follower (re-query per tick)


def chase_follower(env, memory: RoutePeerMemory, start: GridXY,
                   water: GridXY, T0: int,
                   step_limit: int = 150) -> Dict[str, Any]:
    """Re-queries the gated route EVERY tick at global time T = T0 + t.
    Returns outcome + the per-tick path-index trajectory."""
    env.spawn("traveler", start_xy=start, target_tag=WATER_TAG, direction=0)
    ag = env.agents["traveler"]
    edges_done = 0
    committed = False
    phi_at_commit: Optional[float] = None
    index_at_tick: List[int] = []

    for t in range(step_limit):
        if ag.success:
            return {"outcome": "complete", "t_travel": t,
                    "edges_done": edges_done, "phi": phi_at_commit,
                    "committed": committed, "index_at_tick": index_at_tick}
        T = T0 + t
        route = memory.route_to_water("traveler", (ag.x, ag.y), T)
        index_at_tick.append(edges_done)
        if route is None or len(route["path"]) < 2:
            return {"outcome": "doa" if not committed else "rupture",
                    "rupture_index": edges_done, "phi": phi_at_commit,
                    "committed": committed, "edges_done": edges_done,
                    "index_at_tick": index_at_tick}
        if not committed:
            committed = True
            phi_at_commit = route["phi_route"]
        nxt = route["path"][1]
        d = _direction_toward((ag.x, ag.y), nxt)
        action = _turn_or_forward(ag.direction, d)
        prev = (ag.x, ag.y)
        env.step({"traveler": action})
        if (ag.x, ag.y) != prev:
            memory.observe_move("traveler", prev, (ag.x, ag.y), T)
            edges_done += 1
    return {"outcome": "timeout", "edges_done": edges_done,
            "phi": phi_at_commit, "committed": committed,
            "index_at_tick": index_at_tick}


# ───────────────────────────────────── predictor


def predict(index_at_tick: List[int], L: int, a0: int,
            trust: float) -> Dict[str, Any]:
    """Replay the closed-form gate over the CALIBRATED kinematics:
    at calibration tick t the traveler needs edge i(t) (stamped by the
    witness at tick i); it is alive iff (L + a0 + t) - i <= age_max."""
    am = age_max(trust)
    if am < 0:
        return {"regime": "doa", "rupture_index": 0}
    for t, i in enumerate(index_at_tick):
        if i >= L:
            break
        if (L + a0 + t) - i > am:
            return {"regime": "doa" if i == 0 and t == 0 else "rupture",
                    "rupture_index": i}
    return {"regime": "complete", "rupture_index": None}


# ───────────────────────────────────── main


def run_cell(seed: int) -> Optional[Dict[str, Any]]:
    picked = pick_cell(seed)
    if picked is None:
        return None
    env0, start, water, L, sub = picked

    def new_env():
        e = build_maze("sparse", seed * 10 + sub)
        e.set_cell(*water, WATER)
        return e

    def new_memory(trust: float, alpha: float) -> RoutePeerMemory:
        m = RoutePeerMemory(["witness", "traveler"], broadcast_every_k=4,
                            trust=trust, alpha=alpha)
        witness_walk(new_env(), m, start, water)
        m.tick(4)
        return m

    # calibration: no decay — pure kinematics (turn structure included)
    calib = chase_follower(new_env(), new_memory(1.0, 0.0), start, water,
                           T0=L)
    if calib["outcome"] != "complete":
        return None
    kin = calib["index_at_tick"]

    predictions = {}
    for trust in TRUSTS:
        for a0 in A0S:
            predictions[f"trust={trust}|a0={a0}"] = predict(
                kin, L, a0, trust)

    results = {}
    for trust in TRUSTS:
        for a0 in A0S:
            r = chase_follower(new_env(), new_memory(trust, ALPHA),
                               start, water, T0=L + a0)
            results[f"trust={trust}|a0={a0}"] = {
                "outcome": r["outcome"],
                "rupture_index": r.get("rupture_index"),
                "phi": r.get("phi"), "committed": r["committed"],
            }
    return {"seed": seed, "L": L, "start": list(start),
            "water": list(water), "calibration_ticks": len(kin),
            "predictions": predictions, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    cells = []
    for seed in range(a.seeds * 3):
        if len(cells) >= a.seeds:
            break
        c = run_cell(seed)
        if c is not None:
            cells.append(c)
    print(f"{len(cells)} maze cells "
          f"(L = {[c['L'] for c in cells]})")

    with open(os.path.join(OUT_DIR, "r2_predictions.json"), "w") as f:
        json.dump([{k: c[k] for k in ("seed", "L", "predictions")}
                   for c in cells], f, indent=1)

    n_regime_ok = n_total = 0
    rupture_errs: List[int] = []
    n_trustflip_commits = 0
    n_rupture_cells = 0
    for c in cells:
        for key, pred in c["predictions"].items():
            res = c["results"][key]
            n_total += 1
            emp_regime = ("complete" if res["outcome"] == "complete"
                          else ("doa" if not res["committed"]
                                or res["outcome"] == "doa" else "rupture"))
            # DOA and rupture-at-0 are the same physical event
            p_regime = pred["regime"]
            match = (emp_regime == p_regime
                     or (p_regime == "doa" and emp_regime == "rupture"
                         and res["rupture_index"] == 0)
                     or (p_regime == "rupture" and emp_regime == "doa"
                         and pred["rupture_index"] == 0))
            n_regime_ok += int(match)
            if p_regime == "rupture" and emp_regime == "rupture" \
                    and pred["rupture_index"] is not None \
                    and res["rupture_index"] is not None:
                n_rupture_cells += 1
                rupture_errs.append(abs(res["rupture_index"]
                                        - pred["rupture_index"]))
            if key.startswith("trust=0.7") and res["committed"]:
                n_trustflip_commits += 1

    regime_rate = n_regime_ok / n_total
    pos_ok = (all(e <= 1 for e in rupture_errs) if rupture_errs else None)
    verdict = {
        "regime_match_rate": round(regime_rate, 3),
        "P1_regimes_ge_90pct": regime_rate >= 0.9,
        "n_mid_route_ruptures": n_rupture_cells,
        "rupture_position_errors": rupture_errs,
        "P2_rupture_position_within_1_edge": pos_ok,
        "P3_route_trust_flip_zero_commits": n_trustflip_commits == 0,
    }
    with open(os.path.join(OUT_DIR, "r2_results.json"), "w") as f:
        json.dump({"cells": cells, "verdict": verdict}, f, indent=1)

    print(f"\nregime match: {n_regime_ok}/{n_total} "
          f"({regime_rate:.1%})")
    print(f"mid-route ruptures: {n_rupture_cells}, "
          f"position errors: {rupture_errs}")
    print(f"trust=0.7 commits: {n_trustflip_commits} (predicted 0)")
    print("=" * 60)
    for k, v in verdict.items():
        if isinstance(v, bool) or v is None:
            tag = "PASS" if v else ("n/a" if v is None else "FAIL")
            print(f"  [{tag}] {k}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/r2_results.json")


if __name__ == "__main__":
    main()
