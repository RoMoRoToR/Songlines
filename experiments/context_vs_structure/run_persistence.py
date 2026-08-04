"""
Experiment 2: cross-session persistence (the 'stable over time' axis).

An agent lives in a persistent world across S sessions with a small per-session
exploration budget. Between sessions the world DRIFTS: with prob p_move the
water relocates. Three memory arms:

  cold        -- memory reset every session (no persistence);
  warm_naive  -- memory serialized at session end and reloaded (JSON round-trip,
                 the real stage-4 mechanics), no aging: stale records keep full
                 weight;
  warm_decay  -- same persistence + staleness decay: a record's weight is
                 exp(-alpha * sessions_old); candidates below tau are not used.

Predictions: warm beats cold on retrieval availability (structure accumulates
across sessions); under drift warm_naive is POISONED by stale records (commits
to vanished water) while warm_decay recovers -- persistence needs staleness to
be stable, which is the CSM thesis on the time axis.

Per (arm, session) we log: retrieval_nonempty, retrieval_correct (candidate set
contains CURRENT water), success, stale_share, memory size. Pure numpy + our
grid env + planner. Deterministic.

Run: PYTHONPATH=. python experiments/context_vs_structure/run_persistence.py \
        --sessions 8 --seeds 30 --out_dir tmp/cluster/persistence
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from experiments.big_experiment.planner import PlannerState, plan_action
from multiagent_env import HAZARD, WATER, MultiAgentGridWorld

W, H, BUDGET, EPS = 22, 18, 28, 0.6
ALPHA, TAU = 0.7, 0.35          # decay per session-of-age; usability threshold


def build_world(seed: int, water: Tuple[int, int]):
    rng = np.random.default_rng(seed)
    env = MultiAgentGridWorld(width=W, height=H, step_limit=BUDGET + 2,
                              observation_radius=2, rng_seed=seed)
    env.set_cell(*water, WATER)
    placed = 0
    while placed < 16:
        xy = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if xy != water and env.cell(*xy) == 0:
            env.set_cell(*xy, HAZARD)
            placed += 1
    start = (int(rng.integers(0, W)), int(rng.integers(0, H)))
    while start == water:
        start = (int(rng.integers(0, W)), int(rng.integers(0, H)))
    env.spawn("a0", start_xy=start, target_tag="water_source", direction=0)
    return env


def observe_into(memory: Dict, env, session: int, tick: int) -> None:
    obs = env._observation("a0")
    for c in obs.get("cells", []):
        memory[f"{int(c['xy'][0])},{int(c['xy'][1])}"] = {
            "tag": c["tag"], "session": session, "tick": tick}


def water_candidates(memory: Dict, session: int, decay: bool) -> List[Tuple[int, int]]:
    out = []
    for key, rec in memory.items():
        if rec["tag"] != "water_source":
            continue
        if decay:
            w = math.exp(-ALPHA * max(0, session - rec["session"]))
            if w < TAU:
                continue
        x, y = (int(v) for v in key.split(","))
        out.append((x, y))
    return out


def run_seed(arm: str, S: int, p_move: float, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    water = (int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1)))
    memory: Dict = {}
    mem_path = os.path.join(tempfile.gettempdir(), f"persist_{arm}_{seed}.json")
    rows = []
    for s in range(S):
        if s > 0 and rng.random() < p_move:      # drift between sessions
            water = (int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1)))
        if arm == "cold":
            memory = {}
        else:                                    # real serialize/reload round-trip
            if s > 0 and os.path.exists(mem_path):
                with open(mem_path) as f:
                    memory = json.load(f)
        env = build_world(seed * 100 + s, water)
        ps = PlannerState("a0")
        ag = env.agents["a0"]
        r_nonempty = r_correct = c_star = 0
        for tick in range(BUDGET):
            observe_into(memory, env, s, tick)
            cands = water_candidates(memory, s, decay=(arm == "warm_decay"))
            if cands:
                r_nonempty = 1
                if any(abs(c[0] - water[0]) <= EPS and abs(c[1] - water[1]) <= EPS
                       for c in cands):
                    r_correct = 1
            action = plan_action(ps, env, cands, tick, f"pers-{arm}")
            env.step({"a0": action})
            if ag.success:
                c_star = 1
                break
        stale = sum(1 for r in memory.values()
                    if r["tag"] == "water_source" and r["session"] < s)
        n_water_rec = sum(1 for r in memory.values() if r["tag"] == "water_source")
        if arm != "cold":
            with open(mem_path, "w") as f:
                json.dump(memory, f)
        rows.append({"arm": arm, "session": s, "seed": seed, "p_move": p_move,
                     "r_nonempty": r_nonempty, "r_correct": r_correct,
                     "c_star": c_star, "mem_size": len(memory),
                     "stale_water_recs": stale, "water_recs": n_water_rec})
    if os.path.exists(mem_path):
        os.remove(mem_path)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--p_move", type=float, default=0.5)
    ap.add_argument("--out_dir", default="tmp/cluster/persistence")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows: List[Dict] = []
    for arm in ("cold", "warm_naive", "warm_decay"):
        for seed in range(a.seeds):
            rows.extend(run_seed(arm, a.sessions, a.p_move, seed))
    with open(os.path.join(a.out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"Cross-session persistence  (S={a.sessions}, drift p={a.p_move}, "
          f"{a.seeds} seeds, budget {BUDGET} ticks, grid {W}x{H})\n")
    print(f"{'arm':>11} | " + " ".join(f"s={s}" for s in range(a.sessions))
          + "   <- retrieval_correct rate per session")
    for arm in ("cold", "warm_naive", "warm_decay"):
        vals = []
        for s in range(a.sessions):
            g = [r for r in rows if r["arm"] == arm and r["session"] == s]
            vals.append(sum(r["r_correct"] for r in g) / len(g))
        print(f"{arm:>11} | " + " ".join(f"{v:.2f}" for v in vals))
    print(f"\n{'arm':>11} | success (mean over sessions>=2) | stale water recs (last)")
    for arm in ("cold", "warm_naive", "warm_decay"):
        late = [r for r in rows if r["arm"] == arm and r["session"] >= 2]
        lastS = [r for r in rows if r["arm"] == arm and r["session"] == a.sessions - 1]
        succ = sum(r["c_star"] for r in late) / len(late)
        print(f"{arm:>11} | {succ:.3f}{'':26} | "
              f"{np.mean([r['stale_water_recs'] for r in lastS]):.2f}")
    print("\nReading: warm arms accumulate structure across sessions (r_correct and")
    print("success climb vs flat cold). Under drift, warm_naive carries stale water")
    print("records (poisoned commits); warm_decay prunes them -- persistence is only")
    print("STABLE with staleness, which is the CSM rule on the time axis.")


if __name__ == "__main__":
    main()
