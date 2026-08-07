"""K x trust-threshold mission-success sweep (Figure 2b, real data).

Sweeps the CSM architecture over broadcast cadence K x merge/trust
inclusion threshold tau on the standard scarcity scenario, recording
mean mission success per (K, tau) cell.  This turns the previously
illustrative (K x trust) heatmap into a measured surface.

Run (ON SPHINX, CPU)::

    PYTHONPATH=. python experiments/collective_semantic_memory/exp_ktrust_sweep.py \
        --seeds 8 --out tmp/cluster/song_grammar/ktrust
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
from experiments.big_experiment.runner import RunConfig, run_one_config

KS = [1, 2, 4, 8, 16, 32]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LAYOUTS = ["symmetric", "asymmetric", "random"]
HAZARDS = [0.0, 0.05]
SCARCITY = [(3, 2), (5, 3), (8, 5)]


def cell(K, tau, seeds):
    """returns (mean success rate, mean time-to-success over successes)."""
    succ, tsucc = [], []
    for s in range(seeds):
        for layout in LAYOUTS:
            for hz in HAZARDS:
                for N, T in SCARCITY:
                    cfg = RunConfig(n_agents=N, n_waters=T, layout=layout,
                                    architecture="csm", broadcast_every_k=K,
                                    hazard_density=hz, seed=s, step_limit=120,
                                    merge_threshold=tau)
                    out = run_one_config(cfg)
                    sr = out.get("success_rate", float("nan"))
                    ts = out.get("mean_t_succ", float("nan"))
                    if sr == sr:
                        succ.append(float(sr))
                    if ts == ts:
                        tsucc.append(float(ts))
    return (float(np.mean(succ)) if succ else float("nan"),
            float(np.mean(tsucc)) if tsucc else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/ktrust")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    succ_grid, t_grid = {}, {}
    for tau in TAUS:
        sr, tr = {}, {}
        for K in KS:
            s, t = cell(K, tau, a.seeds)
            sr[str(K)] = s; tr[str(K)] = t
        succ_grid[f"tau{tau}"] = sr; t_grid[f"tau{tau}"] = tr
        print(f"tau {tau:.2f}: succ " +
              " ".join(f"K{K}={sr[str(K)]:.3f}" for K in KS) + " | t " +
              " ".join(f"{tr[str(K)]:.2f}" for K in KS), flush=True)
    # interior optimum is on efficiency (t_succ, lower is better)
    cells = [(t_grid[f"tau{t}"][str(k)], k, t) for t in TAUS for k in KS
             if t_grid[f"tau{t}"][str(k)] == t_grid[f"tau{t}"][str(k)]]
    best_t = min(cells, key=lambda z: z[0])
    out = {"Ks": KS, "taus": TAUS, "success_grid": succ_grid,
           "t_succ_grid": t_grid, "seeds": a.seeds,
           "argmin_t_succ": {"t_succ": best_t[0], "K": best_t[1], "tau": best_t[2]}}
    with open(os.path.join(a.out, "ktrust_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"efficiency optimum: t_succ={best_t[0]:.2f} at K={best_t[1]}, tau={best_t[2]}")
    print(f"Saved: {a.out}/ktrust_results.json")


if __name__ == "__main__":
    main()
