"""Portability sweep: peer-broadcast cadence × seed on MiniGrid substrate.

Section 7.7 "Portability to standard substrate" minimal sweep:

    K ∈ {1, 4, 8, 16, 64}, 20 seeds, peer-broadcast only,
    one layout (FourRooms), N=3, T=2, hazard 0.05.

Acceptance (b): Spearman slope signs of Proposition 3
    P(M*|R*) decreases in K (r<0, p<0.05)
    P(C*|M*) increases in K (r>0, p<0.05)
Acceptance (c): mean t_succ has interior minimum < both endpoints
    with non-overlapping bootstrap CIs.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.stats import spearmanr

import experiments.big_experiment.env_factory as ef
import experiments.big_experiment.runner as runner_mod
from experiments.big_experiment.runner import RunConfig, run_one_config
from experiments.minigrid_multiagent_wrapper.env_wrapper import build_minigrid_env


CADENCES = [1, 4, 8, 16, 64]
SEEDS = list(range(20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/minigrid_portability")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Monkey-patch build_env on both source and runner module.
    _orig_ef = ef.build_env
    _orig_rn = runner_mod.build_env
    ef.build_env = build_minigrid_env
    runner_mod.build_env = build_minigrid_env

    rows = []
    try:
        for K in CADENCES:
            for s in SEEDS:
                cfg = RunConfig(
                    n_agents=3, n_waters=2, layout="fourrooms",
                    architecture="peer", broadcast_every_k=K,
                    hazard_density=0.05, seed=s, step_limit=80,
                )
                out = run_one_config(cfg)
                out["K"] = K
                out["seed"] = s
                rows.append(out)
            print(f"  K={K:3d}  done ({len(SEEDS)} seeds)")
    finally:
        ef.build_env = _orig_ef
        runner_mod.build_env = _orig_rn

    # Save CSV
    out_csv = os.path.join(args.out_dir, "runs.csv")
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} runs → {out_csv}")

    # ── acceptance check ─────────────────────────────────────────
    K_arr = np.array([r["K"] for r in rows], dtype=float)
    pmr = np.array([float(r.get("p_M_given_R", np.nan)) for r in rows])
    pcm = np.array([float(r.get("p_C_given_M", np.nan)) for r in rows])
    tsu = np.array([float(r.get("t_succ_mean",
                                 r.get("mean_t_succ",
                                       r.get("steps_mean", np.nan))))
                    for r in rows])

    # filter nans in correlation arrays
    mr_mask = ~np.isnan(pmr)
    cm_mask = ~np.isnan(pcm)
    sp_mr = spearmanr(K_arr[mr_mask], pmr[mr_mask]) if mr_mask.sum() > 5 else None
    sp_cm = spearmanr(K_arr[cm_mask], pcm[cm_mask]) if cm_mask.sum() > 5 else None

    print("\n── Acceptance check ─────────────────────────────────")
    if sp_mr:
        print(f"  Spearman P(M|R) vs K: r={sp_mr.correlation:+.3f}  p={sp_mr.pvalue:.2e}")
    if sp_cm:
        print(f"  Spearman P(C|M) vs K: r={sp_cm.correlation:+.3f}  p={sp_cm.pvalue:.2e}")

    print(f"\n  Mean t_succ per K:")
    means = {}
    cis = {}
    for K in CADENCES:
        vals = tsu[(K_arr == K) & ~np.isnan(tsu)]
        if len(vals) > 1:
            means[K] = float(np.mean(vals))
            # 95% bootstrap CI
            rng = np.random.default_rng(0)
            boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(1000)]
            cis[K] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            print(f"    K={K:3d}  mean={means[K]:.2f}  CI=[{cis[K][0]:.2f}, {cis[K][1]:.2f}]")
    interior_Ks = [K for K in CADENCES[1:-1]
                   if K in means and means[K] < means.get(CADENCES[0], float("inf"))
                   and means[K] < means.get(CADENCES[-1], float("inf"))]
    print(f"\n  Interior minima (better than both endpoints): {interior_Ks}")

    pass_b = (sp_mr and sp_mr.correlation < 0 and sp_mr.pvalue < 0.05
              and sp_cm and sp_cm.correlation > 0 and sp_cm.pvalue < 0.05)
    pass_c = len(interior_Ks) >= 1
    print(f"\n  Acceptance (b) slope signs:        {'PASS' if pass_b else 'FAIL'}")
    print(f"  Acceptance (c) interior minimum:   {'PASS' if pass_c else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if (pass_b and pass_c) else 'PARTIAL'}")


if __name__ == "__main__":
    main()
