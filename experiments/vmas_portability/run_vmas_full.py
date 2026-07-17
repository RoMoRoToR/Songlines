"""
Full continuous-substrate validation of Empirical Claim 1 on VMAS
(upgrades the 'preliminary' App F.7 check: multiple scarcity cells x many seeds,
per-run CSV for the same cluster-robust block bootstrap as the main sweep).

Reuses experiments.vmas_portability.run_vmas_portability (scenario, episode loop,
occupancy-sensitive materialisation) unchanged.

Run:
  PYTHONPATH=. python experiments/vmas_portability/run_vmas_full.py \
      --seeds 40 --out_dir tmp/cluster/vmas_full
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from experiments.vmas_portability.run_vmas_portability import sweep

KS = [1, 2, 4, 8, 16]
CELLS = [(6, 2), (8, 3), (10, 4)]          # (N agents, T waters) scarcity cells
MAX_STEPS = 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out_dir", default="tmp/cluster/vmas_full")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "runs.csv")

    all_rows = []
    for (N, T) in CELLS:
        rows = sweep(KS, N, T, list(range(a.seeds)), MAX_STEPS)
        for r in rows:
            r["n_agents"], r["n_waters"] = N, T
        all_rows.extend(rows)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"cell (N={N},T={T}) done: {len(rows)} agent-rows", flush=True)

    # per-cell conditional rates + cluster-robust Spearman over cells x seeds
    from scipy import stats
    print(f"\n{'cell':>10} {'K':>3} | {'P(M|R)':>7} {'P(C|M)':>7}")
    per_cellK = {}
    for (N, T) in CELLS:
        for K in KS:
            rs = [r for r in all_rows
                  if r["n_agents"] == N and r["n_waters"] == T and r["K"] == K]
            nR = sum(r["R"] for r in rs); nM = sum(r["M"] for r in rs)
            nC = sum(r["C"] for r in rs)
            pmr = nM / nR if nR else float("nan")
            pcm = nC / nM if nM else float("nan")
            per_cellK[(N, T, K)] = (pmr, pcm)
            print(f"  ({N:>2},{T:>2}) {K:>3} | {pmr:>7.3f} {pcm:>7.3f}")

    # block bootstrap over (cell, seed) blocks
    rng = np.random.default_rng(20260717)
    blocks = sorted({(r["n_agents"], r["n_waters"], r["seed"]) for r in all_rows})
    by_block = {}
    for r in all_rows:
        by_block.setdefault((r["n_agents"], r["n_waters"], r["seed"]), []).append(r)

    def spear(rows, num, den):
        vals = {}
        for K in KS:
            rs = [r for r in rows if r["K"] == K]
            d = sum(r[den] for r in rs); n = sum(r[num] for r in rs)
            vals[K] = n / d if d else np.nan
        ks = [k for k in KS if np.isfinite(vals[k])]
        if len(ks) < 3:
            return np.nan
        return stats.spearmanr(ks, [vals[k] for k in ks]).correlation

    B = 2000
    boots = {"MR": [], "CM": []}
    for _ in range(B):
        pick = rng.choice(len(blocks), size=len(blocks), replace=True)
        sample = [row for i in pick for row in by_block[blocks[i]]]
        s1 = spear(sample, "M", "R"); s2 = spear(sample, "C", "M")
        if np.isfinite(s1): boots["MR"].append(s1)
        if np.isfinite(s2): boots["CM"].append(s2)
    for name, pred in (("MR", "<0"), ("CM", ">0")):
        v = np.array(boots[name])
        print(f"Spearman P({'M|R' if name=='MR' else 'C|M'}) vs K: "
              f"{v.mean():+.3f}  95% CI [{np.percentile(v,2.5):+.3f},"
              f"{np.percentile(v,97.5):+.3f}]  (Claim 1 predicts {pred})")


if __name__ == "__main__":
    main()
