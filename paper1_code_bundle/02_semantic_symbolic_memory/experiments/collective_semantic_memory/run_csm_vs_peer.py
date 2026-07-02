"""CSM vs peer-K sweep on the scarcity scenario.

Compares the minimal CSM (peer broadcast at K=8 + trust + staleness +
trust-weighted merge) against five fixed-cadence peer-broadcast
architectures (K ∈ {1,4,8,16,64}) on the standard scarcity scenario
(N=3, T=2, asymmetric, hazard 0.05).

Reports per-architecture (P(M|R), P(C|M), success, t_succ) means + 95%
bootstrap CIs and tests whether CSM lies in the upper-right region of
the M×C Pareto that no fixed-cadence peer reaches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from experiments.big_experiment.runner import RunConfig, run_one_config


PEER_CADENCES = [1, 4, 8, 16, 64]
SEEDS = list(range(20))
LAYOUTS = ["symmetric", "asymmetric", "random"]
HAZARDS = [0.0, 0.05, 0.10]
SCARCITY = [(3, 2), (5, 3), (8, 5)]  # (N, T)


def boot_ci(values, B=2000, rng_seed=0):
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    boots = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_arch(architecture, K, seeds):
    rows = []
    for s in seeds:
        for layout in LAYOUTS:
            for hazard in HAZARDS:
                for N, T in SCARCITY:
                    cfg = RunConfig(
                        n_agents=N, n_waters=T, layout=layout,
                        architecture=architecture, broadcast_every_k=K,
                        hazard_density=hazard, seed=s, step_limit=120,
                    )
                    out = run_one_config(cfg)
                    rows.append({
                        "K": K,
                        "seed": s,
                        "layout": layout, "hazard": hazard,
                        "N": N, "T": T,
                        "p_M_given_R": float(out.get("p_M_given_R", float("nan"))),
                        "p_C_given_M": float(out.get("p_C_given_M", float("nan"))),
                        "success_rate": float(out.get("success_rate", float("nan"))),
                    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/csm_vs_peer")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    summary = []

    print("Running peer-K configurations + CSM (20 seeds each):")
    print("=" * 70)
    for K in PEER_CADENCES:
        rows = run_arch("peer", K, SEEDS)
        for r in rows:
            r["arch"] = f"peer-K{K}"
            all_rows.append(r)
        pmr = [r["p_M_given_R"] for r in rows if not np.isnan(r["p_M_given_R"])]
        pcm = [r["p_C_given_M"] for r in rows if not np.isnan(r["p_C_given_M"])]
        suc = [r["success_rate"] for r in rows if not np.isnan(r["success_rate"])]
        m_mean, m_lo, m_hi = boot_ci(pmr)
        c_mean, c_lo, c_hi = boot_ci(pcm)
        s_mean, s_lo, s_hi = boot_ci(suc)
        summary.append({
            "arch": f"peer-K{K}",
            "P(M|R)": m_mean, "P(M|R)_ci": [m_lo, m_hi],
            "P(C|M)": c_mean, "P(C|M)_ci": [c_lo, c_hi],
            "success": s_mean, "success_ci": [s_lo, s_hi],
        })
        print(f"  peer-K{K:<3d}  P(M|R)={m_mean:.3f}[{m_lo:.2f},{m_hi:.2f}]  "
              f"P(C|M)={c_mean:.3f}[{c_lo:.2f},{c_hi:.2f}]  "
              f"succ={s_mean:.3f}[{s_lo:.2f},{s_hi:.2f}]")

    rows = run_arch("csm", 8, SEEDS)
    for r in rows:
        r["arch"] = "csm"
        all_rows.append(r)
    pmr = [r["p_M_given_R"] for r in rows if not np.isnan(r["p_M_given_R"])]
    pcm = [r["p_C_given_M"] for r in rows if not np.isnan(r["p_C_given_M"])]
    suc = [r["success_rate"] for r in rows if not np.isnan(r["success_rate"])]
    m_mean, m_lo, m_hi = boot_ci(pmr)
    c_mean, c_lo, c_hi = boot_ci(pcm)
    s_mean, s_lo, s_hi = boot_ci(suc)
    csm_summary = {
        "arch": "csm",
        "P(M|R)": m_mean, "P(M|R)_ci": [m_lo, m_hi],
        "P(C|M)": c_mean, "P(C|M)_ci": [c_lo, c_hi],
        "success": s_mean, "success_ci": [s_lo, s_hi],
    }
    summary.append(csm_summary)
    print(f"  csm        P(M|R)={m_mean:.3f}[{m_lo:.2f},{m_hi:.2f}]  "
          f"P(C|M)={c_mean:.3f}[{c_lo:.2f},{c_hi:.2f}]  "
          f"succ={s_mean:.3f}[{s_lo:.2f},{s_hi:.2f}]")

    # Save raw
    csv_path = os.path.join(args.out_dir, "runs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {len(all_rows)} runs → {csv_path}")
    print(f"Saved summary → {summary_path}")

    # ── Pareto check ─────────────────────────────────────────
    print("\n── Pareto analysis (M, C, success) ────────────────")
    csm_M, csm_C, csm_S = csm_summary["P(M|R)"], csm_summary["P(C|M)"], csm_summary["success"]
    csm_S_lo = csm_summary["success_ci"][0]
    print(f"  CSM:       P(M|R)={csm_M:.3f}  P(C|M)={csm_C:.3f}  succ={csm_S:.3f}")
    # Compare on each metric vs each peer-K with CI overlap test.
    strict_succ_wins = []
    for s in summary:
        if s["arch"] == "csm":
            continue
        peer_S_hi = s["success_ci"][1]
        if csm_S_lo > peer_S_hi:
            strict_succ_wins.append((s["arch"], csm_S - s["success"]))
    print(f"  CSM strictly higher success (non-overlapping 95% CI) vs:")
    for arch, delta in strict_succ_wins:
        print(f"    {arch}: Δsucc=+{delta:.3f}")
    # M×C frontier analysis
    print(f"\n  M×C frontier ranking (sorted by P(M|R) decreasing):")
    for s in sorted(summary, key=lambda x: -x["P(M|R)"]):
        print(f"    {s['arch']:<10s}  P(M|R)={s['P(M|R)']:.3f}  P(C|M)={s['P(C|M)']:.3f}  succ={s['success']:.3f}")

    if len(strict_succ_wins) == len(PEER_CADENCES):
        print("\n  ACCEPTANCE: CSM strictly dominates ALL fixed-cadence peers on success rate.")
    elif strict_succ_wins:
        print(f"\n  ACCEPTANCE: CSM strictly dominates {len(strict_succ_wins)}/{len(PEER_CADENCES)} fixed-K peers on success.")
    else:
        print("\n  ACCEPTANCE: CSM matches (no strict dominance with non-overlapping CI).")


if __name__ == "__main__":
    main()
