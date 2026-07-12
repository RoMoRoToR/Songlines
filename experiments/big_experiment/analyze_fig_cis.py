"""Cluster-bootstrap CIs for the conditional-rates figure + a
censoring-aware time metric (reviewer requests: error bars on Fig 2;
survivorship-safe treatment of the K=8 interior optimum).

Reads the unified 35,640-run sweep and computes, per arm (independent,
shared bus, central aggregator, peer at each K):

  * mean P(M*|R*), P(C*|M*) with 95% cluster-bootstrap CIs, resampling
    the 81 independent design cells (n_agents, n_waters, layout,
    hazard_density) --- the same clustering as analyze_effect_sizes.py;
  * mean t_succ among successes (the figure's original metric), with
    the same CIs;
  * a censoring-aware expected time
        E[T] = success_rate * mean_t_succ + (1 - success_rate) * step_limit
    (failed runs are charged the full step limit instead of being
    dropped), with CIs --- the survivorship-robust version of the
    interior-optimum claim;
  * cell-paired differences of E[T] between peer K=8 and K in {2,4,16}
    (bootstrap CI of the difference over cells).

Writes tmp/paper1_clean_experiments_full/fig_aggregates.json, which
scripts/make_paper_figures_v3.py now reads.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/big_experiment/analyze_fig_cis.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

RUNS = "tmp/paper1_clean_experiments_full/multiagent_cadence_full/runs.csv"
OUT = "tmp/paper1_clean_experiments_full/fig_aggregates.json"
CELL = ["n_agents", "n_waters", "layout", "hazard_density"]
B = 4000
RNG = np.random.default_rng(0)


def arm_frames(df: pd.DataFrame):
    yield "indep", df[df.architecture == "independent"]
    yield "shared", df[df.architecture == "shared"]
    yield "central", df[df.architecture == "centralized"]
    for k in [1, 2, 4, 8, 16, 32, 48, 64]:
        yield f"peer_K{k}", df[(df.architecture == "peer")
                               & (df.broadcast_every_k == k)]


def cell_means(sub: pd.DataFrame, col: str) -> np.ndarray:
    return sub.groupby(CELL, observed=True)[col].mean().to_numpy()


def boot_ci(cells: np.ndarray):
    cells = cells[~np.isnan(cells)]
    idx = RNG.integers(0, len(cells), size=(B, len(cells)))
    means = cells[idx].mean(axis=1)
    return (float(np.nanmean(cells)),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main() -> None:
    df = pd.read_csv(RUNS)
    step_limit = float(df.step_limit.iloc[0])
    df = df.copy()
    df["t_censored"] = (df.success_rate * df.mean_t_succ.fillna(step_limit)
                        + (1.0 - df.success_rate) * step_limit)

    out = {"arms": {}, "step_limit": step_limit,
           "clustering": f"{len(df.groupby(CELL, observed=True))} cells "
                         f"({' x '.join(CELL)}), B={B}"}
    print(f"{'arm':<10} {'P(M|R)':>20} {'P(C|M)':>20} "
          f"{'t_succ':>18} {'E[T] censored':>20}")
    for arm, sub in arm_frames(df):
        row = {}
        for key, col in [("p_MR", "p_M_given_R"), ("p_CM", "p_C_given_M"),
                         ("t_succ", "mean_t_succ"),
                         ("t_censored", "t_censored")]:
            row[key] = boot_ci(cell_means(sub, col))
        out["arms"][arm] = row
        f = lambda k: f"{row[k][0]:.3f} [{row[k][1]:.3f},{row[k][2]:.3f}]"
        print(f"{arm:<10} {f('p_MR'):>20} {f('p_CM'):>20} "
              f"{row['t_succ'][0]:>7.2f} "
              f"[{row['t_succ'][1]:.2f},{row['t_succ'][2]:.2f}] "
              f"{row['t_censored'][0]:>8.2f} "
              f"[{row['t_censored'][1]:.2f},{row['t_censored'][2]:.2f}]")

    # cell-paired differences vs peer K=8, on both time metrics
    peer = df[df.architecture == "peer"]
    for metric in ["t_censored", "mean_t_succ"]:
        per_cell = peer.groupby(CELL + ["broadcast_every_k"],
                                observed=True)[metric].mean().unstack()
        indep_cell = df[df.architecture == "independent"].groupby(
            CELL, observed=True)[metric].mean()
        per_cell = per_cell.join(indep_cell.rename("indep"))
        out[f"paired_diffs_{metric}"] = {}
        print(f"\ncell-paired {metric} differences vs peer K=8 "
              "(negative = K=8 faster):")
        for other in [2, 4, 16, "indep"]:
            d = (per_cell[8] - per_cell[other]).dropna().to_numpy()
            m, lo, hi = boot_ci(d)
            out[f"paired_diffs_{metric}"][str(other)] = [m, lo, hi]
            print(f"  K=8 - {str(other):>6}: {m:+.2f} "
                  f"[{lo:+.2f},{hi:+.2f}] "
                  f"({'excludes 0' if hi < 0 or lo > 0 else 'includes 0'})")

    with open(OUT, "w") as fjson:
        json.dump(out, fjson, indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
