"""Hierarchical (mixed-effects) re-analysis of the bottleneck-shift slopes.

The paper reports raw Spearman correlations over the peer arm of the main
sweep -- P(M*|R*) vs K negative, P(C*|M*) vs K positive -- with
cluster-bootstrap CIs over the 81 design cells
(n_agents x n_waters x layout x hazard_density).  This script re-derives the
claim under models that make the dependence structure explicit:

  1. Raw Spearman rho (as in the paper), recomputed here.
  2. Cluster bootstrap of rho: resample the 81 design cells with
     replacement (all runs of a resampled cell come along), B=2000.
  3. Linear mixed model (statsmodels MixedLM, REML) on the rate-level
     outcome:
         p = b0 + b_K*log2(K) + b_N*n_agents + b_T*n_waters
                + b_H*hazard + u_layout + u_seed + u_world
     with crossed random intercepts fitted as variance components inside
     a single super-group (the standard statsmodels device for crossed
     effects).  u_world = the 81-level design cell.
     Model choice, stated honestly: statsmodels has no frequentist
     GLMM-logistic, and MixedLM takes no observation weights, so this is
     an UNWEIGHTED rate-level LMM (per-run trial counts vary 1..N<=8).
     The trial-level GEE below covers both the logit scale and the
     implicit weighting.
  4. GEE logistic at the event (Bernoulli trial) level, exchangeable
     working correlation, clustered by the 81 design cells: every R*
     event is a trial for the M-model (success = the corresponding M*
     lock), every M* event a trial for the C-model.  Event counts are
     reconstructed exactly from the per-agent rates (rate * n_agents is
     integral in the data).  In a small number of runs C* > M*
     (completion without a counted lock); successes are clamped to the
     number of trials and the clamp count is reported.
  5. Leave-one-layout-out and leave-one-seed-block-out (blocks of 10
     seeds) refits of the LMM + raw rho: sign/magnitude stability.

Deterministic (fixed bootstrap seed).  Usage:

    PYTHONPATH=. .venv/bin/python scripts/analyze_hierarchical_slopes.py \
        [--runs tmp/big_experiment_qrmc_40/runs.csv] [--boot 2000]

Writes tmp/hierarchical_stats/summary.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CELL = ["n_agents", "n_waters", "layout", "hazard_density"]
OUTCOMES = {
    "M_given_R": {"p": "p_M_given_R", "trials": "r_star_rate",
                  "succ": "m_star_rate", "expected_sign": -1},
    "C_given_M": {"p": "p_C_given_M", "trials": "m_star_rate",
                  "succ": "c_star_rate", "expected_sign": +1},
}
SEED_BLOCKS = {f"seeds_{a}-{a + 9}": range(a, a + 10) for a in (0, 10, 20, 30)}


def load_peer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    peer = df[df.architecture == "peer"].copy()
    peer["logK"] = np.log2(peer.broadcast_every_k.astype(float))
    peer["cell"] = peer[CELL].astype(str).agg("|".join, axis=1)
    return peer


# ───────────────────────────────────────── raw + cluster-bootstrap Spearman

def raw_rho(sub: pd.DataFrame, pcol: str) -> dict:
    r = spearmanr(sub.logK, sub[pcol], nan_policy="omit")
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(sub))}


def cluster_boot_rho(sub: pd.DataFrame, pcol: str, B: int,
                     rng: np.random.Generator) -> dict:
    groups = {c: g for c, g in sub.groupby("cell")}
    cells = list(groups)
    rhos = np.empty(B)
    for b in range(B):
        take = rng.choice(len(cells), size=len(cells), replace=True)
        boot = pd.concat([groups[cells[i]] for i in take], ignore_index=True)
        rhos[b] = spearmanr(boot.logK, boot[pcol],
                            nan_policy="omit").statistic
    return {"ci_lo": float(np.percentile(rhos, 2.5)),
            "ci_hi": float(np.percentile(rhos, 97.5)),
            "B": B, "n_cells": len(cells)}


# ───────────────────────────────────────────────────────── mixed-effects LMM

def fit_lmm(sub: pd.DataFrame, pcol: str) -> dict:
    import statsmodels.formula.api as smf
    d = sub.dropna(subset=[pcol]).copy()
    d["one"] = 1  # single super-group -> crossed variance components
    vc = {"layout": "0 + C(layout)", "seed": "0 + C(seed)",
          "world": "0 + C(cell)"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(
            f"{pcol} ~ logK + n_agents + n_waters + hazard_density",
            d, groups="one", vc_formula=vc, re_formula="0")
        res = model.fit(reml=True, method=["lbfgs", "powell"])
    warn = sorted({f"{w.category.__name__}: {w.message}" for w in caught
                   if "Convergence" in w.category.__name__
                   or "Singular" in w.category.__name__})
    b, se = res.params["logK"], res.bse["logK"]
    return {
        "beta_logK": float(b), "se": float(se),
        "ci_lo": float(b - 1.96 * se), "ci_hi": float(b + 1.96 * se),
        "p": float(res.pvalues["logK"]),
        "converged": bool(res.converged),
        "warnings": warn,
        "vc": {k: float(v) for k, v in res.vcomp_named.items()}
        if hasattr(res, "vcomp_named") else
        dict(zip(vc, map(float, res.vcomp))),
        "resid_var": float(res.scale),
        "n": int(res.nobs),
        "fixed": {k: float(v) for k, v in res.params.items()
                  if k in ("Intercept", "logK", "n_agents", "n_waters",
                           "hazard_density")},
    }


# ─────────────────────────────────────── GEE logistic at Bernoulli level

def expand_trials(sub: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """One row per event: trials = rate*N of the conditioning event,
    successes = rate*N of the consequent (clamped to trials)."""
    d = sub.copy()
    d["n_trials"] = (d[spec["trials"]] * d.n_agents).round().astype(int)
    d["n_succ"] = (d[spec["succ"]] * d.n_agents).round().astype(int)
    d = d[d.n_trials > 0]
    clamped = int((d.n_succ > d.n_trials).sum())
    d["n_succ"] = np.minimum(d.n_succ, d.n_trials)
    rows = d.loc[d.index.repeat(d.n_trials),
                 ["logK", "n_agents", "n_waters", "hazard_density", "cell"]]
    y = np.concatenate([
        np.r_[np.ones(s), np.zeros(t - s)]
        for s, t in zip(d.n_succ, d.n_trials)])
    rows = rows.reset_index(drop=True)
    rows["y"] = y
    rows.attrs["clamped_runs"] = clamped
    return rows


def fit_gee(sub: pd.DataFrame, spec: dict) -> dict:
    """Try exchangeable working correlation first; if its moment estimator
    degenerates (NaN -- happens for the M-model, where whole cells sit at
    P=1 and the within-cluster residual variance collapses), fall back to
    an independence working correlation.  Either way the SEs are the
    cluster-robust sandwich over the 81 design cells, so the fallback
    stays a valid GEE."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    tr = expand_trials(sub.dropna(subset=[spec["p"]]), spec)
    formula = "y ~ logK + n_agents + n_waters + hazard_density"
    start = smf.glm(formula, tr, family=sm.families.Binomial()).fit().params
    used, res = None, None
    for name, cs in [("exchangeable", sm.cov_struct.Exchangeable()),
                     ("independence", sm.cov_struct.Independence())]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = smf.gee(formula, groups="cell", data=tr,
                        family=sm.families.Binomial(), cov_struct=cs
                        ).fit(start_params=start.values)
        if np.isfinite(r.params["logK"]) and np.isfinite(r.bse["logK"]):
            used, res = name, r
            break
    if res is None:
        return {"failed": True, "n_trials": int(len(tr))}
    b, se = res.params["logK"], res.bse["logK"]
    dep = res.cov_struct.dep_params if used == "exchangeable" else 0.0
    return {
        "beta_logK_logit": float(b), "se": float(se),
        "ci_lo": float(b - 1.96 * se), "ci_hi": float(b + 1.96 * se),
        "p": float(res.pvalues["logK"]),
        "n_trials": int(len(tr)), "n_clusters": int(tr.cell.nunique()),
        "clamped_runs_C_gt_M": tr.attrs["clamped_runs"],
        "cov_struct": used, "working_corr": float(dep),
    }


# ─────────────────────────────────────────────────────────── leave-one-out

def loo(sub: pd.DataFrame, pcol: str) -> dict:
    out = {}
    for layout in sorted(sub.layout.unique()):
        d = sub[sub.layout != layout]
        out[f"drop_layout_{layout}"] = {
            "rho": raw_rho(d, pcol)["rho"],
            "lmm_beta_logK": fit_lmm(d, pcol)["beta_logK"],
        }
    for name, seeds in SEED_BLOCKS.items():
        d = sub[~sub.seed.isin(list(seeds))]
        out[f"drop_{name}"] = {
            "rho": raw_rho(d, pcol)["rho"],
            "lmm_beta_logK": fit_lmm(d, pcol)["beta_logK"],
        }
    return out


# ──────────────────────────────────────────────────────────────────── main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="tmp/big_experiment_qrmc_40/runs.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="tmp/hierarchical_stats")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    peer = load_peer(a.runs)
    os.makedirs(a.out, exist_ok=True)
    summary = {"runs_csv": a.runs, "n_peer_runs": int(len(peer)),
               "n_cells": int(peer.cell.nunique()),
               "n_seeds": int(peer.seed.nunique()),
               "K_values": sorted(int(k) for k in
                                  peer.broadcast_every_k.unique()),
               "outcomes": {}}

    for name, spec in OUTCOMES.items():
        pcol = spec["p"]
        print(f"\n=== {name} ({pcol}), expected sign "
              f"{'+' if spec['expected_sign'] > 0 else '-'} ===")
        raw = raw_rho(peer, pcol)
        print(f"raw Spearman rho = {raw['rho']:+.3f} (p={raw['p']:.2e})")
        boot = cluster_boot_rho(peer, pcol, a.boot, rng)
        print(f"cluster-bootstrap 95% CI over {boot['n_cells']} cells: "
              f"[{boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}]")
        lmm = fit_lmm(peer, pcol)
        print(f"MixedLM beta_logK = {lmm['beta_logK']:+.4f} "
              f"[{lmm['ci_lo']:+.4f}, {lmm['ci_hi']:+.4f}] "
              f"p={lmm['p']:.2e} converged={lmm['converged']} "
              f"warnings={lmm['warnings']}")
        print(f"  variance components: {lmm['vc']} resid={lmm['resid_var']:.4f}")
        gee = fit_gee(peer, spec)
        print(f"GEE-logit beta_logK = {gee['beta_logK_logit']:+.4f} "
              f"[{gee['ci_lo']:+.4f}, {gee['ci_hi']:+.4f}] "
              f"p={gee['p']:.2e} ({gee['n_trials']} trials, "
              f"{gee['n_clusters']} clusters, "
              f"C>M clamped in {gee['clamped_runs_C_gt_M']} runs)")
        loo_res = loo(peer, pcol)
        signs = {k: np.sign(v["lmm_beta_logK"]) for k, v in loo_res.items()}
        stable = all(s == spec["expected_sign"] for s in signs.values())
        print(f"leave-one-out: LMM sign stable = {stable}")
        for k, v in loo_res.items():
            print(f"  {k:<22} rho={v['rho']:+.3f}  "
                  f"lmm_beta={v['lmm_beta_logK']:+.4f}")
        summary["outcomes"][name] = {
            "expected_sign": spec["expected_sign"],
            "raw_spearman": raw, "cluster_bootstrap_rho": boot,
            "mixedlm": lmm, "gee_logistic": gee,
            "leave_one_out": loo_res,
            "loo_sign_stable": bool(stable),
        }

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_md(summary, os.path.join(a.out, "summary.md"))
    print(f"\nsaved {a.out}/summary.json and summary.md")


def write_md(s: dict, path: str) -> None:
    L = ["# Hierarchical re-analysis of the bottleneck-shift slopes", "",
         f"Data: `{s['runs_csv']}` -- {s['n_peer_runs']} peer runs, "
         f"{s['n_cells']} design cells, {s['n_seeds']} seeds, "
         f"K in {s['K_values']}.", "",
         "Model notes: MixedLM = rate-level linear mixed model (REML) with "
         "crossed random intercepts (layout, seed, world=design cell) as "
         "variance components; unweighted because statsmodels MixedLM takes "
         "no weights and has no frequentist GLMM-logistic. GEE = "
         "Bernoulli-trial-level logistic clustered by design cell with "
         "cluster-robust (sandwich) SEs; exchangeable working correlation "
         "where its moment estimator is finite, independence fallback "
         "otherwise (the used structure is named per table). Trial-level "
         "fitting implicitly weights runs by their event counts.", ""]
    for name, o in s["outcomes"].items():
        raw, boot, lmm, gee = (o["raw_spearman"], o["cluster_bootstrap_rho"],
                               o["mixedlm"], o["gee_logistic"])
        L += [f"## {name} vs K (expected sign "
              f"{'+' if o['expected_sign'] > 0 else '-'})", "",
              "| method | estimate | 95% CI | scale |",
              "|---|---|---|---|",
              f"| raw Spearman | {raw['rho']:+.3f} | -- (p={raw['p']:.1e}) "
              "| rho |",
              f"| cluster bootstrap (B={boot['B']}, {boot['n_cells']} cells) "
              f"| {raw['rho']:+.3f} | [{boot['ci_lo']:+.3f}, "
              f"{boot['ci_hi']:+.3f}] | rho |",
              f"| MixedLM (crossed RI) | {lmm['beta_logK']:+.4f} | "
              f"[{lmm['ci_lo']:+.4f}, {lmm['ci_hi']:+.4f}] | rate per "
              "doubling of K |",
              f"| GEE logistic ({gee['cov_struct']}) | "
              f"{gee['beta_logK_logit']:+.4f} | "
              f"[{gee['ci_lo']:+.4f}, {gee['ci_hi']:+.4f}] | logit per "
              "doubling of K |", "",
              f"MixedLM converged={lmm['converged']}, "
              f"warnings={lmm['warnings']}, variance components={lmm['vc']}, "
              f"residual var={lmm['resid_var']:.4f}. "
              f"GEE: {gee['n_trials']} trials, {gee['n_clusters']} clusters, "
              f"working corr={gee['working_corr']:.3f}, C>M clamped in "
              f"{gee['clamped_runs_C_gt_M']} runs.", "",
              "Leave-one-out (sign stability of the LMM slope: "
              f"**{'stable' if o['loo_sign_stable'] else 'NOT stable'}**):",
              "",
              "| held out | Spearman rho | LMM beta_logK |", "|---|---|---|"]
        for k, v in o["leave_one_out"].items():
            L.append(f"| {k} | {v['rho']:+.3f} | "
                     f"{v['lmm_beta_logK']:+.4f} |")
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
