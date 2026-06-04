"""Q/R/M/C stage decomposition analysis.

Companion to ``analyze.py`` — focuses on the unified-framework story:
where is the bottleneck under each architecture/cadence?

Key plots
---------
1. Stage profile per architecture (radar/bar).  4-dim "fingerprint".
2. Stage rates vs cadence K per (N, M, layout).  Tests bottleneck-shift
   hypothesis: cadence should shift mass between R-bottleneck and
   M-bottleneck.
3. Conditional rates P(R|Q), P(M|R), P(C|M) per architecture.
4. Causal explanation of mean_t_succ via stage failure attribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGES = ["q_star_rate", "r_star_rate", "m_star_rate", "c_star_rate"]
STAGE_LABELS = ["Q* (query)", "R* (retrieval)", "M* (material)", "C* (completion)"]
TICK_RATES = ["q_tick_rate", "r_tick_rate", "m_tick_rate"]
TICK_LABELS = ["Q (per-tick)", "R (per-tick)", "M (per-tick)"]


def _arch_label(row):
    if row["architecture"] == "peer":
        return f"peer(K={int(row['broadcast_every_k'])})"
    return row["architecture"]


def load_runs(csv_path):
    df = pd.read_csv(csv_path)
    df["arch_label"] = df.apply(_arch_label, axis=1)
    return df


# ─────────────────────────────────────────── stage profile per architecture


def plot_stage_profile(df: pd.DataFrame, out_path: str) -> None:
    """Per-architecture bar chart of Q*/R*/M*/C* averaged over all configs."""
    arch_order = ["independent", "shared", "centralized",
                  "peer(K=1)", "peer(K=2)", "peer(K=4)", "peer(K=8)", "peer(K=16)"]
    means = []
    for arch in arch_order:
        sub = df[df["arch_label"] == arch]
        if sub.empty:
            means.append([np.nan] * 4)
            continue
        means.append([sub[s].mean() for s in STAGES])
    means = np.array(means)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(arch_order))
    width = 0.20
    for i, label in enumerate(STAGE_LABELS):
        ax.bar(x + (i - 1.5) * width, means[:, i], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("rate (averaged over all configs)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Q*/R*/M*/C* episode-level rates per architecture\n"
                 "(reveals where each architecture's bottleneck sits)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_tick_stage_profile(df: pd.DataFrame, out_path: str) -> None:
    """Same as above but using per-tick rates (continuous diagnostic, sharper signal)."""
    arch_order = ["independent", "shared", "centralized",
                  "peer(K=1)", "peer(K=2)", "peer(K=4)", "peer(K=8)", "peer(K=16)"]
    means = []
    for arch in arch_order:
        sub = df[df["arch_label"] == arch]
        if sub.empty:
            means.append([np.nan] * 3)
            continue
        means.append([sub[s].mean() for s in TICK_RATES])
    means = np.array(means)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(arch_order))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#27ae60"]
    for i, label in enumerate(TICK_LABELS):
        ax.bar(x + (i - 1) * width, means[:, i], width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(arch_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("per-tick rate")
    ax.set_title("Per-tick stage rates per architecture\n"
                 "Sharper signal than episode-level — shows time-density of each event")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────── conditional rates


def plot_conditional_rates(df: pd.DataFrame, out_path: str) -> None:
    """P(R|Q), P(M|R), P(C|M) per architecture."""
    arch_order = ["independent", "shared", "centralized",
                  "peer(K=1)", "peer(K=2)", "peer(K=4)", "peer(K=8)", "peer(K=16)"]
    cond_cols = ["p_R_given_Q", "p_M_given_R", "p_C_given_M"]
    cond_labels = ["P(R|Q)", "P(M|R)", "P(C|M)"]
    means = []
    for arch in arch_order:
        sub = df[df["arch_label"] == arch]
        if sub.empty:
            means.append([np.nan] * 3)
            continue
        means.append([sub[c].mean() for c in cond_cols])
    means = np.array(means)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(arch_order))
    width = 0.25
    colors = ["#9b59b6", "#f39c12", "#16a085"]
    for i, label in enumerate(cond_labels):
        ax.bar(x + (i - 1) * width, means[:, i], width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(arch_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("conditional rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Conditional stage rates per architecture\n"
                 "Identifies the WEAKEST link in the Q→R→M→C chain")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────── bottleneck shift R ↔ M


def plot_bottleneck_shift(df: pd.DataFrame, out_path: str,
                           layout: str = "asymmetric") -> None:
    """For peer architecture, plot R-rate AND M-rate vs K per (N, M).

    Hypothesis: as K decreases, R-rate stays high or grows (fresh memory),
    but M-rate DROPS (peers occupy targets faster).  K* is where their
    product (≈ effective navigation rate) is maximised.
    """
    sub = df[(df["layout"] == layout) & (df["architecture"] == "peer")]
    if sub.empty:
        return
    nm_pairs = sorted(set(zip(sub["n_agents"], sub["n_waters"])))
    nm_pairs = [p for p in nm_pairs if p[0] > p[1]]  # scarcity only
    if not nm_pairs:
        return
    n_plots = len(nm_pairs)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows),
                            squeeze=False)
    for idx, (n, m) in enumerate(nm_pairs):
        ax = axs[idx // ncols][idx % ncols]
        cell = sub[(sub["n_agents"] == n) & (sub["n_waters"] == m)]
        per_k = cell.groupby("broadcast_every_k").agg({
            "r_tick_rate": "mean", "m_tick_rate": "mean",
            "mean_t_succ": "mean",
        }).reset_index().sort_values("broadcast_every_k")
        if per_k.empty:
            continue
        ax.plot(per_k["broadcast_every_k"], per_k["r_tick_rate"],
                marker="o", color="#e74c3c", label="R per-tick rate",
                linewidth=1.8)
        ax.plot(per_k["broadcast_every_k"], per_k["m_tick_rate"],
                marker="s", color="#27ae60", label="M per-tick rate",
                linewidth=1.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("peer cadence K")
        ax.set_ylabel("per-tick rate")
        ax.set_title(f"N={n}, M={m}  (ρ={n/m:.2f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        # twin axis for mean_t_succ
        ax2 = ax.twinx()
        ax2.plot(per_k["broadcast_every_k"], per_k["mean_t_succ"],
                 marker="x", color="#7f8c8d", linestyle="--",
                 alpha=0.5, label="mean_t_succ")
        ax2.set_ylabel("mean t_succ", color="#7f8c8d", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#7f8c8d", labelsize=8)
    # hide unused
    for idx in range(n_plots, nrows * ncols):
        axs[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(
        f"R/M bottleneck shift in peer architecture — layout={layout}\n"
        "Hypothesis: K decreases → R-rate grows (info fresh), M-rate drops (targets occupy faster).",
        fontsize=11, y=0.995,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────── stage attribution of mean_t_succ


def plot_stage_attribution(df: pd.DataFrame, out_path: str) -> None:
    """For each architecture, fit mean_t_succ ~ Q + R + M + C and show coefs.

    Interpretation: large abs coef on a stage means t_succ is driven by
    that stage's failure rate in that architecture.
    """
    arch_order = ["independent", "shared", "centralized",
                  "peer(K=1)", "peer(K=2)", "peer(K=4)", "peer(K=8)", "peer(K=16)"]
    rows = []
    for arch in arch_order:
        sub = df[df["arch_label"] == arch].copy()
        sub = sub.dropna(subset=["mean_t_succ"] + TICK_RATES)
        if len(sub) < 30:
            continue
        # Simple linear regression via numpy
        X = sub[TICK_RATES].values
        y = sub["mean_t_succ"].values
        # Add intercept
        X_aug = np.column_stack([np.ones(len(X)), X])
        try:
            coefs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            r2 = 1 - np.var(y - X_aug @ coefs) / np.var(y)
            rows.append({
                "arch": arch,
                "intercept": coefs[0],
                "q_coef": coefs[1],
                "r_coef": coefs[2],
                "m_coef": coefs[3],
                "r2": r2,
            })
        except Exception:
            continue
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    archs = [r["arch"] for r in rows]
    qc = [r["q_coef"] for r in rows]
    rc = [r["r_coef"] for r in rows]
    mc = [r["m_coef"] for r in rows]
    x = np.arange(len(archs))
    width = 0.27
    ax.bar(x - width, qc, width, label="Q coefficient", color="#3498db")
    ax.bar(x, rc, width, label="R coefficient", color="#e74c3c")
    ax.bar(x + width, mc, width, label="M coefficient", color="#27ae60")
    for i, r in enumerate(rows):
        ax.text(i, max(qc[i], rc[i], mc[i]) + 0.5, f"R²={r['r2']:.2f}",
                ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("regression coefficient")
    ax.set_title("Linear attribution: mean_t_succ ~ Q_rate + R_rate + M_rate per arch\n"
                 "Negative = stage_rate↑ → t_succ↓.  Strongest negative = bottleneck stage.")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────── bottleneck shift test (M↔C)


def test_bottleneck_shift(df: pd.DataFrame) -> Dict:
    """REVISED hypothesis based on data:

    Cadence shifts the bottleneck between two CONDITIONAL transitions:
      - P(M|R): probability of materialising given retrieval succeeded
      - P(C|M): probability of completing given materialisation

    Prediction:
      - As K decreases (faster broadcast), P(M|R) increases (fresh memory →
        agents can lock onto valid targets reliably)
      - As K decreases, P(C|M) decreases (peers race for the same targets →
        committed agents get blocked)
      - K* maximises the PRODUCT P(M|R) × P(C|M)
    """
    from scipy.stats import spearmanr
    sub = df[(df["architecture"] == "peer")]
    if sub.empty:
        return {}
    out = {}

    pmr_corr, pmr_p = spearmanr(sub["broadcast_every_k"], sub["p_M_given_R"],
                                  nan_policy="omit")
    pcm_corr, pcm_p = spearmanr(sub["broadcast_every_k"], sub["p_C_given_M"],
                                  nan_policy="omit")
    out["p_M_given_R_vs_K"] = {
        "spearman": float(pmr_corr), "p": float(pmr_p),
        "interpretation": (
            "negative = P(M|R) decreases with K (stale info hurts commitment)"
            if pmr_corr < 0 else "positive"
        ),
    }
    out["p_C_given_M_vs_K"] = {
        "spearman": float(pcm_corr), "p": float(pcm_p),
        "interpretation": (
            "positive = P(C|M) increases with K (less peer racing → committed = succeeds)"
            if pcm_corr > 0 else "negative"
        ),
    }

    # Bottleneck shift M↔C supported when both correlations have predicted
    # signs AND are significant
    out["supports_bottleneck_shift_MC"] = bool(
        pmr_corr < 0 and pcm_corr > 0 and pmr_p < 0.05 and pcm_p < 0.05
    )

    # ────────────────────────────────────────────────────
    # Two summaries of "the product P(M|R)*P(C|M) per K":
    # (a) product-of-means  — quick visual, prone to Jensen inflation when
    #     within-K correlation between the two rates is strongly negative.
    # (b) mean-of-products  — the proper per-configuration expected product
    #     under the chain decomposition; we report this as primary.
    valid = sub.dropna(subset=["p_M_given_R", "p_C_given_M"]).copy()
    valid["prod"] = valid["p_M_given_R"] * valid["p_C_given_M"]

    products_of_means = {}
    means_of_products = {}
    within_corr = {}
    rng = np.random.default_rng(42)
    B = 5000
    ci_marginal = {}
    ci_paired = {}
    for k, g in valid.groupby("broadcast_every_k"):
        pmr = g["p_M_given_R"].values
        pcm = g["p_C_given_M"].values
        products_of_means[int(k)] = float(pmr.mean() * pcm.mean())
        means_of_products[int(k)] = float(g["prod"].mean())
        within_corr[int(k)] = float(np.corrcoef(pmr, pcm)[0, 1])
        boot = np.array([g["prod"].values[rng.integers(0, len(g), len(g))].mean()
                         for _ in range(B)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ci_paired[int(k)] = [float(lo), float(hi)]

    out["product_of_means_per_K"] = products_of_means
    out["mean_of_products_per_K"] = means_of_products
    out["within_K_corr_PMR_PCM"] = within_corr
    out["mean_of_products_95ci"] = ci_paired

    k_star_marginal = max(products_of_means, key=products_of_means.get)
    k_star_proper = max(means_of_products, key=means_of_products.get)
    out["K_star_from_product_of_means"] = int(k_star_marginal)
    out["K_star_from_mean_of_products"] = int(k_star_proper)
    out["jensen_gap_at_K_star_marginal"] = (
        products_of_means[k_star_marginal] - means_of_products[k_star_marginal]
    )

    # Pairwise paired-bootstrap between K_star_proper and each other K
    sub["cell"] = (sub["n_agents"].astype(str) + "_"
                   + sub["n_waters"].astype(str) + "_"
                   + sub["layout"] + "_"
                   + sub["hazard_density"].astype(str) + "_"
                   + sub["seed"].astype(str))
    valid["cell"] = (valid["n_agents"].astype(str) + "_"
                     + valid["n_waters"].astype(str) + "_"
                     + valid["layout"] + "_"
                     + valid["hazard_density"].astype(str) + "_"
                     + valid["seed"].astype(str))
    pairwise = {}
    p_proper = valid[valid["broadcast_every_k"] == k_star_proper].set_index("cell")["prod"]
    for k_other in sorted(means_of_products.keys()):
        if k_other == k_star_proper:
            continue
        p_o = valid[valid["broadcast_every_k"] == k_other].set_index("cell")["prod"]
        j = pd.concat([p_proper, p_o], axis=1, keys=["k_star", "k_o"]).dropna()
        diffs = (j["k_star"] - j["k_o"]).values
        if len(diffs) == 0:
            continue
        boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                         for _ in range(B)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pairwise[f"K{k_star_proper}_vs_K{k_other}"] = {
            "delta_mean": float(diffs.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "n_pairs": int(len(diffs)),
            "significant": bool(lo > 0 or hi < 0),
        }
    out["paired_K_star_vs_others"] = pairwise

    return out


def plot_product_curve(df: pd.DataFrame, out_path: str) -> None:
    """Plot P(M|R) × P(C|M) vs K — the "effective navigation rate" curve."""
    sub = df[(df["architecture"] == "peer")]
    if sub.empty:
        return
    per_k = sub.groupby("broadcast_every_k").agg({
        "p_M_given_R": "mean",
        "p_C_given_M": "mean",
        "mean_t_succ": "mean",
        "success_rate": "mean",
    }).reset_index().sort_values("broadcast_every_k")
    per_k["product"] = per_k["p_M_given_R"] * per_k["p_C_given_M"]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(per_k["broadcast_every_k"], per_k["p_M_given_R"],
             marker="o", color="#f39c12", label="P(M|R)", linewidth=1.8)
    ax1.plot(per_k["broadcast_every_k"], per_k["p_C_given_M"],
             marker="s", color="#16a085", label="P(C|M)", linewidth=1.8)
    ax1.plot(per_k["broadcast_every_k"], per_k["product"],
             marker="D", color="#9b59b6", label="P(M|R) × P(C|M)",
             linewidth=2.3)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("peer cadence K")
    ax1.set_ylabel("conditional rate / product")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(per_k["broadcast_every_k"], per_k["mean_t_succ"],
             marker="x", color="#7f8c8d", linestyle="--",
             label="mean t_succ", alpha=0.7, linewidth=1.6)
    ax2.set_ylabel("mean t_succ", color="#7f8c8d")
    ax2.tick_params(axis="y", labelcolor="#7f8c8d")
    ax2.legend(loc="upper right", fontsize=9)

    ax1.set_title(
        "Bottleneck shift M↔C in peer architecture\n"
        "P(M|R) drops at large K (stale memory). P(C|M) drops at small K (peer racing). "
        "Product peaks at K* — predicted by framework."
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────── main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_runs(args.runs_csv)

    plot_stage_profile(df, os.path.join(args.out_dir, "stage_profile_episode.png"))
    plot_per_tick_stage_profile(df, os.path.join(args.out_dir, "stage_profile_pertick.png"))
    plot_conditional_rates(df, os.path.join(args.out_dir, "stage_conditional_rates.png"))
    plot_stage_attribution(df, os.path.join(args.out_dir, "stage_attribution.png"))
    plot_product_curve(df, os.path.join(args.out_dir, "bottleneck_product_MC.png"))

    for layout in ["asymmetric", "random", "symmetric"]:
        plot_bottleneck_shift(
            df, os.path.join(args.out_dir, f"bottleneck_shift_{layout}.png"),
            layout=layout,
        )

    bottleneck = test_bottleneck_shift(df)
    out_json = {
        "bottleneck_shift_test": bottleneck,
    }
    with open(os.path.join(args.out_dir, "qrmc_validation.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"✓ QRMC analysis written to {args.out_dir}")
    print("\nBottleneck shift test (M↔C):")
    for k, v in bottleneck.items():
        if isinstance(v, dict) and "spearman" in v:
            print(f"  {k}: spearman={v['spearman']:+.3f}  p={v['p']:.4f}  "
                  f"  ← {v['interpretation']}")
        elif k in ("product_of_means_per_K", "mean_of_products_per_K",
                   "within_K_corr_PMR_PCM", "mean_of_products_95ci"):
            print(f"  {k}:")
            for subk, subv in v.items():
                print(f"    K={subk}: {subv}")
        elif k == "paired_K_star_vs_others":
            print(f"  {k}:")
            for pair, stats in v.items():
                sig = "★" if stats["significant"] else " "
                print(f"    {pair}: Δ={stats['delta_mean']:+.4f}  "
                      f"CI=[{stats['ci_lo']:+.4f}, {stats['ci_hi']:+.4f}]  "
                      f"n={stats['n_pairs']}  {sig}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
