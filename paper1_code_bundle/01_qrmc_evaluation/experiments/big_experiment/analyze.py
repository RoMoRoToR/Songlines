"""Analysis: aggregates + phase diagram + cadence curves + claim validation."""

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

# ─────────────────────────────────────────────────────── helpers


def load_runs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Coerce numerics; pandas reads "nan" correctly already
    return df


def _eff_arch_label(row) -> str:
    if row["architecture"] == "peer":
        return f"peer(K={int(row['broadcast_every_k'])})"
    return row["architecture"]


def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (N, M, layout, architecture, K, hazard) across seeds."""
    df = df.copy()
    df["arch_label"] = df.apply(_eff_arch_label, axis=1)
    group_cols = ["n_agents", "n_waters", "layout", "arch_label",
                  "architecture", "broadcast_every_k", "hazard_density"]
    agg = df.groupby(group_cols).agg(
        n_seeds=("seed", "count"),
        success_rate_mean=("success_rate", "mean"),
        success_rate_std=("success_rate", "std"),
        mean_t_succ_mean=("mean_t_succ", "mean"),
        mean_t_succ_std=("mean_t_succ", "std"),
        p95_t_succ_mean=("p95_t_succ", "mean"),
        total_trail_mean=("total_trail", "mean"),
        n_hazard_hits_mean=("n_hazard_hits", "mean"),
    ).reset_index()
    agg["scarcity"] = agg["n_agents"] / agg["n_waters"]
    return agg


# ─────────────────────────────────────────────────────── plots


def plot_cadence_curves(
    agg: pd.DataFrame, out_path: str,
    layout: str = "asymmetric", hazard: float = 0.05,
) -> None:
    """For each (N, M), plot mean_t_succ vs K (peer only) + horizontal lines
    for independent / centralized / shared."""
    sub = agg[(agg["layout"] == layout) & (agg["hazard_density"] == hazard)]
    if sub.empty:
        return
    nm_pairs = sorted(set(zip(sub["n_agents"], sub["n_waters"])))
    n_plots = len(nm_pairs)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows),
                            squeeze=False)
    for idx, (n, m) in enumerate(nm_pairs):
        ax = axs[idx // ncols][idx % ncols]
        cell = sub[(sub["n_agents"] == n) & (sub["n_waters"] == m)]
        peer = cell[cell["architecture"] == "peer"].sort_values("broadcast_every_k")
        # cadence curve
        if not peer.empty:
            ax.errorbar(peer["broadcast_every_k"], peer["mean_t_succ_mean"],
                        yerr=peer["mean_t_succ_std"],
                        marker="o", linewidth=1.6, capsize=3,
                        color="#9b59b6", label="peer(K)")
        # baselines as horizontal lines
        for arch, color, style in [
            ("independent", "#7f8c8d", ":"),
            ("centralized", "#c0392b", "--"),
            ("shared",      "#2980b9", "-."),
        ]:
            row = cell[cell["architecture"] == arch]
            if not row.empty:
                y = row["mean_t_succ_mean"].iloc[0]
                ax.axhline(y, color=color, linestyle=style, linewidth=1.2,
                           label=arch, alpha=0.85)
        ax.set_title(f"N={n}, M={m}  (ρ={n/m:.2f})", fontsize=11)
        ax.set_xlabel("peer cadence K", fontsize=9)
        ax.set_ylabel("mean t_succ", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    # hide unused subplots
    for idx in range(n_plots, nrows * ncols):
        axs[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(
        f"Cadence curves — layout={layout}, hazard={hazard:.0%}\n"
        "Convex shape with minimum at K* > 1 validates H1; K* growing with ρ validates H2.",
        fontsize=11, y=0.995,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_phase_diagram(
    agg: pd.DataFrame, out_path: str,
    layout: str = "asymmetric", hazard: float = 0.05,
) -> None:
    """Heatmap: rows = scarcity ρ, cols = K (peer only), cells = mean_t_succ."""
    sub = agg[(agg["layout"] == layout)
              & (agg["hazard_density"] == hazard)
              & (agg["architecture"] == "peer")]
    if sub.empty:
        return
    # Aggregate over (N, M) → one row per unique scarcity
    sub = sub.copy()
    sub["scarcity_bin"] = sub["scarcity"].round(2)
    pivot = sub.pivot_table(
        index="scarcity_bin", columns="broadcast_every_k",
        values="mean_t_succ_mean", aggfunc="mean",
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"K={k}" for k in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"ρ={v}" for v in pivot.index])
    ax.set_xlabel("peer broadcast cadence K")
    ax.set_ylabel("resource scarcity ρ = N/M")

    # annotate K* per row
    for i in range(len(pivot.index)):
        row = pivot.iloc[i].values
        finite = ~np.isnan(row)
        if finite.any():
            k_star_col = np.nanargmin(row)
            ax.add_patch(plt.Rectangle((k_star_col - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=2))

    # cell labels
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="white")

    fig.colorbar(im, ax=ax, label="mean t_succ")
    ax.set_title(
        f"Phase diagram — layout={layout}, hazard={hazard:.0%}\n"
        "Red box = K* (minimum mean_t_succ for that scarcity)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(agg: pd.DataFrame, out_path: str,
                layout: str = "asymmetric", hazard: float = 0.05) -> None:
    """Pareto: success_rate vs mean_t_succ, coloured by architecture."""
    sub = agg[(agg["layout"] == layout) & (agg["hazard_density"] == hazard)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "independent": "#7f8c8d",
        "shared":      "#2980b9",
        "centralized": "#c0392b",
        "peer":        "#9b59b6",
    }
    for arch, color in colors.items():
        rows = sub[sub["architecture"] == arch]
        if rows.empty:
            continue
        ax.scatter(rows["mean_t_succ_mean"], rows["success_rate_mean"],
                   c=color, s=60, label=arch, alpha=0.7,
                   edgecolors="black", linewidth=0.5)
    ax.set_xlabel("mean t_succ (lower = faster)")
    ax.set_ylabel("success_rate (higher = better)")
    ax.set_title(
        f"Pareto frontier — layout={layout}, hazard={hazard:.0%}\n"
        "Top-left = fast and complete.  Peer should populate the frontier."
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_hazard_robustness(agg: pd.DataFrame, out_path: str,
                            layout: str = "asymmetric") -> None:
    """For each architecture, plot mean_t_succ vs hazard_density."""
    sub = agg[agg["layout"] == layout]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {
        "independent": "#7f8c8d",
        "shared":      "#2980b9",
        "centralized": "#c0392b",
        "peer":        "#9b59b6",
    }
    for arch, color in colors.items():
        rows = sub[sub["architecture"] == arch]
        if rows.empty:
            continue
        grouped = rows.groupby("hazard_density")["mean_t_succ_mean"].mean().reset_index()
        ax.plot(grouped["hazard_density"], grouped["mean_t_succ_mean"],
                marker="o", color=color, label=arch, linewidth=1.6)
    ax.set_xlabel("hazard density")
    ax.set_ylabel("mean t_succ (averaged over all N, M, K)")
    ax.set_title(f"Robustness to hazards — layout={layout}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────── claim validation


def _bootstrap_mean(values, n_iter=2000, ci=0.95):
    rng = np.random.default_rng(42)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    samples = rng.choice(values, size=(n_iter, n), replace=True)
    means = np.nanmean(samples, axis=1)
    alpha = (1 - ci) / 2
    lo = np.percentile(means, 100 * alpha)
    hi = np.percentile(means, 100 * (1 - alpha))
    return float(np.nanmean(values)), float(lo), float(hi)


def validate_claim(df: pd.DataFrame) -> Dict:
    """Test H1, H2, H3, H4 from raw run data."""
    df = df.copy()
    out: Dict = {}

    # H1: for each (N, M, layout) with M < N, K* in peer(K∈{1,2,4,8,16})
    # has finite minimum strictly inside the range
    h1_results = []
    for (n, m, layout), grp in df[df["architecture"] == "peer"].groupby(
            ["n_agents", "n_waters", "layout"]):
        if n <= m:
            continue
        per_k = grp.groupby("broadcast_every_k")["mean_t_succ"].mean()
        if per_k.dropna().empty:
            continue
        k_star = int(per_k.idxmin())
        finite_ks = per_k.dropna().index.tolist()
        boundary = (k_star == min(finite_ks)) or (k_star == max(finite_ks))
        h1_results.append({
            "n": n, "m": m, "layout": layout,
            "k_star": k_star, "interior": not boundary,
            "mean_t_at_k_star": float(per_k.min()),
            "mean_t_at_k1": float(per_k.get(1, float("nan"))),
            "mean_t_at_k_max": float(per_k.get(max(finite_ks), float("nan"))),
        })
    interior_count = sum(1 for r in h1_results if r["interior"])
    out["H1_existence_of_interior_K_star"] = {
        "n_cases": len(h1_results),
        "n_interior": interior_count,
        "interior_fraction": (interior_count / len(h1_results)
                              if h1_results else float("nan")),
        "supports_H1": (interior_count / max(1, len(h1_results)) > 0.5),
        "details": h1_results,
    }

    # H2: K* should grow (non-decreasing trend) with scarcity ρ = N/M
    if h1_results:
        rhos = [r["n"] / r["m"] for r in h1_results]
        ks = [r["k_star"] for r in h1_results]
        # Spearman correlation
        if len(set(rhos)) > 1:
            from scipy.stats import spearmanr
            rho_corr, p_val = spearmanr(rhos, ks)
            out["H2_K_star_grows_with_scarcity"] = {
                "spearman_rho": float(rho_corr),
                "p_value": float(p_val),
                "supports_H2": bool(rho_corr > 0 and p_val < 0.10),
            }
        else:
            out["H2_K_star_grows_with_scarcity"] = {"note": "insufficient ρ variation"}

    # H3: for M >= N (ρ ≤ 1), K* should be 1 (fast broadcast wins)
    h3_cases = []
    for (n, m, layout), grp in df[df["architecture"] == "peer"].groupby(
            ["n_agents", "n_waters", "layout"]):
        if n > m:
            continue
        per_k = grp.groupby("broadcast_every_k")["mean_t_succ"].mean()
        if per_k.dropna().empty:
            continue
        k_star = int(per_k.idxmin())
        h3_cases.append({"n": n, "m": m, "layout": layout, "k_star": k_star})
    n_k1 = sum(1 for c in h3_cases if c["k_star"] == 1)
    out["H3_K_star_is_1_when_no_scarcity"] = {
        "n_cases": len(h3_cases),
        "n_with_k_star_eq_1": n_k1,
        "fraction": n_k1 / max(1, len(h3_cases)),
        "supports_H3": (len(h3_cases) > 0 and n_k1 / len(h3_cases) > 0.5),
    }

    # Headline: average t_succ per architecture (over ALL valid configs)
    arch_mean = []
    for arch, grp in df.groupby("architecture"):
        m, lo, hi = _bootstrap_mean(grp["mean_t_succ"].dropna().values)
        arch_mean.append({
            "architecture": arch, "n_runs": len(grp),
            "mean_t_succ": m, "ci95_lo": lo, "ci95_hi": hi,
        })
    out["headline_mean_t_succ_by_architecture"] = arch_mean

    # Peer at K* vs centralized
    # Compute "best peer K per (N, M, layout, hazard)" and compare to centralized
    rows_peer_best = []
    rows_centralized = []
    for (n, m, layout, hazard), grp in df.groupby(
            ["n_agents", "n_waters", "layout", "hazard_density"]):
        if n <= m:
            continue
        peer = grp[grp["architecture"] == "peer"]
        cent = grp[grp["architecture"] == "centralized"]
        if peer.empty or cent.empty:
            continue
        # Best peer K (lowest mean across seeds)
        per_k = peer.groupby("broadcast_every_k")["mean_t_succ"].mean()
        best_k = per_k.idxmin()
        peer_at_best = peer[peer["broadcast_every_k"] == best_k]["mean_t_succ"]
        rows_peer_best.extend(peer_at_best.dropna().tolist())
        rows_centralized.extend(cent["mean_t_succ"].dropna().tolist())
    if rows_peer_best and rows_centralized:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(rows_peer_best, rows_centralized, alternative="less")
        peer_mean = float(np.mean(rows_peer_best))
        cent_mean = float(np.mean(rows_centralized))
        out["peer_at_K_star_vs_centralized"] = {
            "peer_mean": peer_mean,
            "centralized_mean": cent_mean,
            "delta": peer_mean - cent_mean,
            "mann_whitney_p_one_sided": float(p),
            "supports_peer_better": bool(p < 0.05 and peer_mean < cent_mean),
        }

    return out


# ─────────────────────────────────────────────────────── main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--layouts", default="symmetric,asymmetric,random")
    parser.add_argument("--hazards", default="0.0,0.05,0.10")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_runs(args.runs_csv)
    agg = compute_aggregates(df)
    agg.to_csv(os.path.join(args.out_dir, "aggregates.csv"), index=False)

    layouts = [s.strip() for s in args.layouts.split(",") if s.strip()]
    hazards = [float(s) for s in args.hazards.split(",") if s.strip()]

    for layout in layouts:
        for hazard in hazards:
            tag = f"{layout}_h{hazard}".replace(".", "")
            plot_cadence_curves(
                agg, os.path.join(args.out_dir, f"cadence_curves_{tag}.png"),
                layout=layout, hazard=hazard,
            )
            plot_phase_diagram(
                agg, os.path.join(args.out_dir, f"phase_diagram_{tag}.png"),
                layout=layout, hazard=hazard,
            )
            plot_pareto(
                agg, os.path.join(args.out_dir, f"pareto_{tag}.png"),
                layout=layout, hazard=hazard,
            )

    for layout in layouts:
        plot_hazard_robustness(
            agg, os.path.join(args.out_dir, f"hazard_robustness_{layout}.png"),
            layout=layout,
        )

    claim = validate_claim(df)
    with open(os.path.join(args.out_dir, "claim_validation.json"), "w") as f:
        json.dump(claim, f, indent=2, default=str)

    print(f"✓ Analysis written to {args.out_dir}")
    print("\nClaim validation summary:")
    h1 = claim.get("H1_existence_of_interior_K_star", {})
    print(f"  H1 (interior K* exists): supports={h1.get('supports_H1')}  "
          f"interior_fraction={h1.get('interior_fraction', 0):.2f} "
          f"({h1.get('n_interior', 0)}/{h1.get('n_cases', 0)})")
    h2 = claim.get("H2_K_star_grows_with_scarcity", {})
    if "spearman_rho" in h2:
        print(f"  H2 (K* grows with ρ): supports={h2.get('supports_H2')}  "
              f"spearman={h2['spearman_rho']:.3f}  p={h2['p_value']:.4f}")
    h3 = claim.get("H3_K_star_is_1_when_no_scarcity", {})
    print(f"  H3 (K*=1 when ρ≤1): supports={h3.get('supports_H3')}  "
          f"fraction_K1={h3.get('fraction', 0):.2f} "
          f"({h3.get('n_with_k_star_eq_1', 0)}/{h3.get('n_cases', 0)})")
    peer_vs_cent = claim.get("peer_at_K_star_vs_centralized")
    if peer_vs_cent:
        print(f"  Peer(K*) vs centralized: "
              f"peer={peer_vs_cent['peer_mean']:.2f} "
              f"vs centralized={peer_vs_cent['centralized_mean']:.2f}  "
              f"p={peer_vs_cent['mann_whitney_p_one_sided']:.4f}  "
              f"supports={peer_vs_cent['supports_peer_better']}")


if __name__ == "__main__":
    main()
