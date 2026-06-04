"""Generate Figure 4 (bottleneck-shift line plot) and Figure 5 (Jensen scatter).

Both figures use the 25,920-run base sweep (K<=16) + 9,720-run extra-K sweep
(K in {32, 48, 64}) at 40 seeds.  No synthetic data.  All numbers are computed
from CSVs and rendered with matplotlib using the paper-wide style guide.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paper-wide style guide ────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "primary":   "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary":  "#2ca02c",
    "accent":    "#d62728",
    "neutral":   "#7f7f7f",
}

REPO = "/Users/taniyashuba/PycharmProjects/Songlines"
FIG_DIR = os.path.join(
    REPO, "docs", "Formatting_Instructions_For_NeurIPS_2026",
    "songlines_symbolic_memory_figures",
)


def _load_peer_sweep() -> pd.DataFrame:
    base = pd.read_csv(os.path.join(REPO, "tmp", "big_experiment_qrmc_40", "runs.csv"))
    extra = pd.read_csv(os.path.join(REPO, "tmp", "big_experiment_extraK", "runs.csv"))
    df = pd.concat([base, extra], ignore_index=True)
    return df.dropna(subset=["p_M_given_R", "p_C_given_M"]).copy()


def _bootstrap_ci(values: np.ndarray, B: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ──────────────────────────────────────────────────────────────────────
# Figure 4 — bottleneck-shift line plot
# Top: P(M*|R*) and P(C*|M*) vs K, with CIs
# Bottom: mean_t_succ vs K, with U-shape and indep baseline
# ──────────────────────────────────────────────────────────────────────


def make_figure4():
    df = _load_peer_sweep()
    peer = df[df["architecture"] == "peer"]
    ks = [1, 2, 4, 8, 16, 32, 48, 64]

    rows = []
    for k in ks:
        sub = peer[peer["broadcast_every_k"] == k]
        pmr = sub["p_M_given_R"].values
        pcm = sub["p_C_given_M"].values
        tsucc = sub["mean_t_succ"].dropna().values
        pmr_m, pmr_lo, pmr_hi = _bootstrap_ci(pmr)
        pcm_m, pcm_lo, pcm_hi = _bootstrap_ci(pcm)
        t_m, t_lo, t_hi = _bootstrap_ci(tsucc)
        rows.append({
            "K": k,
            "pmr_mean": pmr_m, "pmr_lo": pmr_lo, "pmr_hi": pmr_hi,
            "pcm_mean": pcm_m, "pcm_lo": pcm_lo, "pcm_hi": pcm_hi,
            "tsucc_mean": t_m, "tsucc_lo": t_lo, "tsucc_hi": t_hi,
        })
    d = pd.DataFrame(rows)

    indep = df[df["architecture"] == "independent"]
    indep_t_m, indep_t_lo, indep_t_hi = _bootstrap_ci(indep["mean_t_succ"].dropna().values)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True,
                                          gridspec_kw={"hspace": 0.12})

    # ── top: conditional rates ───────────────────────────────
    ax_top.errorbar(
        d["K"], d["pmr_mean"],
        yerr=[d["pmr_mean"] - d["pmr_lo"], d["pmr_hi"] - d["pmr_mean"]],
        marker="o", linewidth=2, color=COLORS["primary"],
        capsize=3, label=r"$P(M^\star | R^\star)$",
    )
    ax_top.errorbar(
        d["K"], d["pcm_mean"],
        yerr=[d["pcm_mean"] - d["pcm_lo"], d["pcm_hi"] - d["pcm_mean"]],
        marker="s", linewidth=2, color=COLORS["secondary"],
        capsize=3, label=r"$P(C^\star | M^\star)$",
    )

    # Crossing visualisation: find where the two interpolated curves cross
    # (roughly between K=4 and K=8 by inspection)
    ax_top.axvline(8, color=COLORS["neutral"], linestyle="--", linewidth=1, alpha=0.6)
    ax_top.text(8.3, 0.71, "K = 8\n(time optimum)", fontsize=9, color=COLORS["neutral"])

    ax_top.set_xscale("log", base=2)
    ax_top.set_xticks(d["K"])
    ax_top.set_xticklabels([str(k) for k in d["K"]])
    ax_top.set_ylabel("conditional rate")
    ax_top.set_ylim(0.55, 1.02)
    ax_top.set_title("Bottleneck shift: as $K$ decreases, $P(M^\\star|R^\\star)$ rises and $P(C^\\star|M^\\star)$ falls",
                     fontsize=12, pad=6)
    ax_top.legend(loc="center right", frameon=False)

    # ── bottom: mean t_succ ──────────────────────────────────
    ax_bot.errorbar(
        d["K"], d["tsucc_mean"],
        yerr=[d["tsucc_mean"] - d["tsucc_lo"], d["tsucc_hi"] - d["tsucc_mean"]],
        marker="D", linewidth=2, color=COLORS["tertiary"],
        capsize=3, label=r"peer$(K)$",
    )

    # Independent baseline as a horizontal line
    ax_bot.axhline(indep_t_m, color=COLORS["neutral"], linestyle="--",
                   linewidth=1.5, label=f"independent baseline ({indep_t_m:.2f})")
    ax_bot.fill_between(
        d["K"], indep_t_lo, indep_t_hi,
        color=COLORS["neutral"], alpha=0.12,
    )

    # Annotate K=8 minimum
    kmin = d.loc[d["tsucc_mean"].idxmin()]
    ax_bot.scatter([kmin["K"]], [kmin["tsucc_mean"]], s=140, marker="*",
                   color=COLORS["accent"], zorder=10, edgecolors="black", linewidth=0.5)
    ax_bot.annotate(
        f"$K^\\star = {int(kmin['K'])}$  (mean $t_{{\\mathrm{{succ}}}}={kmin['tsucc_mean']:.2f}$)",
        xy=(kmin["K"], kmin["tsucc_mean"]),
        xytext=(kmin["K"] * 2.0, kmin["tsucc_mean"] - 0.55),
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.2),
        fontsize=10, color=COLORS["accent"],
    )

    ax_bot.set_xscale("log", base=2)
    ax_bot.set_xticks(d["K"])
    ax_bot.set_xticklabels([str(k) for k in d["K"]])
    ax_bot.set_xlabel("peer broadcast cadence $K$")
    ax_bot.set_ylabel("mean time-to-success (ticks)")
    ax_bot.set_title("Efficiency metric has interior minimum at $K=8$, below independent baseline",
                     fontsize=12, pad=6)
    ax_bot.legend(loc="upper center", frameon=False, ncol=2)

    out = os.path.join(FIG_DIR, "fig_bottleneck_shift.pdf")
    fig.savefig(out)
    out_png = os.path.join(FIG_DIR, "fig_bottleneck_shift.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out_png}")


# ──────────────────────────────────────────────────────────────────────
# Figure 5 — Jensen scatter: per-configuration (P(M|R), P(C|M)) at K in {1, 4, 64}
# ──────────────────────────────────────────────────────────────────────


def make_figure5():
    df = _load_peer_sweep()
    peer = df[df["architecture"] == "peer"]
    selected_ks = [1, 4, 64]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)

    for ax, k in zip(axes, selected_ks):
        sub = peer[peer["broadcast_every_k"] == k]
        x = sub["p_M_given_R"].values
        y = sub["p_C_given_M"].values
        corr = np.corrcoef(x, y)[0, 1]

        ax.scatter(x, y, s=8, alpha=0.35, color=COLORS["primary"], edgecolors="none")

        # OLS regression line for visual
        if len(x) > 2 and np.std(x) > 1e-6:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m * xs + b, color=COLORS["accent"], linewidth=1.6,
                    label=f"OLS slope = {m:.2f}")

        ax.set_title(f"$K = {k}$  (Pearson $r = {corr:+.2f}$)", fontsize=12, pad=4)
        ax.set_xlabel(r"$P(M^\star | R^\star)$ per configuration")
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_aspect("equal")
        ax.legend(loc="lower left", frameon=False, fontsize=9)

    axes[0].set_ylabel(r"$P(C^\star | M^\star)$ per configuration")
    fig.suptitle("Bottleneck-shift signature: negative within-cadence correlation peaks at $K=4$ and decays at $K=64$",
                 fontsize=12, y=1.04)

    out = os.path.join(FIG_DIR, "fig_jensen_scatter.pdf")
    fig.savefig(out)
    out_png = os.path.join(FIG_DIR, "fig_jensen_scatter.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out_png}")


# ──────────────────────────────────────────────────────────────────────
def main():
    print("Figure 4 (bottleneck-shift line plot):")
    make_figure4()
    print("Figure 5 (Jensen scatter):")
    make_figure5()
    print("Done.")


if __name__ == "__main__":
    main()
