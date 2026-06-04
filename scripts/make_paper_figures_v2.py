"""Generate Figures 6, 7, 8 with the same style guide as figures 4 and 5.

  Fig 6 — N=12 scale-up bottleneck signature (peer cadence sweep)
  Fig 7 — CommNet training curve + Q/R/M/C profile vs symbolic peer K=8
  Fig 8 — Refined attribution: 91% empty calls + MLP performance
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── style guide (matches scripts/make_paper_figures.py) ────────────────
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


def _bootstrap_ci(values: np.ndarray, B: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ─────────────────────────────────────────────────────────────────────
# Figure 6 — N=12 scale-up bottleneck signature
# ─────────────────────────────────────────────────────────────────────


def make_figure6():
    df = pd.read_csv(os.path.join(REPO, "tmp", "big_experiment_N12", "runs.csv"))
    df = df[df["architecture"] == "peer"].dropna(
        subset=["p_M_given_R", "p_C_given_M"]
    ).copy()

    ks = [1, 2, 4, 8, 16, 32, 48, 64]
    rows = []
    for k in ks:
        sub = df[df["broadcast_every_k"] == k]
        pmr = sub["p_M_given_R"].values
        pcm = sub["p_C_given_M"].values
        tsucc = sub["mean_t_succ"].dropna().values
        corr = float(np.corrcoef(pmr, pcm)[0, 1])
        m_t, lo_t, hi_t = _bootstrap_ci(tsucc)
        rows.append({
            "K": k, "pmr": pmr.mean(), "pcm": pcm.mean(),
            "corr": corr,
            "t_mean": m_t, "t_lo": lo_t, "t_hi": hi_t,
        })
    d = pd.DataFrame(rows)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 6.8), sharex=True, gridspec_kw={"hspace": 0.14}
    )

    # ── top: both conditional rates + within-cadence correlation overlay ──
    ax_top.plot(d["K"], d["pmr"], marker="o", linewidth=2,
                color=COLORS["primary"], label=r"$P(M^\star|R^\star)$")
    ax_top.plot(d["K"], d["pcm"], marker="s", linewidth=2,
                color=COLORS["secondary"], label=r"$P(C^\star|M^\star)$")
    ax_top.set_xscale("log", base=2)
    ax_top.set_xticks(d["K"])
    ax_top.set_xticklabels([str(k) for k in d["K"]])
    ax_top.set_ylabel("conditional rate")
    ax_top.set_ylim(0.55, 0.92)
    ax_top.legend(loc="center right", frameon=False)
    ax_top.set_title(
        "Scale-up to $N=12$ ($n=8{,}640$ runs, 40 seeds)",
        fontsize=12, pad=6,
    )

    # right axis with within-cadence Pearson correlation
    ax_top2 = ax_top.twinx()
    ax_top2.plot(d["K"], d["corr"], marker="D", linestyle="--",
                 color=COLORS["accent"], linewidth=2, alpha=0.85,
                 label=r"within-cadence Pearson $r$")
    ax_top2.set_ylabel("Pearson $r$", color=COLORS["accent"])
    ax_top2.tick_params(axis="y", labelcolor=COLORS["accent"])
    ax_top2.set_ylim(-0.75, 0.0)
    ax_top2.spines["top"].set_visible(False)
    ax_top2.grid(False)
    # mark peak
    k_peak = int(d.loc[d["corr"].idxmin(), "K"])
    r_peak = d["corr"].min()
    ax_top2.scatter([k_peak], [r_peak], s=180, marker="*",
                    color=COLORS["accent"], edgecolors="black", linewidth=0.6,
                    zorder=10)
    ax_top2.annotate(
        f"peak $r = {r_peak:.2f}$\nat $K = {k_peak}$",
        xy=(k_peak, r_peak),
        xytext=(k_peak * 1.5, r_peak + 0.15),
        fontsize=10, color=COLORS["accent"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.0),
    )
    ax_top2.legend(loc="lower right", frameon=False)

    # ── bottom: mean t_succ ──────────────────────────────────────────
    ax_bot.errorbar(
        d["K"], d["t_mean"],
        yerr=[d["t_mean"] - d["t_lo"], d["t_hi"] - d["t_mean"]],
        marker="D", linewidth=2, color=COLORS["tertiary"],
        capsize=3, label=r"peer$(K)$",
    )
    kmin = d.loc[d["t_mean"].idxmin()]
    ax_bot.scatter([kmin["K"]], [kmin["t_mean"]], s=180, marker="*",
                   color=COLORS["accent"], zorder=10,
                   edgecolors="black", linewidth=0.5)
    ax_bot.annotate(
        f"$K^\\star = {int(kmin['K'])}$  (mean $t={kmin['t_mean']:.2f}$)",
        xy=(kmin["K"], kmin["t_mean"]),
        xytext=(kmin["K"] * 2.0, kmin["t_mean"] - 0.45),
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.2),
        fontsize=10, color=COLORS["accent"],
    )
    ax_bot.set_xscale("log", base=2)
    ax_bot.set_xticks(d["K"])
    ax_bot.set_xticklabels([str(k) for k in d["K"]])
    ax_bot.set_xlabel("peer broadcast cadence $K$")
    ax_bot.set_ylabel("mean time-to-success (ticks)")
    ax_bot.set_title(
        "Interior minimum preserved at $K=8$ ($t_{\\mathrm{succ}}=4.77$)",
        fontsize=12, pad=6,
    )
    ax_bot.legend(loc="upper left", frameon=False)

    out = os.path.join(FIG_DIR, "fig_scale_n12.pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(FIG_DIR, "fig_scale_n12.png"))
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────
# Figure 7 — CommNet training curve + Q/R/M/C profile vs symbolic peer K=8
# ─────────────────────────────────────────────────────────────────────


def make_figure7():
    with open(os.path.join(REPO, "experiments", "commnet_baseline", "train_log.json")) as f:
        tl = json.load(f)
    with open(os.path.join(REPO, "experiments", "commnet_baseline", "qrmc_eval.json")) as f:
        qe = json.load(f)
    s = qe["summary"]

    # Get symbolic peer K=8 numbers on N=3, M=2, asymmetric (matched scenario)
    # from the main 40-seed sweep
    base = pd.read_csv(os.path.join(REPO, "tmp", "big_experiment_qrmc_40", "runs.csv"))
    matched = base[
        (base["architecture"] == "peer")
        & (base["broadcast_every_k"] == 8)
        & (base["n_agents"] == 3)
        & (base["n_waters"] == 2)
        & (base["layout"] == "asymmetric")
    ].dropna(subset=["p_M_given_R", "p_C_given_M"])
    peer_pmr = float(matched["p_M_given_R"].mean())
    peer_pcm = float(matched["p_C_given_M"].mean())
    peer_succ = float(matched["success_rate"].mean())
    # Q* and R* approximation for peer: per-tick Q and R rates (saturated near 1)
    peer_q = float(matched["q_star_rate"].mean())
    peer_r = float(matched["r_star_rate"].mean())
    peer_m = float(matched["m_star_rate"].mean())
    peer_c = float(matched["c_star_rate"].mean())

    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.30)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # ── left: training curve (smoothed success rate over training episodes) ──
    sr = np.array(tl["success_rate"])
    window = 100
    smooth = np.convolve(sr, np.ones(window) / window, mode="valid")
    ax_l.plot(np.arange(len(smooth)) + window // 2, smooth,
              color=COLORS["primary"], linewidth=1.8,
              label=f"CommNet success (mean over {window}-ep window)")
    ax_l.axhline(peer_succ, color=COLORS["tertiary"], linestyle="--",
                 linewidth=1.5,
                 label=f"symbolic peer $K=8$ (eval, {peer_succ:.2f})")
    ax_l.set_xlabel("training episode")
    ax_l.set_ylabel("success rate")
    ax_l.set_ylim(0, max(0.9, peer_succ + 0.1))
    ax_l.set_title("CommNet REINFORCE training curve (2000 episodes)",
                   fontsize=12, pad=6)
    ax_l.legend(loc="upper right", frameon=False)

    # ── right: Q/R/M/C bar chart ────────────────────────────────────
    stages = ["$Q^\\star$", "$R^\\star$", "$M^\\star$", "$C^\\star$"]
    commnet_vals = [s["q_star_rate_mean"], s["r_star_rate_mean"],
                    s["m_star_rate_mean"], s["c_star_rate_mean"]]
    peer_vals = [peer_q, peer_r, peer_m, peer_c]
    x = np.arange(len(stages))
    width = 0.38
    ax_r.bar(x - width/2, commnet_vals, width, color=COLORS["primary"],
             label="CommNet (50 held-out seeds)", edgecolor="black", linewidth=0.4)
    ax_r.bar(x + width/2, peer_vals, width, color=COLORS["tertiary"],
             label=f"symbolic peer $K=8$", edgecolor="black", linewidth=0.4)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(stages, fontsize=12)
    ax_r.set_ylabel("episode-level rate")
    ax_r.set_ylim(0, 1.05)
    ax_r.set_title("Q/R/M/C profile: CommNet vs symbolic peer", fontsize=12, pad=6)
    # annotate Q saturation
    ax_r.annotate(
        "$Q$ saturated\n(degenerate)",
        xy=(0 - width/2, commnet_vals[0]),
        xytext=(1.4, 0.93),
        fontsize=9, color=COLORS["accent"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.0),
    )
    ax_r.legend(loc="center right", frameon=False, bbox_to_anchor=(1.0, 0.55))

    out = os.path.join(FIG_DIR, "fig_commnet_baseline.pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(FIG_DIR, "fig_commnet_baseline.png"))
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────
# Figure 8 — Refined attribution: empty-set diagnostic + MLP underperforms majority
# ─────────────────────────────────────────────────────────────────────


def make_figure8():
    # ── left: composition of 349 retrieval calls ──────────────────────
    # 317 empty (91%), 32 with candidates, of which 31 satisfied (97%)
    n_empty = 317
    n_satisfied = 31
    n_unsatisfied = 1

    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.0], wspace=0.30)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    cats = ["empty\n(no candidate)",
            "non-empty,\nsatisfied",
            "non-empty,\nunsatisfied"]
    counts = [n_empty, n_satisfied, n_unsatisfied]
    pcts = [100 * c / sum(counts) for c in counts]
    colors_bar = [COLORS["accent"], COLORS["tertiary"], COLORS["secondary"]]
    bars = ax_l.bar(cats, counts, color=colors_bar, edgecolor="black", linewidth=0.4)
    for bar, count, pct in zip(bars, counts, pcts):
        ax_l.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 6,
                  f"{count}\n({pct:.1f}%)",
                  ha="center", va="bottom", fontsize=10)
    ax_l.set_ylabel("retrieval calls")
    ax_l.set_ylim(0, max(counts) * 1.25)
    ax_l.set_title(
        "Diagnostic refinement: 91% of retrieval calls have no candidates",
        fontsize=12, pad=6,
    )

    # ── right: train/val/test accuracy vs majority-class baseline ─────
    splits = ["train", "val", "test"]
    mlp_acc = [0.584, 0.553, 0.590]
    maj_acc = [0.903, 0.776, 0.813]
    x = np.arange(len(splits))
    width = 0.38
    ax_r.bar(x - width/2, mlp_acc, width, color=COLORS["primary"],
             label="MLP candidate-generator", edgecolor="black", linewidth=0.4)
    ax_r.bar(x + width/2, maj_acc, width, color=COLORS["neutral"],
             label="majority-class baseline", edgecolor="black", linewidth=0.4)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(splits)
    ax_r.set_ylabel("accuracy")
    ax_r.set_ylim(0, 1.0)
    ax_r.set_title(
        "MLP underperforms majority-class baseline on all splits",
        fontsize=12, pad=6,
    )
    ax_r.axhline(0.5, color="lightgrey", linestyle=":", linewidth=0.8)
    ax_r.legend(loc="lower right", frameon=False)

    fig.suptitle(
        "Refined attribution: retrieval bottleneck is candidate generation, not ranking",
        fontsize=12, y=1.02,
    )

    out = os.path.join(FIG_DIR, "fig_refined_attribution.pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(FIG_DIR, "fig_refined_attribution.png"))
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Figure 6 (N=12 scale-up):")
    make_figure6()
    print("Figure 7 (CommNet baseline):")
    make_figure7()
    print("Figure 8 (refined attribution):")
    make_figure8()
    print("Done.")


if __name__ == "__main__":
    main()
