"""Generate Figures 1, 2, 3, 9 (the "more pretty pictures" set).

  Fig 1 (hero) — Q/R/M/C single-agent pipeline overview with example
  Fig 2 — Four multi-agent architectures schematic
  Fig 3 — Per-architecture conditional rates (replaces heavy Table 1)
  Fig 9 — Oracle waterfall (FourRooms + LavaGapS7) attribution

Same style guide as v2 (DejaVu Sans, 4-color palette, no top/right spines).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
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
STAGE_COLORS = {
    "Q": "#4C72B0",
    "R": "#55A868",
    "M": "#C44E52",
    "C": "#8172B2",
}

REPO = "/Users/taniyashuba/PycharmProjects/Songlines"
FIG_DIR = os.path.join(
    REPO, "docs", "Formatting_Instructions_For_NeurIPS_2026",
    "songlines_symbolic_memory_figures",
)


# ────────────────────────────────────────────────────────────────────────
# Fig 1 — Hero Q/R/M/C overview
# ────────────────────────────────────────────────────────────────────────
def fig1_hero_overview():
    fig = plt.figure(figsize=(13, 6.0))
    gs = fig.add_gridspec(
        2, 4, height_ratios=[1.0, 1.1],
        hspace=0.40, wspace=0.40,
        left=0.04, right=0.98, top=0.91, bottom=0.07,
    )

    # ── Top row: 4 stage boxes Q → R → M → C ────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 3.0)
    ax_top.axis("off")

    stages = [
        ("Q", "Query formation",
         "Build a semantic query\nfrom intent and state",
         r"$Q^\star$: query has $\geq 1$ candidate",
         STAGE_COLORS["Q"]),
        ("R", "Retrieval",
         "Memory returns a candidate\nsatisfying the predicate",
         r"$R^\star$: candidate satisfies predicate",
         STAGE_COLORS["R"]),
        ("M", "Materialization",
         "Planner locks onto a concrete\nreachable target cell",
         r"$M^\star$: target lock acquired",
         STAGE_COLORS["M"]),
        ("C", "Completion",
         "Controller executes and\nreaches the materialized target",
         r"$C^\star$: agent reaches target",
         STAGE_COLORS["C"]),
    ]
    box_w = 2.15
    box_h = 2.30
    box_gap = 0.22
    x0 = 0.22
    for i, (letter, name, desc, evt, col) in enumerate(stages):
        x = x0 + i * (box_w + box_gap)
        y_bot = 0.20
        # Outer rounded box
        bbox = FancyBboxPatch(
            (x, y_bot), box_w, box_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.6, edgecolor=col, facecolor=col + "15",
        )
        ax_top.add_patch(bbox)
        # Letter circle (top-left)
        ax_top.add_patch(Circle(
            (x + 0.35, y_bot + box_h - 0.40), 0.24,
            facecolor=col, edgecolor="white", linewidth=2, zorder=3,
        ))
        ax_top.text(x + 0.35, y_bot + box_h - 0.40, letter,
                    ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white", zorder=4)
        # Name (next to letter)
        ax_top.text(x + 0.72, y_bot + box_h - 0.40, name,
                    ha="left", va="center", fontsize=12,
                    fontweight="bold", color="#222")
        # Description (middle)
        ax_top.text(x + box_w / 2, y_bot + box_h - 1.30, desc,
                    ha="center", va="center", fontsize=10, color="#333",
                    linespacing=1.30)
        # Separator
        ax_top.plot([x + 0.18, x + box_w - 0.18], [y_bot + 0.55, y_bot + 0.55],
                    color=col, linewidth=0.8, alpha=0.5)
        # Event label below separator
        ax_top.text(x + box_w / 2, y_bot + 0.30, evt,
                    ha="center", va="center", fontsize=9.0,
                    color=col, fontweight="bold",
                    fontstyle="italic")
        # Arrow to next
        if i < len(stages) - 1:
            ax_top.add_patch(FancyArrowPatch(
                (x + box_w + 0.01, y_bot + box_h / 2),
                (x + box_w + box_gap - 0.01, y_bot + box_h / 2),
                arrowstyle="-|>", mutation_scale=20,
                color="#555", linewidth=2.0,
            ))

    ax_top.text(5.0, 2.85,
                "Q/R/M/C: a four-stage decomposition of semantic memory navigation",
                ha="center", va="top", fontsize=14, fontweight="bold", color="#222")

    # ── Bottom-left: example trajectory grid (water search) ─────────────
    ax_traj = fig.add_subplot(gs[1, 0:2])
    ax_traj.set_xlim(-0.7, 8.7)
    ax_traj.set_ylim(-1.8, 5.5)
    ax_traj.set_aspect("equal")
    ax_traj.axis("off")
    # Grid lines
    for i in range(9):
        ax_traj.plot([i - 0.5, i - 0.5], [-0.5, 5.5], color="#ddd", linewidth=0.5)
    for j in range(7):
        ax_traj.plot([-0.5, 8.5], [j - 0.5, j - 0.5], color="#ddd", linewidth=0.5)
    # Water targets
    waters = [(7, 4), (6, 1)]
    for (wx, wy) in waters:
        ax_traj.add_patch(Rectangle((wx - 0.4, wy - 0.4), 0.8, 0.8,
                                    facecolor="#4FC3F7", edgecolor="#0277BD",
                                    linewidth=1.2, alpha=0.85))
        ax_traj.text(wx, wy, "water", ha="center", va="center",
                     fontsize=6.5, color="#01579B", fontweight="bold")
    # Hazard
    haz = [(3, 2), (4, 2)]
    for (hx, hy) in haz:
        ax_traj.add_patch(Rectangle((hx - 0.4, hy - 0.4), 0.8, 0.8,
                                    facecolor="#FFCDD2", edgecolor="#C62828",
                                    linewidth=1.0, alpha=0.7))
    # Trajectory
    traj = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3),
            (5, 3), (5, 4), (6, 4), (7, 4)]
    tx = [p[0] for p in traj]
    ty = [p[1] for p in traj]
    ax_traj.plot(tx, ty, color=COLORS["accent"], linewidth=2.0, alpha=0.85, zorder=5)
    ax_traj.scatter(tx, ty, color=COLORS["accent"], s=18, zorder=6,
                    edgecolors="white", linewidth=0.8)
    # Start agent
    ax_traj.add_patch(Circle((0, 0), 0.30, facecolor="#FFD54F",
                             edgecolor="#F57F17", linewidth=1.4, zorder=7))
    ax_traj.text(0, 0, "S", ha="center", va="center", fontsize=9, fontweight="bold")
    # Stage annotations placed BELOW grid in a row to avoid overlap
    annot = [
        (0.7, "Q", "intent: water"),
        (3.0, "R", "candidate set"),
        (5.5, "M", "lock target"),
        (7.7, "C", "arrived"),
    ]
    annot_y = -1.0
    for (ax_, letter, lbl) in annot:
        col = STAGE_COLORS[letter]
        ax_traj.add_patch(Circle((ax_, annot_y), 0.22,
                                 facecolor=col, edgecolor="white",
                                 linewidth=1.4, zorder=8))
        ax_traj.text(ax_, annot_y, letter, ha="center", va="center",
                     color="white", fontsize=9.5, fontweight="bold", zorder=9)
        ax_traj.text(ax_, annot_y - 0.55, lbl, ha="center", va="top",
                     fontsize=8.0, color="#333", style="italic")
    ax_traj.set_title("(a) Example episode: stages anchor at observable events",
                      fontsize=11, loc="left", pad=6)

    # ── Bottom-right: per-stage bottleneck attribution bar chart ───────
    ax_bar = fig.add_subplot(gs[1, 2:4])
    tasks = ["Water\nsearch", "Safe\nrest", "Goal-region\n(FourRooms)",
             "Hazard rec.\n(LavaGapS7)"]
    # End-to-end success
    succ = [0.95, 0.93, 0.075, 0.39]
    # Weakest stage label
    weak = ["—\n(strong)", "—\n(strong)", "downstream\n(C after R)",
            "retrieval\n(R)"]
    weak_col = ["#999", "#999", STAGE_COLORS["C"], STAGE_COLORS["R"]]

    x = np.arange(len(tasks))
    ax_bar.bar(x, succ, color=[COLORS["tertiary"], COLORS["tertiary"],
                                COLORS["secondary"], COLORS["accent"]],
               edgecolor="black", linewidth=0.5, width=0.6, alpha=0.85)
    ax_bar.set_ylim(0, 1.32)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tasks, fontsize=9.5)
    ax_bar.set_ylabel("End-to-end success", fontsize=11)
    ax_bar.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for xi, (s, w, col) in enumerate(zip(succ, weak, weak_col)):
        ax_bar.text(xi, s + 0.04, f"{s:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax_bar.text(xi, 1.27, "weakest:", ha="center", va="top",
                    fontsize=7.5, color="#777")
        ax_bar.text(xi, 1.20, w, ha="center", va="top", fontsize=8.5,
                    color=col, fontweight="bold", linespacing=1.15)
    ax_bar.set_title("(b) Same scalar success hides distinct stage failures",
                     fontsize=11, loc="left", pad=6)

    out = os.path.join(FIG_DIR, "fig_hero_qrmc.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"saved {out}")


# ────────────────────────────────────────────────────────────────────────
# Fig 2 — Four multi-agent architectures
# ────────────────────────────────────────────────────────────────────────
def fig2_architectures():
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    for ax in axes:
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.6, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # 4 agents arranged on a circle
    n_agents = 4
    angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False) + np.pi / 4
    agent_xy = [(np.cos(a) * 1.0, np.sin(a) * 1.0) for a in angles]

    def draw_agent(ax, x, y, idx, color="#FFD54F", with_mem=False):
        ax.add_patch(Circle((x, y), 0.20,
                            facecolor=color, edgecolor="#F57F17",
                            linewidth=1.2, zorder=5))
        ax.text(x, y, f"A{idx+1}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=6)
        if with_mem:
            # Small memory glyph attached to agent
            mx = x + 0.32 * np.sign(x) if x != 0 else x + 0.32
            my = y + 0.20 * np.sign(y) if y != 0 else y - 0.32
            ax.add_patch(Rectangle((mx - 0.13, my - 0.10), 0.26, 0.20,
                                   facecolor="#E1F5FE", edgecolor="#0277BD",
                                   linewidth=0.8, zorder=4))
            ax.text(mx, my, r"$\mathcal{M}_%d$" % (idx + 1),
                    ha="center", va="center", fontsize=7, zorder=5)

    def draw_central_mem(ax, label, color="#E1F5FE", edgecolor="#0277BD",
                         size=0.30):
        ax.add_patch(Rectangle((-size, -size + 0.05), 2 * size, 2 * size,
                               facecolor=color, edgecolor=edgecolor,
                               linewidth=1.4, zorder=5))
        ax.text(0, 0.05, label, ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=6)

    # (a) Independent
    ax = axes[0]
    ax.set_title("(a) Independent", fontsize=12, pad=6)
    for i, (x, y) in enumerate(agent_xy):
        draw_agent(ax, x, y, i, with_mem=True)
    ax.text(0, -1.35, "no inter-agent\ncommunication",
            ha="center", va="top", fontsize=9, color="#555",
            style="italic", linespacing=1.2)

    # (b) Shared bus
    ax = axes[1]
    ax.set_title("(b) Shared bus", fontsize=12, pad=6)
    draw_central_mem(ax, r"$\mathcal{M}_{\text{shared}}$",
                     color="#FFE0B2", edgecolor="#E65100", size=0.32)
    for i, (x, y) in enumerate(agent_xy):
        draw_agent(ax, x, y, i)
        ax.plot([x, 0], [y, 0.05], color="#555", linewidth=1.2,
                linestyle="-", alpha=0.6, zorder=2)
    ax.text(0, -1.35, "single global memory,\nread/write each tick",
            ha="center", va="top", fontsize=9, color="#555",
            style="italic", linespacing=1.2)

    # (c) Central aggregator
    ax = axes[2]
    ax.set_title("(c) Central aggregator", fontsize=12, pad=6)
    draw_central_mem(ax, "HUB", color="#FFCDD2", edgecolor="#B71C1C", size=0.30)
    for i, (x, y) in enumerate(agent_xy):
        draw_agent(ax, x, y, i, with_mem=True)
        ax.plot([x, 0], [y, 0.05], color="#555", linewidth=1.2,
                linestyle="-", alpha=0.6, zorder=2)
    ax.text(0, -1.35, "aggregator merges\nper-agent memories",
            ha="center", va="top", fontsize=9, color="#555",
            style="italic", linespacing=1.2)

    # (d) Peer broadcast (K)
    ax = axes[3]
    ax.set_title("(d) Peer broadcast (cadence $K$)", fontsize=12, pad=6)
    for i, (x, y) in enumerate(agent_xy):
        draw_agent(ax, x, y, i, with_mem=True)
    # Pairwise edges
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            xi, yi = agent_xy[i]
            xj, yj = agent_xy[j]
            ax.plot([xi, xj], [yi, yj], color="#2ca02c", linewidth=1.1,
                    linestyle="--", alpha=0.65, zorder=2)
    # K annotation: place at top center to avoid edge clipping
    ax.text(0, 1.35, r"broadcast every $K$ ticks",
            ha="center", va="bottom", fontsize=9, color="#2ca02c",
            fontweight="bold")
    ax.text(0, -1.35, "peer-to-peer broadcast,\ndistributed memories",
            ha="center", va="top", fontsize=9, color="#555",
            style="italic", linespacing=1.2)

    fig.suptitle("Four multi-agent memory architectures",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "fig_architectures.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"saved {out}")


# ────────────────────────────────────────────────────────────────────────
# Fig 3 — Per-architecture conditional rates (replaces Table 1 visual)
# ────────────────────────────────────────────────────────────────────────
def fig3_conditional_rates():
    archs = ["indep", "shared\nbus", "central\nagg",
             "peer\nK=1", "peer\nK=2", "peer\nK=4", "peer\nK=8",
             "peer\nK=16", "peer\nK=32", "peer\nK=48", "peer\nK=64"]
    p_MR = [0.68, 1.00, 0.97, 0.97, 0.97, 0.92, 0.69, 0.67, 0.67, 0.68, 0.68]
    p_CM = [0.88, 0.60, 0.61, 0.60, 0.62, 0.66, 0.87, 0.89, 0.89, 0.88, 0.88]
    t_succ = [8.02, 8.33, 7.64, 7.80, 7.59, 8.00, 7.47, 7.87, 8.28, 8.79, 9.19]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.65], "hspace": 0.18},
    )

    x = np.arange(len(archs))
    w = 0.36

    ax1.bar(x - w / 2, p_MR, w, color=STAGE_COLORS["M"],
            label=r"$P(M^\star|R^\star)$", edgecolor="black",
            linewidth=0.5, alpha=0.88)
    ax1.bar(x + w / 2, p_CM, w, color=STAGE_COLORS["C"],
            label=r"$P(C^\star|M^\star)$", edgecolor="black",
            linewidth=0.5, alpha=0.88)
    for xi, (m, c) in enumerate(zip(p_MR, p_CM)):
        ax1.text(xi - w / 2, m + 0.015, f"{m:.2f}", ha="center", va="bottom",
                 fontsize=7.5, color=STAGE_COLORS["M"])
        ax1.text(xi + w / 2, c + 0.015, f"{c:.2f}", ha="center", va="bottom",
                 fontsize=7.5, color=STAGE_COLORS["C"])

    # Annotate the M/C trade-off regimes
    ax1.axhline(y=1.0, color="#bbb", linewidth=0.5, linestyle=":")
    ax1.set_ylim(0, 1.32)
    ax1.set_xlim(-0.6, 10.6)
    ax1.set_ylabel("Conditional rate", fontsize=11)
    ax1.legend(loc="lower right", ncol=2, frameon=True, fontsize=10)
    ax1.set_title(
        "Per-architecture stage decomposition "
        "(35,640-run sweep, $N{\\in}\\{3,5,8\\}$, 40 seeds)",
        fontsize=12, loc="left", pad=6,
    )

    # M-saturation/C-collapse vs M-deficit/C-saturation annotations
    ax1.axvspan(0.5, 5.5, alpha=0.04, color=STAGE_COLORS["M"], zorder=0)
    ax1.axvspan(5.5, 10.5, alpha=0.04, color=STAGE_COLORS["C"], zorder=0)
    ax1.text(3.0, 1.27, "M-saturation / C-collapse",
             ha="center", va="top", fontsize=9.5,
             color=STAGE_COLORS["M"], fontweight="bold")
    ax1.text(8.0, 1.27, "M-deficit / C-saturation",
             ha="center", va="top", fontsize=9.5,
             color=STAGE_COLORS["C"], fontweight="bold")

    # Bottom: mean time-to-success
    ax2.plot(x, t_succ, marker="o", color=COLORS["accent"],
             linewidth=1.8, markersize=7,
             markeredgecolor="black", markeredgewidth=0.4)
    # Highlight K=8 minimum
    k8_idx = archs.index("peer\nK=8")
    ax2.scatter([k8_idx], [t_succ[k8_idx]], s=180, facecolor="none",
                edgecolor=COLORS["accent"], linewidth=2.0, zorder=5)
    ax2.annotate(
        "interior optimum",
        xy=(k8_idx, t_succ[k8_idx]),
        xytext=(k8_idx + 1.3, t_succ[k8_idx] - 0.4),
        fontsize=9.5, color=COLORS["accent"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.2),
    )
    ax2.axhline(t_succ[0], color="#888", linewidth=1.0,
                linestyle="--", alpha=0.7,
                label=f"independent baseline ({t_succ[0]:.2f})")
    ax2.set_ylabel("Mean $t_{\\text{succ}}$ (ticks)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(archs, fontsize=9)
    ax2.legend(loc="upper left", fontsize=9, frameon=True)
    ax2.set_ylim(7.2, 9.5)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_conditional_rates.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"saved {out}")


# ────────────────────────────────────────────────────────────────────────
# Fig 9 — Oracle waterfall single-agent
# ────────────────────────────────────────────────────────────────────────
def fig9_oracle_waterfall():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Common waterfall style
    def waterfall(ax, regimes, vals, end_to_end, title, ylim, ylabel,
                  show_metric_panel=False):
        x = np.arange(len(regimes))
        deltas = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        bottoms = [0] + list(vals[:-1])
        colors_each = [COLORS["neutral"]] + [
            STAGE_COLORS["R"], STAGE_COLORS["M"], STAGE_COLORS["C"]
        ]
        bars = []
        for i, (b, d, col) in enumerate(zip(bottoms, deltas, colors_each)):
            bar = ax.bar(x[i], d, bottom=b, color=col,
                         edgecolor="black", linewidth=0.5, width=0.62,
                         alpha=0.88)
            bars.append(bar)
            # delta label
            if i == 0:
                ax.text(x[i], b + d / 2, f"{vals[i]:.2f}",
                        ha="center", va="center", fontsize=10.5,
                        fontweight="bold", color="white")
            else:
                sign = "+" if d >= 0 else ""
                ax.text(x[i], b + d + 0.02, f"{sign}{d:.2f}",
                        ha="center", va="bottom",
                        fontsize=10, color=col, fontweight="bold")
                # connector line
                ax.plot([x[i - 1] + 0.31, x[i] - 0.31],
                        [vals[i - 1], vals[i - 1]],
                        color="#888", linewidth=0.8, linestyle=":")

        ax.set_xticks(x)
        ax.set_xticklabels(regimes, fontsize=9.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11.5, loc="left", pad=6)

    # FourRooms (downstream-limited): end-to-end success stays flat at 0.075
    fr_regimes = ["base", "+oracle R", "+oracle M", "+oracle C"]
    fr_vals = [0.075, 0.080, 0.080, 0.080]
    waterfall(axes[0], fr_regimes, fr_vals, end_to_end=True,
              title="(a) FourRooms goal-rejoin — downstream-limited",
              ylim=(0, 0.5),
              ylabel="End-to-end success")
    axes[0].text(2.5, 0.42,
                 "Oracle R lifts semantic-path\n"
                 "completion to 0.08, but\n"
                 "end-to-end stays flat",
                 ha="center", va="top", fontsize=9.5,
                 color="#555", style="italic", linespacing=1.25,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#F5F5F5", edgecolor="#bbb",
                           linewidth=0.6))

    # LavaGapS7 (retrieval/materialization-limited): jumps with R/M, small with C
    lg_regimes = ["base", "+oracle C", "+oracle M", "+oracle R"]
    lg_vals = [0.39, 0.43, 0.59, 0.60]
    # Reuse waterfall but with reordered colors: first delta = C, then M, then R
    x = np.arange(len(lg_regimes))
    deltas = [lg_vals[0]] + [lg_vals[i] - lg_vals[i - 1] for i in range(1, 4)]
    bottoms = [0] + list(lg_vals[:-1])
    cols_lg = [COLORS["neutral"], STAGE_COLORS["C"],
               STAGE_COLORS["M"], STAGE_COLORS["R"]]
    for i, (b, d, col) in enumerate(zip(bottoms, deltas, cols_lg)):
        axes[1].bar(x[i], d, bottom=b, color=col,
                    edgecolor="black", linewidth=0.5, width=0.62, alpha=0.88)
        if i == 0:
            axes[1].text(x[i], b + d / 2, f"{lg_vals[i]:.2f}",
                         ha="center", va="center", fontsize=10.5,
                         fontweight="bold", color="white")
        else:
            sign = "+" if d >= 0 else ""
            axes[1].text(x[i], b + d + 0.012, f"{sign}{d:.2f}",
                         ha="center", va="bottom",
                         fontsize=10, color=col, fontweight="bold")
            axes[1].plot([x[i - 1] + 0.31, x[i] - 0.31],
                         [lg_vals[i - 1], lg_vals[i - 1]],
                         color="#888", linewidth=0.8, linestyle=":")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lg_regimes, fontsize=9.5)
    axes[1].set_ylim(0, 0.95)
    axes[1].set_ylabel("End-to-end success", fontsize=11)
    axes[1].set_title("(b) LavaGapS7 hazard recovery — retrieval/materialization-limited",
                      fontsize=11.5, loc="left", pad=6)
    axes[1].text(2.5, 0.88,
                 "Controller oracle barely helps (+0.04);\n"
                 "R and M oracles lift to 0.60",
                 ha="center", va="top", fontsize=9.5,
                 color="#555", style="italic", linespacing=1.25,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#F5F5F5", edgecolor="#bbb",
                           linewidth=0.6))

    # Shared legend for stage colors
    legend_handles = [
        mpatches.Patch(color=COLORS["neutral"], label="base"),
        mpatches.Patch(color=STAGE_COLORS["R"], label="+oracle R (retrieval)"),
        mpatches.Patch(color=STAGE_COLORS["M"], label="+oracle M (materialization)"),
        mpatches.Patch(color=STAGE_COLORS["C"], label="+oracle C (controller)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "Oracle stage interventions localise the bottleneck "
        "(single-agent, 10 seeds × 8 episodes)",
        fontsize=12.5, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_oracle_waterfall.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig1_hero_overview()
    fig2_architectures()
    fig3_conditional_rates()
    fig9_oracle_waterfall()


if __name__ == "__main__":
    main()
