"""Overview schematic of the Q/R/M/C protocol: the nested conditional
regime (with the first-failing-link diagnosis and the stage-level
repair menu) and the marginal regime for systems without a lock chain.

Writes papers/figures/fig_protocol_overview.{pdf,png}.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/qrmc_external/make_overview_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = "papers/figures"
BLUE, RED, GREY, GREEN = "#2c7fb8", "#c0392b", "#555555", "#1b9e77"


def box(ax, x, y, w, h, text, fc="#eaf2fa", ec=BLUE, fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, label=None, color=GREY, fs=8.6, dy=0.13):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.2))
    if label:
        ax.text((x0 + x1) / 2, y0 + dy, label, ha="center",
                va="bottom", fontsize=fs, color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.1)
    ax.axis("off")

    # ── Panel A: nested conditional regime ─────────────────────────
    ax.text(0.12, 5.86, "A.  Nested regime (explicit query--lock chain)",
            fontsize=9.6, fontweight="bold")
    names = [("$Q^\\star$\nquery\nissued"), ("$R^\\star$\nmemory\nanswers"),
             ("$M^\\star$\npublic\ncommitment"), ("$C^\\star$\ncommitment\nrealized")]
    xs = [1.62, 3.72, 5.82, 7.92]
    y, w, h = 4.62, 1.5, 1.0
    box(ax, 0.12, y, 1.0, h, "task", fc="#f2f2f2", ec=GREY)
    arrow(ax, 1.12, y + h / 2, xs[0], y + h / 2, "$q$", color=BLUE)
    labels = ["$q$", "$r$", "$m$", "$c$"]
    for i, (xi, nm) in enumerate(zip(xs, names)):
        box(ax, xi, y, w, h, nm)
        if i < 3:
            arrow(ax, xi + w, y + h / 2, xs[i + 1], y + h / 2,
                  labels[i + 1], color=BLUE)
    ax.text(5.6, 4.3,
            "nested by construction:  $C^\\star\\subseteq M^\\star\\subseteq"
            " R^\\star\\subseteq Q^\\star$"
            "  $\\Rightarrow$  $P(C^\\star)=q\\,r\\,m\\,c$",
            ha="center", fontsize=9, color=BLUE)

    # diagnosis -> repair row
    box(ax, 0.12, 3.0, 2.6, 0.82,
        "diagnosis:\nfirst failing link", fc="#fdeeee", ec=RED)
    arrow(ax, 2.72, 3.41, 3.42, 3.41, color=RED)
    box(ax, 3.42, 3.0, 6.46, 0.82,
        "stage-level repair menu:   Q: restore query channel  |  "
        "R: consolidate memory\nM: remove untrue records  |  "
        "C: relax execution (budget, tools)", fc="#fdeeee", ec=RED, fs=8.2)
    arrow(ax, 2.37, y - 0.06, 1.5, 3.88, color=RED)

    # ── Panel B: marginal regime ────────────────────────────────────
    ax.text(0.12, 2.34, "B.  Marginal regime (no lock chain: tool traces)",
            fontsize=9.6, fontweight="bold")
    y2, w2, h2 = 0.86, 1.5, 0.86
    box(ax, 0.12, y2, 1.3, h2, "tool\ntrace", fc="#f2f2f2", ec=GREY)
    # контейнер с четырьмя независимыми дескрипторами (без цепочки)
    ax.add_patch(FancyBboxPatch((2.6, y2 - 0.14), 7.25, h2 + 0.28,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                fc="none", ec=GREEN, lw=1.0,
                                linestyle=(0, (3, 2))))
    arrow(ax, 1.42, y2 + h2 / 2, 2.6, y2 + h2 / 2, color=GREEN)
    for i, xi in enumerate([2.85, 4.7, 6.55, 8.4]):
        nm = ["$f(Q^\\star)$", "$f(R^\\star)$", "$f(M^\\star)$",
              "$f(C^\\star)$"][i]
        box(ax, xi, y2, w2, h2, nm, fc="#eefaf2", ec=GREEN)
    ax.text(5.6, 0.44,
            "independent behavioral descriptors: events not nested "
            "($M^\\star$ may occur without $R^\\star$), no product claimed",
            ha="center", fontsize=8.6, color=GREEN)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG_DIR}/fig_protocol_overview.{ext}",
                    bbox_inches="tight", dpi=170)
    print(f"written: {FIG_DIR}/fig_protocol_overview.pdf/.png")


if __name__ == "__main__":
    main()
