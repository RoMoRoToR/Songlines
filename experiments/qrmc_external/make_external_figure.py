"""Figure for the external-validation section: stage profiles of three
third-party agent frameworks on the four engineered task variants.

Reads tmp/qrmc_external/summary_llama20.json; writes
papers/figures/fig_external_stages.{pdf,png}

Usage::

    PYTHONPATH=. .venv/bin/python experiments/qrmc_external/make_external_figure.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "papers/figures"
FRAMEWORKS = [("openai_sdk", "OpenAI SDK", "#2c7fb8"),
              ("langgraph", "LangGraph", "#1b9e77"),
              ("autogen", "AutoGen", "#d95f0e")]
VARIANTS = [("control", "control (clean)"),
            ("r_starved", "consolidation gap"),
            ("m_ambiguous", "stale ambiguity"),
            ("c_budget", "tight budget")]
STAGES = ["Q", "R", "M", "C"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
})


def wilson(p: float, n: int, z: float = 1.96):
    """Wilson 95% interval; returns (lo, hi)."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    s = json.load(open("tmp/qrmc_external/summary_llama20.json"))["summary"]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.6), sharey=True)

    for ax, (variant, title) in zip(axes, VARIANTS):
        x = np.arange(len(STAGES))
        for i, (fw, label, color) in enumerate(FRAMEWORKS):
            cell = s[f"{fw}|{variant}"]
            n = int(cell.get("n", 20))
            vals = [cell[st] for st in STAGES]
            cis = [wilson(v, n) for v in vals]
            yerr = np.array([[v - lo for v, (lo, _) in zip(vals, cis)],
                             [hi - v for v, (_, hi) in zip(vals, cis)]])
            ax.bar(x + (i - 1) * 0.27, vals, width=0.25, color=color,
                   yerr=yerr, error_kw=dict(lw=0.8, capsize=1.5,
                                            ecolor="#444"),
                   label=label if variant == "control" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"${st}^\\star$" for st in STAGES])
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        if variant == "r_starved":
            ax.annotate("R = 0.00\n(all stacks)", xy=(1, 0.02),
                        xytext=(1, 0.45), ha="center", fontsize=8,
                        color="#a33",
                        arrowprops=dict(arrowstyle="->", color="#a33"))
        if variant == "c_budget":
            ax.annotate("C = 0.00\n(all stacks)", xy=(3, 0.02),
                        xytext=(2.7, 0.45), ha="center", fontsize=8,
                        color="#a33",
                        arrowprops=dict(arrowstyle="->", color="#a33"))
    axes[0].set_ylabel("event frequency $f(E)$\n(n = 20, Wilson 95% CI)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_external_stages.{ext}"),
                    bbox_inches="tight")
    print(f"written: {FIG_DIR}/fig_external_stages.pdf/.png")


if __name__ == "__main__":
    main()
