from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"


COLORS = {
    "qrmc": "#1b6ca8",
    "blue": "#4e79a7",
    "teal": "#2a9d8f",
    "green": "#59a14f",
    "orange": "#f28e2b",
    "red": "#d1495b",
    "purple": "#7b5ea7",
    "gray": "#8a8f98",
    "light_gray": "#d7dce2",
    "dark": "#222831",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e8ebef",
            "grid.linewidth": 0.65,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def bar_colors(labels: list[str], highlight: str = "Q/R/M/C") -> list[str]:
    return [COLORS["qrmc"] if label == highlight else COLORS["gray"] for label in labels]


def make_blind_repair_figure() -> None:
    baselines = load_json(PAPER_DIR / "artifacts" / "baselines_verdict.json")
    repair = load_json(PAPER_DIR / "artifacts" / "repair_verdict.json")

    diag_keys = [
        "success_only",
        "progress",
        "two_stage",
        "conformance",
        "learned_lr_leave_fault_out",
        "qrmc",
    ]
    diag_labels = ["Success", "Progress", "2-stage", "Conform.", "Learned", "Q/R/M/C"]
    diag_acc = [baselines["methods"][key]["accuracy"] for key in diag_keys]
    diag_f1 = [
        baselines["methods"][key].get("macro_f1", np.nan)
        for key in diag_keys
    ]

    repair_keys = [
        "random_repair",
        "progress",
        "conformance",
        "two_stage",
        "qrmc",
        "oracle_best",
    ]
    repair_labels = ["Random", "Progress", "Conform.", "2-stage", "Q/R/M/C", "Oracle"]
    gains = []
    regrets = []
    for key in repair_keys:
        if key in repair["per_method"]:
            entry = repair["per_method"][key]
        else:
            entry = repair[key]
        gains.append(entry["repair_gain"])
        regrets.append(entry["repair_regret"])

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.26), gridspec_kw={"wspace": 0.28})

    ax = axes[0]
    x = np.arange(len(diag_labels))
    ax.bar(x, diag_acc, color=bar_colors(diag_labels), width=0.68, label="Exact accuracy")
    ax.plot(x, diag_f1, color=COLORS["orange"], marker="o", linewidth=1.4, markersize=4.0, label="Macro-F1")
    ax.axhline(
        baselines["two_stage_side_level_accuracy"],
        color=COLORS["light_gray"],
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )
    ax.set_title("(a) Blinded fault localization")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(diag_labels, rotation=25, ha="right")
    ax.legend(loc="upper left", frameon=False, handlelength=1.5)
    ax.text(
        0.98,
        0.05,
        "Q/R/M/C subsets\n30/30 structural\n20/27 behavioral\n5/6 controls",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7.1,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d7dce2", "lw": 0.6},
    )

    ax = axes[1]
    x = np.arange(len(repair_labels))
    repair_colors = bar_colors(repair_labels)
    repair_colors[-1] = COLORS["green"]
    ax.bar(x, gains, color=repair_colors, width=0.68, label="Gain")
    ax.plot(x, regrets, color=COLORS["red"], marker="D", linewidth=1.25, markersize=3.6, label="Regret")
    ax.set_title("(b) Repair selected by diagnosis")
    ax.set_ylim(0, 0.39)
    ax.set_ylabel("Completion delta")
    ax.set_xticks(x)
    ax.set_xticklabels(repair_labels, rotation=25, ha="right")
    ax.legend(loc="upper left", frameon=False, handlelength=1.5)
    ax.text(
        0.98,
        0.05,
        "Q/R/M/C\nGain 0.293\nRegret 0.061",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7.1,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d7dce2", "lw": 0.6},
    )

    fig.savefig(FIG_DIR / "fig_blind_repair_tae.pdf")
    fig.savefig(FIG_DIR / "fig_blind_repair_tae.png")
    plt.close(fig)


def make_multiagent_figure() -> None:
    aggregates = load_json(ROOT / "tmp" / "paper1_clean_experiments_full" / "fig_aggregates.json")
    strata = load_json(ROOT / "tmp" / "warp" / "w1_gain" / "strata_byK.json")

    ks = [1, 2, 4, 8, 16, 32, 64]
    x = np.arange(len(ks))
    arms = aggregates["arms"]
    p_mr = np.array([arms[f"peer_K{k}"]["p_MR"][0] for k in ks])
    p_mr_lo = np.array([arms[f"peer_K{k}"]["p_MR"][1] for k in ks])
    p_mr_hi = np.array([arms[f"peer_K{k}"]["p_MR"][2] for k in ks])
    p_cm = np.array([arms[f"peer_K{k}"]["p_CM"][0] for k in ks])
    p_cm_lo = np.array([arms[f"peer_K{k}"]["p_CM"][1] for k in ks])
    p_cm_hi = np.array([arms[f"peer_K{k}"]["p_CM"][2] for k in ks])
    indep_mr = arms["indep"]["p_MR"][0]
    indep_cm = arms["indep"]["p_CM"][0]

    strata_ks = [1, 2, 4, 8, 16]
    sx = np.arange(len(strata_ks))
    peer_completion = np.array([strata[str(k)]["p_C_given_W"][0] for k in strata_ks])
    own_completion = np.array([strata[str(k)]["p_C_given_own"][0] for k in strata_ks])
    peer_share = np.array([strata[str(k)]["warp_share"][0] for k in strata_ks])

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.26), gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    ax.fill_between(x, p_mr_lo, p_mr_hi, color=COLORS["teal"], alpha=0.14, linewidth=0)
    ax.fill_between(x, p_cm_lo, p_cm_hi, color=COLORS["orange"], alpha=0.14, linewidth=0)
    ax.plot(x, p_mr, marker="o", color=COLORS["teal"], linewidth=1.7, label=r"$P(M^\star|R^\star)$")
    ax.plot(x, p_cm, marker="s", color=COLORS["orange"], linewidth=1.7, label=r"$P(C^\star|M^\star)$")
    ax.axhline(indep_mr, color=COLORS["teal"], linestyle=":", linewidth=1.1)
    ax.axhline(indep_cm, color=COLORS["orange"], linestyle=":", linewidth=1.1)
    ax.set_title("(a) Sharing shifts the bottleneck")
    ax.set_ylim(0.50, 1.02)
    ax.set_ylabel("Conditional rate")
    ax.set_xlabel("Peer broadcast cadence K")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend(loc="lower left", frameon=False, handlelength=1.7)
    ax.text(
        0.98,
        0.94,
        "dotted: independent memory",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=7.0,
        color=COLORS["dark"],
    )

    ax = axes[1]
    ax.plot(sx, own_completion, marker="o", color=COLORS["green"], linewidth=1.7, label="Own-evidence locks")
    ax.plot(sx, peer_completion, marker="s", color=COLORS["red"], linewidth=1.7, label="Peer-supported locks")
    ax.plot(sx, peer_share, marker="^", color=COLORS["purple"], linewidth=1.5, label="Peer-supported share")
    ax.set_title("(b) Provenance localizes C loss")
    ax.set_ylim(0.0, 0.62)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Peer broadcast cadence K")
    ax.set_xticks(sx)
    ax.set_xticklabels([str(k) for k in strata_ks])
    ax.legend(loc="upper right", frameon=False, handlelength=1.6)

    fig.savefig(FIG_DIR / "fig_multiagent_tae.pdf")
    fig.savefig(FIG_DIR / "fig_multiagent_tae.png")
    plt.close(fig)


def main() -> None:
    setup()
    make_blind_repair_figure()
    make_multiagent_figure()


if __name__ == "__main__":
    main()
