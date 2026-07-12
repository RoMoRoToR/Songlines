"""Figures for the route-warp / semantic-identity paper.

Fig 1  fig_route_cliff     the cliff: place-transfer success collapses
                           at the first wall; below it, place evidence
                           is WORSE than blind exploration.
Fig 2  fig_route_hazard    the song protects: hazard hits by arm, and
                           the risk resuming after the predicted
                           rupture (R3 x R2 composition).
Fig 3  fig_route_identity  meaning-based identity: warp-lock precision
                           under SE(2) frames, and the fail-closed vs
                           fail-open asymmetry in unrecoverable worlds.

Data: tmp/warp/{r1_route_gain,r3_hazard,w9_rotation,
w7_semantic_identity}/*.json.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/make_route_figures.py
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

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
})
C_ROUTE, C_PLACE, C_BLIND = "#1b9e77", "#d95f0e", "#7a7a7a"
C_SEM, C_COO = "#1b9e77", "#c0392b"


def fig_cliff() -> None:
    d = json.load(open("tmp/warp/r1_route_gain/r1_results.json"))
    s = d["summary"]
    buckets = ["D1", "D15", "D2", "D4"]
    Ds = [s[b]["mean_D"] for b in buckets]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9))

    # (a) success step
    for key, color, marker, label in [
            ("succ_route", C_ROUTE, "o", "route (WHERE + HOW)"),
            ("succ_place", C_PLACE, "s", "place (WHERE only)"),
            ("succ_blind", C_BLIND, "^", "blind")]:
        vals = [s[bk][key] / s[bk]["n"] for bk in buckets]
        a.plot(Ds, vals, color=color, marker=marker, ms=5, lw=1.6,
               label=label)
    a.axvspan(1.0, 1.38, color="#fdd", alpha=0.5)
    a.text(1.19, 0.45, "first\nwall", ha="center", fontsize=8,
           color="#a33")
    a.set_xlabel("measured detour factor $D$")
    a.set_ylabel("success rate (20 seeds/bucket)")
    a.set_ylim(-0.05, 1.08)
    a.legend(frameon=False, loc="center right")
    a.set_title("(a) the cliff: place transfer dies at the first wall")

    # (b) place vs blind: helps -> actively harmful
    m = [s[bk]["gain_place_vs_blind"][0] for bk in buckets]
    lo = [s[bk]["gain_place_vs_blind"][1] for bk in buckets]
    hi = [s[bk]["gain_place_vs_blind"][2] for bk in buckets]
    colors = [C_ROUTE if v > 0 else C_PLACE for v in m]
    b.bar(range(len(buckets)), m,
          yerr=[np.array(m) - lo, np.array(hi) - np.array(m)],
          color=colors, capsize=3, width=0.55)
    b.axhline(0, color="k", lw=0.8)
    b.set_xticks(range(len(buckets)))
    b.set_xticklabels([f"D={v:.2f}" for v in Ds])
    b.set_ylabel("gain place-vs-blind (ticks)")
    b.set_title("(b) below the cliff, WHERE is worse than nothing")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_route_cliff.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


def fig_hazard() -> None:
    d = json.load(open("tmp/warp/r3_hazard/r3_results.json"))
    v = d["verdict"]
    place = v["hits"]["place"]
    blind = v["hits"]["blind"]
    post = v["hits_post_rupture"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                               gridspec_kw={"width_ratios": [1.2, 1]})

    arms = ["route\n(foreign safe path)", "place\n(WHERE only)", "blind"]
    m = [0.0, place[0], blind[0]]
    lo = [0.0, place[1], blind[1]]
    hi = [0.0, place[2], blind[2]]
    a.bar(arms, m, yerr=[np.array(m) - lo, np.array(hi) - np.array(m)],
          color=[C_ROUTE, C_PLACE, C_BLIND], capsize=3, width=0.55)
    a.set_ylabel("hazard hits per episode")
    a.text(0, 0.6, "0 in all\n20 cells", ha="center", fontsize=8,
           color=C_ROUTE)
    a.set_title("(a) safety travels with the route")

    phases = ["before rupture", "after rupture"]
    m2 = [0.0, post[0]]
    lo2 = [0.0, post[1]]
    hi2 = [0.0, post[2]]
    b.bar(phases, m2,
          yerr=[np.array(m2) - lo2, np.array(hi2) - np.array(m2)],
          color=[C_ROUTE, C_PLACE], capsize=3, width=0.5)
    b.set_ylabel("hazard hits")
    b.set_title("(b) the risk resumes at the\npredicted rupture (19/19 exact)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_route_hazard.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


def fig_identity() -> None:
    w9 = json.load(open("tmp/warp/w9_rotation/w9_results.json"))
    w7 = json.load(open("tmp/warp/w7_semantic_identity/w7_results.json"))
    prec = w9["verdict"]["precision"]

    # unrecoverable worlds: aliased (W7, h=0) and 180-symmetric (W9 P3)
    s_alias_locks = w7["summary"]["aliased_h0"]["semantic_locks"]  # count
    c_alias_wrong = w7["summary"]["aliased_h0"]["coordinate_nonzero_locks"]
    p3 = w9["p3_rows"]
    s_sym = np.mean([1.0 if r["locked"] else 0.0
                     for r in p3 if r["arm"] == "se2"])
    c_sym = np.mean([1.0 if (r["locked"] and not r["lock_is_true_water"])
                     else 0.0 for r in p3 if r["arm"] == "coordinate"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    x = np.arange(2)
    a.bar(x - 0.17, [prec["semantic"], 0], width=0.34, color=C_SEM,
          label="semantic identity")
    a.bar(x + 0.17, [prec["coordinate"], 0], width=0.34, color=C_COO,
          label="coordinate identity")
    a.bar([1 - 0.17], [8 / 8], width=0.34, color=C_SEM)
    a.set_xticks(x)
    a.set_xticklabels(["strict-$W^\\star$ lock precision\n(full stack, "
                       "random $SE(2)$ frames)",
                       "frame recovery\n($80/80$ exact, se2)"])
    a.set_ylim(0, 1.15)
    a.text(-0.17, prec["semantic"] + 0.03, "8/8", ha="center", fontsize=8)
    a.text(0.17, 0.03, "0/992\nphantom", ha="center", fontsize=7,
           color=C_COO)
    a.legend(frameon=False, loc="lower right")
    a.set_title("(a) locks point at real water")

    labels = ["aliased world\n(featureless band)",
              "$180^\\circ$-symmetric world\n(unrecoverable in principle)"]
    x = np.arange(2)
    b.bar(x - 0.17, [s_alias_locks, s_sym], width=0.34, color=C_SEM,
          label="semantic: lock rate")
    b.bar(x + 0.17, [c_alias_wrong, c_sym], width=0.34, color=C_COO,
          label="coordinate: wrong-lock rate")
    b.set_xticks(x)
    b.set_xticklabels(labels)
    b.set_ylim(0, 1.15)
    b.text(-0.17, 0.03, "fails\nclosed", ha="center", fontsize=7,
           color=C_SEM)
    b.text(0.17, 1.02, "fails open", ha="center", fontsize=7, color=C_COO)
    b.legend(frameon=False, loc="center left")
    b.set_title("(b) opposite failure directions")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_route_identity.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_cliff()
    fig_hazard()
    fig_identity()
    print(f"figures written to {FIG_DIR}/fig_route_"
          f"{{cliff,hazard,identity}}.{{pdf,png}}")


if __name__ == "__main__":
    main()
