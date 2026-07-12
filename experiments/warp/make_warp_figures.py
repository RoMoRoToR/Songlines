"""Paper figures for the Semantic Warp skeleton.

Fig 1  fig_warp_strata     phi-stratified completion vs cadence K:
                           P(C*|M*,own) vs P(C*|W*) with warp share —
                           the C-collapse concentrates in the W*-stratum
                           at fast K.  (Also the rebuttal figure for the
                           main paper's bottleneck-shift.)
Fig 2  fig_warp_law        the warp distance law: (a) trust×staleness
                           gate with empirical closures; (b) predicted
                           vs empirical completion breakpoints, original
                           grid + registered hold-out on the diagonal.
Fig 3  fig_warp_drive      unconditional Warp Drive results by K:
                           success rate and censored time, base vs +WD.

Data sources: tmp/warp/{w1_gain,w2_age_law,w3_drive}/*.json plus a
small dedicated peer-K sweep (K ∈ {1,2,4,8,16}, scarcity cells, random
layout, 20 seeds) cached at tmp/warp/w1_gain/strata_byK.json.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/make_warp_figures.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "papers/figures"
STRATA_CACHE = "tmp/warp/w1_gain/strata_byK.json"
KS = [1, 2, 4, 8, 16]
NM_CELLS = [(3, 2), (5, 3), (8, 5)]
SEEDS = 20

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
})
C_OWN, C_WARP, C_SHARE = "#2c7fb8", "#d95f0e", "#7a7a7a"
C_BASE, C_WD = "#7a7a7a", "#1b9e77"


# ───────────────────────────────── strata-by-K sweep (Fig 1 data)


def _strata_job(job):
    from experiments.big_experiment.runner import RunConfig
    from experiments.warp.warp_runner import run_one_config_warp
    (n, m, k, seed) = job
    cfg = RunConfig(n_agents=n, n_waters=m, layout="random",
                    architecture="peer", broadcast_every_k=k,
                    hazard_density=0.05, seed=seed, step_limit=120)
    _, log = run_one_config_warp(cfg)
    return {"k": k, "tag": cfg.as_tag(),
            "events": [{"soft": e.w_star_soft, "completed": e.completed}
                       for e in log.m_star_events()]}


def strata_by_k(workers: int = 8) -> Dict[str, Any]:
    if os.path.exists(STRATA_CACHE):
        return json.load(open(STRATA_CACHE))
    jobs = [(n, m, k, s) for (n, m) in NM_CELLS for k in KS
            for s in range(SEEDS)]
    print(f"strata-by-K sweep: {len(jobs)} episodes …")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_strata_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    def boot(pairs, n_boot=4000, seed=0):
        """pairs: list of per-episode lists of 0/1 — cluster bootstrap."""
        pairs = [p for p in pairs if p]
        if not pairs:
            return (float("nan"),) * 3
        rng = np.random.default_rng(seed)
        point = np.mean([v for p in pairs for v in p])
        stats = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(pairs), len(pairs))
            pooled = [v for i in idx for v in pairs[i]]
            if pooled:
                stats.append(np.mean(pooled))
        lo, hi = np.percentile(stats, [2.5, 97.5])
        return float(point), float(lo), float(hi)

    out: Dict[str, Any] = {}
    for k in KS:
        krows = [r for r in rows if r["k"] == k]
        warp_c = [[int(e["completed"]) for e in r["events"] if e["soft"]]
                  for r in krows]
        own_c = [[int(e["completed"]) for e in r["events"] if not e["soft"]]
                 for r in krows]
        share = [[int(e["soft"]) for e in r["events"]] for r in krows]
        out[str(k)] = {
            "p_C_given_W": boot(warp_c, seed=k),
            "p_C_given_own": boot(own_c, seed=100 + k),
            "warp_share": boot(share, seed=200 + k),
            "n_events": sum(len(r["events"]) for r in krows),
        }
    os.makedirs(os.path.dirname(STRATA_CACHE), exist_ok=True)
    with open(STRATA_CACHE, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ───────────────────────────────── Fig 1: strata vs K


def fig_strata(data: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ks = np.array(KS, dtype=float)

    def series(key):
        m = np.array([data[str(k)][key][0] for k in KS])
        lo = np.array([data[str(k)][key][1] for k in KS])
        hi = np.array([data[str(k)][key][2] for k in KS])
        return m, m - lo, hi - m

    for key, color, label, marker in [
            ("p_C_given_own", C_OWN, r"$P(C^\star\,|\,M^\star,\ \mathrm{own})$", "o"),
            ("p_C_given_W", C_WARP, r"$P(C^\star\,|\,W^\star)$", "s")]:
        m, elo, ehi = series(key)
        ax.errorbar(ks, m, yerr=[elo, ehi], color=color, marker=marker,
                    ms=4, lw=1.5, capsize=2.5, label=label)

    m, elo, ehi = series("warp_share")
    ax2 = ax.twinx()
    ax2.errorbar(ks, m, yerr=[elo, ehi], color=C_SHARE, marker="^", ms=4,
                 lw=1.2, ls="--", capsize=2.5,
                 label=r"warp share $P(W^\star|M^\star)$")
    ax2.set_ylabel(r"warp share $P(W^\star|M^\star)$", color=C_SHARE)
    ax2.tick_params(axis="y", colors=C_SHARE)
    ax2.set_ylim(0, 0.65)

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel(r"broadcast cadence $K$ (peer, scarcity cells, random layout)")
    ax.set_ylabel("completion probability")
    ax.set_ylim(0, 0.62)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_warp_strata.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────── Fig 2: the distance law


def fig_law() -> None:
    w2 = json.load(open("tmp/warp/w2_age_law/w2_results.json"))
    ho = json.load(open("tmp/warp/w2_age_law/holdout_results.json"))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.0))

    # (a) trust×staleness gate
    ages = np.linspace(0, 30, 301)
    ALPHA, TAU, CONF = 0.05, 0.30, 0.95
    cmap = plt.get_cmap("viridis")
    trusts = [1.0, 0.8, 0.6, 0.4, 0.25]
    for i, t in enumerate(trusts):
        w = t * CONF * np.exp(-ALPHA * ages)
        color = cmap(i / (len(trusts) - 0.999))
        a.plot(ages, w, color=color, lw=1.5,
               label=rf"$\tau_i={t}$")
        emp = w2["w2a"][str(t)]["empirical_age_max"]
        if emp >= 0:
            a.plot([emp], [TAU], marker="x", ms=7, mew=2, color=color)
    a.axhline(TAU, color="k", lw=0.8, ls=":")
    a.text(29.5, TAU + 0.012, r"inclusion threshold $\tau$",
           ha="right", fontsize=8)
    # peer: no staleness term — flat weight, gate never closes
    a.plot(ages, np.full_like(ages, 0.7 * math.log(2)), color=C_SHARE,
           lw=1.5, ls="--")
    a.text(29.5, 0.7 * math.log(2) + 0.012, "fixed-$K$ peer: no gate",
           ha="right", fontsize=8, color=C_SHARE)
    a.set_xlabel("evidence age (ticks)")
    a.set_ylabel(r"merge weight  $\tau_i\, c\, e^{-\alpha\,\mathrm{age}}$")
    a.set_ylim(0, 1.0)
    a.legend(frameon=False, loc="upper right", ncol=2)
    a.set_title("(a) warp gate; $\\times$ = empirical closure")

    # (b) predicted vs empirical breakpoints
    orig_x, orig_y = [], []
    for name, v in w2["breakpoints"].items():
        if v.get("predicted_breakpoint_age") is None:
            continue
        orig_x.append(v["predicted_breakpoint_age"])
        orig_y.append(v["empirical_breakpoint_age"])
    hold_x = [v["predicted_breakpoint"] for v in ho["results"].values()]
    hold_y = [v["empirical_breakpoint"] for v in ho["results"].values()]

    lim = (-2.5, 23)
    b.plot(lim, lim, color="k", lw=0.8, ls=":")
    b.scatter(orig_x, orig_y, s=28, facecolor="none", edgecolor=C_OWN,
              label="original grid (11 cells)")
    b.scatter(hold_x, hold_y, s=55, marker="*", color=C_WARP,
              label="registered hold-out (6/6 exact)")
    b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel("predicted breakpoint age (ticks)")
    b.set_ylabel("empirical breakpoint age (ticks)")
    b.text(-1.7, -0.5, "never", fontsize=8, rotation=90, va="bottom")
    b.legend(frameon=False, loc="upper left")
    b.set_title("(b) completion breakpoints")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_warp_law.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────── Fig 3: Warp Drive unconditional


def fig_drive() -> None:
    d = json.load(open("tmp/warp/w3_drive/w3_results.json"))
    rows = d["rows"]
    STEP = 120

    def mean_ci(vals, seed=3):
        rng = np.random.default_rng(seed)
        arr = np.array(vals, dtype=float)
        boots = [np.mean(rng.choice(arr, len(arr))) for _ in range(4000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(np.mean(arr)), float(lo), float(hi)

    def t_cens(r):
        t = r["mean_t_succ"] if r["mean_t_succ"] == r["mean_t_succ"] else STEP
        return r["success_rate"] * t + (1 - r["success_rate"]) * STEP

    def get(k, wd, fn):
        return mean_ci([fn(r) for r in rows
                        if r["broadcast_every_k"] == k and r["with_wd"] == wd])

    ks = [1, 2, 4]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9))

    for ax, fn, ylabel, ref_label in [
            (a, lambda r: r["success_rate"], "success rate", None),
            (b, t_cens, r"censored time-to-success (fail $=120$)", None)]:
        x = np.arange(len(ks))
        for wd, color, label, off in [(False, C_BASE, "peer base", -0.16),
                                      (True, C_WD, "peer + Warp Drive", 0.16)]:
            m = [get(k, wd, fn) for k in ks]
            ax.bar(x + off, [v[0] for v in m], width=0.3, color=color,
                   label=label,
                   yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]],
                   capsize=2.5, error_kw={"lw": 1.0})
        ref = get(8, False, fn)
        ax.axhline(ref[0], color="#c0392b", lw=1.2, ls="--")
        ax.text(2.42, ref[0], r" $K{=}8$ base", color="#c0392b",
                fontsize=8, va="bottom" if ax is a else "top")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_ylabel(ylabel)
    a.set_ylim(0.5, 0.66)
    b.set_ylim(45, 62)
    a.legend(frameon=False, loc="upper left")
    a.set_title("(a) unconditional success rate")
    b.set_title("(b) censored time")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_warp_drive.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────── Fig 4: universality across substrates


def fig_universality() -> None:
    """Warp share vs cadence K on three substrates — one law, one shape."""
    grid = json.load(open(STRATA_CACHE))
    grid_pts = [(k, grid[str(k)]["warp_share"][0],
                 grid[str(k)]["warp_share"][1], grid[str(k)]["warp_share"][2])
                for k in KS]
    vmas = json.load(open("tmp/warp/w5_vmas/w5_results.json"))["strata"]
    vmas_pts = [(k, vmas[str(k)]["warp_share"][0],
                 vmas[str(k)]["warp_share"][1], vmas[str(k)]["warp_share"][2])
                for k in (1, 4, 16)]
    llm = json.load(open("tmp/warp/w6_llm_full/w6_results.json"))["verdict"][
        "warp_share"]
    llm_pts = [(2, float(llm["K2"])), (8, float(llm["K8"]))]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for pts, color, marker, label in [
            (grid_pts, "#2c7fb8", "o", "grid (symbolic, 300 ep.)"),
            (vmas_pts, "#d95f0e", "s", "VMAS (continuous, 240 ep.)")]:
        ks = [p[0] for p in pts]
        m = np.array([p[1] for p in pts])
        lo = np.array([p[2] for p in pts])
        hi = np.array([p[3] for p in pts])
        ax.errorbar(ks, m, yerr=[m - lo, hi - m], color=color, marker=marker,
                    ms=4.5, lw=1.5, capsize=2.5, label=label)
    ax.plot([p[0] for p in llm_pts], [p[1] for p in llm_pts],
            color="#1b9e77", marker="^", ms=6, lw=1.5, ls="--",
            label="LLM text-world (8 ep., no CI)")

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlabel(r"broadcast cadence $K$")
    ax.set_ylabel(r"warp share $P(W^\star\,|\,M^\star)$")
    ax.set_ylim(-0.02, 0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_warp_universality.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    data = strata_by_k()
    print("strata by K:")
    for k in KS:
        s = data[str(k)]
        print(f"  K={k:<3} share={s['warp_share'][0]:.3f} "
              f"P(C|W)={s['p_C_given_W'][0]:.3f} "
              f"P(C|own)={s['p_C_given_own'][0]:.3f} n={s['n_events']}")
    fig_strata(data)
    fig_law()
    fig_drive()
    try:
        fig_universality()
    except FileNotFoundError as e:
        print(f"  (skipping universality figure: {e})")
    print(f"figures written to {FIG_DIR}/fig_warp_"
          f"{{strata,law,drive,universality}}.{{pdf,png}}")


if __name__ == "__main__":
    main()
