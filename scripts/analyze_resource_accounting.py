"""Aggregate Package-I resource accounting into Pareto tables + figure.

Reads the per-run jsonl shards written by
experiments/song_grammar/exp_resource_accounting.py, and writes into
the same directory:

  resources.json  -- raw runs + per-policy means + Pareto verdict
  resources.csv   -- flat per-run table (growth curve dropped)
  pareto.md       -- markdown Pareto table (paper-ready)
  resources_pareto.png  -- small-multiples scatter, one panel per
                           resource axis, arms as labeled points
                           (needs matplotlib; skipped with a note if
                           absent or --no-fig)

Pareto rule: utility axis is team_cost (lower better, straight from
exp_b_unified's own row); resource axes are stored bytes, transmitted
bytes, update time, query latency, CPU-seconds.  Arm A dominates B if
A is <= B on team cost AND on every resource, strictly < somewhere.
Frontier membership is reported per arm; no scalarization.

Usage::

    PYTHONPATH=. python scripts/analyze_resource_accounting.py \
        --dir tmp/resource_accounting_smoke
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

UTILITY_COL = "team_cost"          # lower is better
RESOURCE_COLS = [
    ("stored_bytes_final_mean", "stored B/agent"),
    ("transmitted_bytes", "tx B"),
    ("update_time_s", "update s"),
    ("query_latency_ms_mean", "query ms"),
    ("cpu_time_s", "CPU s"),
    ("amortized_formation_ms_per_use", "amort ms/use"),
]
EXTRA_COLS = ["success_first", "fail_open", "llm_calls",
              "memory_bits", "wire_bits", "resv_bits_est",
              "n_entries_final_total", "wall_time_s"]
# fixed arm order + validated categorical palette (dataviz skill,
# light mode; every point is also direct-labeled, so identity is
# never color-alone)
ARM_ORDER = ["independent", "songline_full", "decision_centric",
             "execution_path", "graph_memory", "learned_formation"]
ARM_COLOR = {"independent": "#2a78d6", "songline_full": "#008300",
             "decision_centric": "#e87ba4", "execution_path": "#eda100",
             "graph_memory": "#1baf7a", "learned_formation": "#eb6834"}
ARM_MARKER = {"independent": "o", "songline_full": "D",
              "decision_centric": "s", "execution_path": "^",
              "graph_memory": "v", "learned_formation": "P"}


def load_runs(d: str) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(d, "resources_*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
    return runs


def per_policy_means(runs) -> Dict[str, Dict[str, float]]:
    by: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        by[r["policy"]].append(r)
    cols = ([UTILITY_COL] + [c for c, _ in RESOURCE_COLS] + EXTRA_COLS)
    out = {}
    for pol, rows in by.items():
        out[pol] = {"n_seeds": len(rows)}
        for c in cols:
            vals = [r[c] for r in rows if c in r]
            out[pol][c] = sum(vals) / len(vals) if vals else float("nan")
    return out


def pareto_frontier(means: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    axes = [UTILITY_COL] + [c for c, _ in RESOURCE_COLS]
    pols = [p for p in ARM_ORDER if p in means] + \
           [p for p in means if p not in ARM_ORDER]
    dominated_by: Dict[str, List[str]] = {p: [] for p in pols}
    for a in pols:
        for b in pols:
            if a == b:
                continue
            va = [means[a][c] for c in axes]
            vb = [means[b][c] for c in axes]
            # all axes lower-is-better
            if all(x <= y for x, y in zip(va, vb)) and \
                    any(x < y for x, y in zip(va, vb)):
                dominated_by[b].append(a)
    # 2D frontiers: utility vs EACH resource separately (the reviewer
    # question is per-resource, not the near-vacuous joint 7D verdict)
    front2d: Dict[str, List[str]] = {}
    for c, _ in RESOURCE_COLS:
        front = []
        for a in pols:
            dom = any(means[b][UTILITY_COL] <= means[a][UTILITY_COL]
                      and means[b][c] <= means[a][c]
                      and (means[b][UTILITY_COL] < means[a][UTILITY_COL]
                           or means[b][c] < means[a][c])
                      for b in pols if b != a)
            if not dom:
                front.append(a)
        front2d[c] = front
    return {"axes": axes,
            "frontier": [p for p in pols if not dominated_by[p]],
            "dominated_by": {p: d for p, d in dominated_by.items() if d},
            "frontier_2d": front2d}


def write_csv(runs, path: str) -> None:
    drop = {"stored_bytes_curve", "group_cost", "llm_calls_note"}
    keys: List[str] = []
    for r in runs:
        for k in r:
            if k not in drop and k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in keys})


def fmt(x: float) -> str:
    if x != x:                                    # nan
        return "-"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}"


def pareto_md(means, verdict) -> str:
    pols = [p for p in ARM_ORDER if p in means] + \
           [p for p in means if p not in ARM_ORDER]
    head = (["policy", "team cost v", "succ ^", "LLM calls"]
            + [lbl + " v" for _, lbl in RESOURCE_COLS]
            + ["Pareto"])
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    for p in pols:
        m = means[p]
        n2d = sum(p in v for v in verdict["frontier_2d"].values())
        mark = f"{n2d}/{len(RESOURCE_COLS)} 2D"
        if p not in verdict["frontier"]:
            mark += (" (7D-dominated by "
                     + ",".join(verdict["dominated_by"][p]) + ")")
        row = ([p, fmt(m[UTILITY_COL]), fmt(m["success_first"]),
                str(int(m["llm_calls"]))]
               + [("*" if p in verdict["frontier_2d"][c] else "")
                  + fmt(m[c]) for c, _ in RESOURCE_COLS] + [mark])
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("`*` = on the 2D Pareto frontier of (team cost x that "
                 "resource); the Pareto column counts these.")
    lines.append("v = lower is better, ^ = higher is better; means over "
                 f"{means[pols[0]]['n_seeds']} seeds. Frontier = not "
                 "dominated on (team cost + all six resource axes) "
                 "jointly. Reservation traffic is estimated "
                 "(resv_bits_est in resources.csv): the B-unified env "
                 "resolves contention without wire messages.")
    return "\n".join(lines)


def make_figure(means, verdict, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    pols = [p for p in ARM_ORDER if p in means] + \
           [p for p in means if p not in ARM_ORDER]
    n = len(RESOURCE_COLS)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow),
                            facecolor="#fcfcfb", sharey=True)
    axs = axs.ravel()
    for k, (col, lbl) in enumerate(RESOURCE_COLS):
        ax = axs[k]
        ax.set_facecolor("#fcfcfb")
        for p in pols:
            x, y = means[p][col], means[p][UTILITY_COL]
            on_front = p in verdict["frontier_2d"][col]
            ax.scatter([x], [y], s=64,
                       marker=ARM_MARKER.get(p, "o"),
                       c=ARM_COLOR.get(p, "#52514e"),
                       edgecolors="#0b0b0b" if on_front else "none",
                       linewidths=1.4 if on_front else 0.0,
                       zorder=3)
            ax.annotate(p.replace("_", " "), (x, y),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color="#52514e")
        vals = [means[p][col] for p in pols]
        if min(vals) > 0 and max(vals) / max(min(vals), 1e-12) > 50:
            ax.set_xscale("log")
        ax.set_xlabel(lbl + " (lower better)", fontsize=8,
                      color="#0b0b0b")
        if k % ncol == 0:
            ax.set_ylabel("team cost (lower better)", fontsize=8,
                          color="#0b0b0b")
        ax.grid(True, color="#e5e4e0", linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, colors="#52514e")
    for k in range(n, nrow * ncol):
        axs[k].set_visible(False)
    fig.suptitle("Resource accounting on the B-unified arms "
                 "(black outline = 2D Pareto frontier of that panel)",
                 fontsize=9, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tmp/resource_accounting_smoke")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    runs = load_runs(a.dir)
    if not runs:
        sys.exit(f"no resources_*.jsonl shards in {a.dir}")
    means = per_policy_means(runs)
    verdict = pareto_frontier(means)
    write_csv(runs, os.path.join(a.dir, "resources.csv"))
    with open(os.path.join(a.dir, "resources.json"), "w") as f:
        json.dump({"runs": runs, "per_policy_means": means,
                   "pareto": verdict,
                   "notes": {
                       "utility": "team_cost from exp_b_unified row, "
                                  "unmodified",
                       "llm_calls": "0 for every arm (deterministic; "
                                    "no LLM client imported)",
                       "reservations": "resv_bits_est is an estimate "
                                       "(uses x RESV_BITS); B-unified "
                                       "sends no reservation messages",
                   }}, f, indent=2)
    md = pareto_md(means, verdict)
    with open(os.path.join(a.dir, "pareto.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    if not a.no_fig:
        fig_path = os.path.join(a.dir, "resources_pareto.png")
        if make_figure(means, verdict, fig_path):
            print(f"figure: {fig_path}")
        else:
            print("matplotlib not available; figure skipped")
    print(f"wrote: {a.dir}/resources.json, resources.csv, pareto.md")


if __name__ == "__main__":
    main()
