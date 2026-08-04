"""Aggregate the Stage-8 30-seed unified benchmark into the main table
and the paired BU.1 verdict (songline_full <= best direct baseline).

Reads all bench30 shards, aligns policies on shared seeds, computes the
per-policy means and a paired two-sided sign test of songline_full vs
the best direct baseline (lowest mean team cost among the four), with a
world-block bootstrap CI on the paired difference. Prints a markdown
table and writes v30_verdict.json.

Usage::

    python scripts/aggregate_bench30.py tmp/cluster/song_grammar/bench30
"""

from __future__ import annotations

import glob
import json
import os
import sys
from math import comb, sqrt

POLICIES = ["independent", "songline_full", "decision_centric",
            "execution_path", "graph_memory", "learned_formation"]
DIRECT = ["decision_centric", "execution_path", "graph_memory",
          "learned_formation"]
LABEL = {"independent": "independent",
         "songline_full": "songline_full",
         "decision_centric": "decision_centric (DeMem)",
         "execution_path": "execution_path (Mage)",
         "graph_memory": "graph_memory (RIR)",
         "learned_formation": "learned_formation (Mem-α−)"}


def load(root):
    rows = {}
    for f in glob.glob(os.path.join(root, "bench_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.setdefault(r["policy"], {})[r["seed"]] = r
    return rows


def sign_p(wins, n):
    # two-sided sign test, ignoring ties
    k = min(wins, n - wins)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mean(xs):
    return sum(xs) / len(xs)


def boot_ci(diffs, seed=12345, iters=10000):
    # deterministic LCG bootstrap (no numpy dependency, reproducible)
    n = len(diffs)
    s = seed
    means = []
    for _ in range(iters):
        acc = 0.0
        for _ in range(n):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            acc += diffs[s % n]
        means.append(acc / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return lo, hi


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "tmp/cluster/song_grammar/bench30"
    rows = load(root)
    present = [p for p in POLICIES if p in rows]
    if "songline_full" not in present:
        print("no songline_full shards yet"); return
    # shared seed set across all present policies
    seedsets = [set(rows[p]) for p in present]
    shared = sorted(set.intersection(*seedsets))
    print(f"# 30-seed benchmark ({root})")
    print(f"shared seeds: {len(shared)}  ({shared[0]}..{shared[-1]})\n")

    table = {}
    for p in present:
        rs = [rows[p][s] for s in shared]
        table[p] = {
            "team": mean([r["team_cost"] for r in rs]),
            "succ": mean([r["success_first"] for r in rs]),
            "fo": mean([r["fail_open"] for r in rs]),
            "mem_kb": mean([r["memory_bits"] for r in rs]) / 8192.0,
            "wire_kb": mean([r["wire_bits"] for r in rs]) / 8192.0,
        }

    print("| Policy | team cost ↓ | success ↑ | fail-open ↓ | mem (KB) | wire (KB) |")
    print("|---|---:|---:|---:|---:|---:|")
    for p in present:
        t = table[p]
        print(f"| {LABEL[p]} | {t['team']:.1f} | {t['succ']:.2f} | "
              f"{t['fo']:.3f} | {t['mem_kb']:.1f} | {t['wire_kb']:.1f} |")

    # BU.1 paired verdict vs best direct baseline
    direct_present = [p for p in DIRECT if p in present]
    verdict = {}
    if direct_present:
        best = min(direct_present, key=lambda p: table[p]["team"])
        sf = [rows["songline_full"][s]["team_cost"] for s in shared]
        bb = [rows[best][s]["team_cost"] for s in shared]
        diffs = [b - a for a, b in zip(sf, bb)]  # >0 => songline better
        wins = sum(1 for d in diffs if d > 0)
        n_nontie = sum(1 for d in diffs if d != 0)
        p_val = sign_p(wins, n_nontie)
        lo, hi = boot_ci(diffs)
        red = 100 * (table[best]["team"] - table["songline_full"]["team"]) \
            / table[best]["team"]
        print(f"\n**BU.1 (30 seeds):** songline_full {table['songline_full']['team']:.1f} "
              f"vs best direct baseline {LABEL[best]} {table[best]['team']:.1f} "
              f"→ −{red:.1f}%, paired wins {wins}/{n_nontie}, sign-test p={p_val:.4g}, "
              f"bootstrap 95% CI on Δ(team) [{lo:.1f}, {hi:.1f}]")
        vs_indep = None
        if "independent" in present:
            ind = [rows["independent"][s]["team_cost"] for s in shared]
            di = [b - a for a, b in zip(sf, ind)]
            wi = sum(1 for d in di if d > 0)
            vs_indep = {"wins": wi, "n": sum(1 for d in di if d != 0),
                        "reduction_pct": 100 * (table["independent"]["team"]
                        - table["songline_full"]["team"])
                        / table["independent"]["team"]}
        verdict = {
            "experiment": "Stage 8 — 30-seed rerun of the unified equal-budget benchmark",
            "shared_seeds": len(shared), "seed_range": [shared[0], shared[-1]],
            "table": {p: table[p] for p in present},
            "best_direct_baseline": best,
            "BU1": {"songline_team": table["songline_full"]["team"],
                    "best_baseline_team": table[best]["team"],
                    "reduction_pct": red, "paired_wins": wins,
                    "n_nontie": n_nontie, "sign_test_p": p_val,
                    "bootstrap95_delta_team": [lo, hi],
                    "PASS": table["songline_full"]["team"] <= table[best]["team"]},
            "vs_independent": vs_indep,
        }
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "v30_verdict.json"), "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"\nWrote {root}/v30_verdict.json")


if __name__ == "__main__":
    main()
