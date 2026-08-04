#!/usr/bin/env python3
"""Stage 8 aggregator: harvest every song-grammar result artifact into
one long-format CSV so all paper tables draw from a single source.

Long format: experiment, method, seed, metric, value  (+ ablation,
noise, agents, episodes when present).  Reads both the seeded jsonl
shards (per-seed rows) and the *_results.json verdict summaries.

Usage:
    python3 scripts/aggregate_song_grammar.py \
        --roots tmp/song_grammar tmp/cluster/song_grammar \
        --out artifacts/song_grammar_long.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, Iterable, List


def _flat(prefix: str, obj: Any) -> Iterable[tuple]:
    """Flatten nested dict/number into (metric, value) pairs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(f"{prefix}.{k}" if prefix else str(k), v)
    elif isinstance(obj, (int, float, bool)):
        yield prefix, float(obj)


def rows_from_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    exp = os.path.basename(path).split("_")[0]
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        base = {"experiment": exp,
                "method": r.get("config") or r.get("policy")
                or r.get("arm") or exp,
                "seed": r.get("seed", ""),
                "noise": r.get("noise", ""),
                "agents": r.get("agents", ""),
                "episodes": r.get("episodes", "")}
        for metric, value in _flat("", {k: v for k, v in r.items()
                                        if k not in
                                        ("config", "policy", "arm",
                                         "seed", "noise", "agents",
                                         "episodes")}):
            yield {**base, "metric": metric, "value": value}


def rows_from_results(path: str) -> Iterable[Dict[str, Any]]:
    exp = os.path.basename(os.path.dirname(path)) or "?"
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return
    verdict = d.get("verdict", {k: v for k, v in d.items()
                                 if isinstance(v, bool)})
    for k, v in verdict.items():
        yield {"experiment": exp, "method": "verdict", "seed": "",
               "noise": "", "agents": "", "episodes": "",
               "metric": k, "value": 1.0 if v else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["tmp/song_grammar"])
    ap.add_argument("--out", default="artifacts/song_grammar_long.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    for root in a.roots:
        for f in sorted(glob.glob(f"{root}/**/*.jsonl", recursive=True)):
            all_rows.extend(rows_from_jsonl(f))
        for f in sorted(glob.glob(f"{root}/**/*results.json",
                                  recursive=True)):
            all_rows.extend(rows_from_results(f))
    cols = ["experiment", "method", "seed", "noise", "agents",
            "episodes", "metric", "value"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"wrote {len(all_rows)} rows -> {a.out} "
          f"({len({r['experiment'] for r in all_rows})} experiments)")


if __name__ == "__main__":
    main()
