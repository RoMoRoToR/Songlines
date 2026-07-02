"""Lock-chain analysis — 'warp as taxi' (post-hoc over W1 logs).

Resolves the apparent W1 tension: shared|random shows warp gain
+10.8 ticks while P(C*|W*) = 0.004.  Mechanism hypothesis: a W*-lock
fails (target claimed) but TRANSPORTS the agent into the right region,
where it self-confirms a water and completes under a subsequent OWN
lock.  phi is frozen at lock time (§3.1), so the completion is honestly
booked in the own-stratum — the warp's contribution is the ride.

Chain definition (per episode, per agent, events sorted by tick):
  warp-assisted own completion :=
      a completed lock with phi < θ_soft
      preceded by ≥1 soft-W* lock of the same agent (dropped earlier).

Transport effect := warp_radius of the first W* lock in the chain vs
the radius at which the completing own lock was made.

Reads tmp/warp/w1_gain/w1_rows.jsonl (full runs only, mask off).

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/analyze_warp_chains.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

ROWS = "tmp/warp/w1_gain/w1_rows.jsonl"
OUT = "tmp/warp/w1_gain/w1_chains.json"


def arch_key(r: Dict[str, Any]) -> str:
    k = r["broadcast_every_k"]
    return f"{r['architecture']}-k{k}"


def main() -> None:
    rows = [json.loads(line) for line in open(ROWS)]
    rows = [r for r in rows if not r["mask_foreign"]]

    stats: Dict[str, Dict[str, float]] = {}
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[arch_key(r)].append(r)
        if r["layout"] == "random":
            groups[arch_key(r) + "|random"].append(r)

    for name, rs in sorted(groups.items()):
        own_completions = 0
        warp_assisted = 0
        transport_pairs = []  # (radius of first W* in chain, radius of own lock)
        n_w_locks = 0
        for r in rs:
            per_agent: Dict[str, List[Dict]] = defaultdict(list)
            for e in r["events"]:
                per_agent[e["agent_id"]].append(e)
            for aid, evts in per_agent.items():
                evts.sort(key=lambda e: e["tick"])
                w_seen: List[Dict] = []
                for e in evts:
                    if e["w_star_soft"]:
                        n_w_locks += 1
                        w_seen.append(e)
                        continue
                    if e["completed"]:
                        own_completions += 1
                        prior_w = [w for w in w_seen if w["tick"] < e["tick"]]
                        if prior_w:
                            warp_assisted += 1
                            transport_pairs.append(
                                (prior_w[0]["warp_radius_cells"],
                                 e["warp_radius_cells"]))
        share = warp_assisted / own_completions if own_completions else float("nan")
        entry = {
            "own_completions": own_completions,
            "warp_assisted_own_completions": warp_assisted,
            "warp_assisted_share": round(share, 3) if share == share else None,
        }
        if transport_pairs:
            entry["mean_radius_first_W_lock"] = round(
                float(np.mean([p[0] for p in transport_pairs])), 1)
            entry["mean_radius_completing_own_lock"] = round(
                float(np.mean([p[1] for p in transport_pairs])), 1)
        stats[name] = entry

    with open(OUT, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"{'arm':<24} {'own C*':>7} {'warp-assisted':>14} {'share':>7} "
          f"{'r(W*)':>6} {'r(own)':>7}")
    for name, s in sorted(stats.items()):
        print(f"{name:<24} {s['own_completions']:>7} "
              f"{s['warp_assisted_own_completions']:>14} "
              f"{str(s['warp_assisted_share']):>7} "
              f"{str(s.get('mean_radius_first_W_lock', '-')):>6} "
              f"{str(s.get('mean_radius_completing_own_lock', '-')):>7}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
