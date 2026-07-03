"""W6-C — Warp Drive on the LLM substrate (closing the arc on language).

W6 showed raw sharing LOWERS success on the LLM substrate (1/3 vs the
2/3 independent baseline on identical layouts): warp collisions cost
more than the information is worth --- the same pathology as the grid.
This experiment attaches the same decentralised contract (reservation
riding the broadcast cadence + anti-M* rollback with backoff,
``warp_drive.WarpDriveProtocol``, unchanged) to the LLM collective.

Registered predictions (written before the runs):
  C1: pooled success rate with Warp Drive > without, at K=2.
  C2: rollback latency of retracted locks is finite and <= 2K.
  C3: warp-collision episodes do not increase under the protocol.

Same 4 layouts, K in {2, 8}; the no-drive arm re-runs from the warm
deterministic cache (verifying reproducibility for free).

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_llm_drive.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.llm_collective.llm_backend import OllamaBackend
from experiments.warp.exp_warp_llm_full import (
    KS, MultiAgentTextNav, make_layouts_a, run_episode_a,
)
from experiments.warp.warp_drive import WarpDriveProtocol

OUT_DIR = "tmp/warp/w6_llm_full"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "LLM_COLLECTIVE_MODEL", "llama3.1:latest"))
    ap.add_argument("--layouts", type=int, default=4)
    ap.add_argument("--step_limit", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "w6c_registered.json"), "w") as f:
        json.dump({
            "C1": "pooled success with WD > without at K=2",
            "C2": "rollback latency finite and <= 2K",
            "C3": "collision episodes do not increase under WD",
        }, f, indent=2)

    backend = OllamaBackend(model=a.model,
                            cache_dir=os.path.join(OUT_DIR, ".cache_llm"))
    print(f"W6-C: Warp Drive on LLM (model={a.model}, "
          f"{a.layouts} layouts × K∈{KS} × WD on/off)")

    rows: List[Dict[str, Any]] = []
    for li, layout in enumerate(make_layouts_a(a.layouts)):
        for k in KS:
            for with_wd in (False, True):
                env = MultiAgentTextNav(layout["starts"], layout["apples"],
                                        step_limit=a.step_limit)
                agent_ids = list(layout["starts"].keys())
                wd = (WarpDriveProtocol(agent_ids, broadcast_every_k=k)
                      if with_wd else None)
                r = run_episode_a(backend, env, k, a.step_limit,
                                  warp_drive=wd)
                r.update({"layout_id": li, "with_wd": with_wd,
                          "wd_stats": wd.stats() if wd else None})
                rows.append(r)
                lat = [e["rollback_latency"] for e in r["events"]
                       if e.get("retracted")]
                print(f"  L{li} K={k} {'WD ' if with_wd else 'base'}: "
                      f"succ={r['n_succeeded']}/3 locks={len(r['events'])} "
                      f"collision={r['collision']} "
                      f"retr={len(lat)} lat={lat}")

    def pooled(k: int, wd: bool, field: str) -> float:
        rs = [r for r in rows if r["k"] == k and r["with_wd"] == wd]
        return sum(r[field] for r in rs) / (3 * len(rs))

    succ = {(k, wd): pooled(k, wd, "n_succeeded")
            for k in KS for wd in (False, True)}
    all_lat = [e["rollback_latency"] for r in rows if r["with_wd"]
               for e in r["events"] if e.get("retracted")]
    coll = {wd: sum(1 for r in rows if r["with_wd"] == wd and r["collision"])
            for wd in (False, True)}

    c1 = succ[(2, True)] > succ[(2, False)]
    c2 = bool(all_lat) is False or all(
        lat <= 2 * k for r in rows if r["with_wd"]
        for k in [r["k"]] for e in r["events"] if e.get("retracted")
        for lat in [e["rollback_latency"]])
    c3 = coll[True] <= coll[False]

    verdict = {
        "C1_success_up_at_K2": c1,
        "C2_rollback_latency_finite": c2,
        "C3_collisions_not_up": c3,
        "success_pooled": {f"K{k}_{'wd' if wd else 'base'}":
                           round(succ[(k, wd)], 3)
                           for k in KS for wd in (False, True)},
        "collision_episodes": coll,
        "rollback_latencies": all_lat,
    }
    with open(os.path.join(OUT_DIR, "w6c_results.json"), "w") as f:
        json.dump({"rows": rows, "verdict": verdict,
                   "llm_stats": backend.summary()}, f, indent=1,
                  default=str)

    print("=" * 60)
    for k, v in verdict.items():
        if isinstance(v, bool):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  success: {verdict['success_pooled']}")
    print(f"  collisions base/WD: {coll[False]}/{coll[True]}, "
          f"latencies: {all_lat}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/w6c_results.json")


if __name__ == "__main__":
    main()
