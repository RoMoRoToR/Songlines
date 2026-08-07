"""Package A / Part 1 — coordination baselines vs Warp Drive.

Reviewer claim: "reservation is an obvious engineering patch — compare
with proper coordination alternatives."  Seven arms run on the SAME
benchmark as the Warp Drive evaluation (exp_warp_drive part C): scarcity
cells (N,M) ∈ {(3,2),(5,3),(8,5)} × layouts {random, asymmetric} ×
cadence K ∈ {1,2,4} × identical seeds across every arm.  All arms see
identical information (broadcast-cadence announcements only).

═══════════════════════════════════════════════════════════════════════
REGISTERED EXPECTATIONS — formulated BEFORE the first smoke run
(2026-08-07).  Independent armed comparisons on identical seeds; these
are predictions, not post-hoc fits.

  E1  Any coordination arm ≥ `none` on p(C*|M*) and success at K∈{1,2}
      (contention is the principal cost at fast cadence; every
      deconfliction mechanism should recover some of it).

  E2  Reservation-style arms with a stable priority (`wd`, `nearest`,
      `greedy`, `random`) reduce duplicate-commitment ticks by ≥ 50%
      vs `none`; `wd` and `nearest` are statistically indistinguishable
      or `wd` slightly better (its priority = earlier-lock-then-nearer
      is a superset of nearest-wins), and both ≥ `random` on t_cens
      (distance-blind coin retracts the nearly-arrived agent half the
      time → wasted travel).

  E3  `backoff` (no reservations) resolves conflicts but with MORE
      residual duplicate ticks and worse t_cens than every
      reservation-style arm: both colliders retract, and re-collisions
      recur until random desync + physical arrival.  It should still
      beat `none` on duplicates.

  E4  `occupancy` (soft penalty) lands between `none` and the hard
      arms on duplicates; it should never hurt success vs `none`.

  E5  Message accounting: all arms send announcements at the same
      cadence, so message COUNTS are comparable (wd adds release
      markers); wd pays ~1.2–2x the BYTES of the 3-byte arms
      (6 B reservations carrying reserved_at + distance).

  E6  No baseline arm beats `wd` on the primary pair
      (success, t_cens) by more than noise at any K — i.e. the
      reservation protocol is at least as good as the standard
      alternatives at equal information.  If `greedy` or `nearest`
      matches `wd`, that is a REPORTABLE result, not a failure: the
      claim under test is that coordination (any) removes the
      contention cost, and that wd is not dominated.
═══════════════════════════════════════════════════════════════════════

Usage::

    PYTHONPATH=. .venv/bin/python experiments/coordination/exp_coord_baselines.py \\
        --seeds 2 --workers 8 --out_dir tmp/coord_baselines_smoke      # smoke
    PYTHONPATH=. .venv/bin/python experiments/coordination/exp_coord_baselines.py \\
        --seeds 20 --workers 16 --out_dir tmp/cluster/coord_baselines  # full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.big_experiment.runner import RunConfig
from experiments.coordination.coord_protocols import ARMS
from experiments.coordination.coord_runner import run_coord_episode

NM_CELLS = [(3, 2), (5, 3), (8, 5)]      # same as exp_warp_drive part C
LAYOUTS = ["random", "asymmetric"]
KS = [1, 2, 4]
HAZARD = 0.05
STEP_LIMIT = 120

CSV_FIELDS = [
    "arm", "n_agents", "n_waters", "layout", "broadcast_every_k", "seed",
    "success_rate", "n_succeeded", "mean_t_succ", "t_cens", "ticks_played",
    "m_star_rate", "c_star_rate", "p_C_given_M",
    "warp_share_soft", "p_C_given_W_soft", "n_success_no_lock",
    "dup_lock_ticks", "dup_lock_events",
    "coord_n_retractions", "coord_mean_rollback_latency",
    "coord_n_waves", "coord_n_msgs", "coord_n_bytes", "coord_n_deliveries",
    "scarcity", "hazard_density",
]


def _one_job(job: Tuple) -> Dict[str, Any]:
    (n, m, layout, k, seed, arm) = job
    cfg = RunConfig(
        n_agents=n, n_waters=m, layout=layout, architecture="peer",
        broadcast_every_k=k, hazard_density=HAZARD, seed=seed,
        step_limit=STEP_LIMIT,
    )
    metrics, _ = run_coord_episode(cfg, arm)
    return metrics


def _mean_ci(vals: List[float], n_boot: int = 2000) -> Tuple[float, float, float]:
    vals = [v for v in vals
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return (float("nan"),) * 3
    rng = np.random.default_rng(7)
    arr = np.array(vals, dtype=float)
    boots = [float(np.mean(rng.choice(arr, len(arr)))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def summarize(rows: List[Dict[str, Any]], arms: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in KS:
        for arm in arms:
            cell = [r for r in rows
                    if r["broadcast_every_k"] == k and r["arm"] == arm]
            if not cell:
                continue
            out[f"k{k}/{arm}"] = {
                "n": len(cell),
                "success": _mean_ci([r["success_rate"] for r in cell]),
                "t_cens": _mean_ci([r["t_cens"] for r in cell]),
                "p_C_given_M": _mean_ci([r["p_C_given_M"] for r in cell]),
                "dup_lock_ticks": _mean_ci([r["dup_lock_ticks"] for r in cell]),
                "dup_lock_events": _mean_ci([r["dup_lock_events"] for r in cell]),
                "retractions": _mean_ci([r["coord_n_retractions"] for r in cell]),
                "msgs": _mean_ci([r["coord_n_msgs"] for r in cell]),
                "bytes": _mean_ci([r["coord_n_bytes"] for r in cell]),
            }
    return out


def print_table(summary: Dict[str, Any], arms: List[str]) -> None:
    hdr = (f"{'K':>2} {'arm':<10} {'succ':>6} {'t_cens':>7} {'p(C|M)':>7} "
           f"{'dup_tk':>7} {'dup_ev':>7} {'retr':>6} {'msgs':>7} {'bytes':>8}")
    print(hdr)
    print("-" * len(hdr))
    for k in KS:
        for arm in arms:
            s = summary.get(f"k{k}/{arm}")
            if s is None:
                continue
            print(f"{k:>2} {arm:<10} {s['success'][0]:>6.3f} "
                  f"{s['t_cens'][0]:>7.1f} {s['p_C_given_M'][0]:>7.3f} "
                  f"{s['dup_lock_ticks'][0]:>7.1f} "
                  f"{s['dup_lock_events'][0]:>7.2f} "
                  f"{s['retractions'][0]:>6.2f} {s['msgs'][0]:>7.1f} "
                  f"{s['bytes'][0]:>8.1f}")
        print("-" * len(hdr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--out_dir", default="tmp/coord_baselines_smoke")
    parser.add_argument("--arms", nargs="*", default=ARMS)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = [(n, m, layout, k, seed, arm)
            for (n, m) in NM_CELLS
            for layout in LAYOUTS
            for seed in range(args.seeds)      # identical seeds across arms
            for k in KS
            for arm in args.arms]
    print(f"Coordination baselines: {len(jobs)} episodes "
          f"({len(NM_CELLS)} cells x {len(LAYOUTS)} layouts x "
          f"{args.seeds} seeds x K{KS} x {len(args.arms)} arms), "
          f"workers={args.workers}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(jobs)}  elapsed={el:.1f}s "
                      f"({el / (i + 1):.2f}s/ep)")
    elapsed = time.time() - t0
    print(f"done: {len(rows)}/{len(jobs)} episodes in {elapsed:.1f}s "
          f"({elapsed / max(1, len(rows)):.2f}s/ep)\n")

    csv_path = os.path.join(args.out_dir, "coord_runs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})

    summary = summarize(rows, args.arms)
    print_table(summary, args.arms)

    with open(os.path.join(args.out_dir, "coord_summary.json"), "w") as f:
        json.dump({"summary": summary, "n_rows": len(rows),
                   "elapsed_s": elapsed, "arms": args.arms,
                   "seeds": args.seeds}, f, indent=2, default=str)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {os.path.join(args.out_dir, 'coord_summary.json')}")


if __name__ == "__main__":
    main()
