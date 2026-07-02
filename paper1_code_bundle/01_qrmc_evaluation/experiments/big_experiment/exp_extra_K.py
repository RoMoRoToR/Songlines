"""Extend the cadence sweep with K in {32, 48, 64} to test the K=16 boundary.

Same full sweep (N x M x layout x hazard x seeds) but only peer architecture
and only the new K values, so it runs ~5x faster than the full sweep.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.big_experiment.config import (
    N_AGENTS_FULL, M_BY_N_FULL, LAYOUTS_FULL,
    HAZARD_DENSITIES_FULL, SEEDS_FULL, STEP_LIMIT_FULL,
)
from experiments.big_experiment.exp_cadence_phase import CSV_FIELDS
from experiments.big_experiment.runner import RunConfig, run_one_config


EXTRA_KS = [32, 48, 64]


def expand_extra_configs():
    configs = []
    for n in N_AGENTS_FULL:
        for m in M_BY_N_FULL[n]:
            if m > n:
                continue
            for layout in LAYOUTS_FULL:
                for k in EXTRA_KS:
                    for h in HAZARD_DENSITIES_FULL:
                        for s in SEEDS_FULL:
                            configs.append(RunConfig(
                                n_agents=n, n_waters=m, layout=layout,
                                architecture="peer", broadcast_every_k=k,
                                hazard_density=h, seed=s,
                                step_limit=STEP_LIMIT_FULL,
                            ))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/big_experiment_extraK")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    configs = expand_extra_configs()
    print(f"Extra-K sweep: {len(configs)} configs, workers={args.workers}, K in {EXTRA_KS}")
    csv_path = os.path.join(args.out_dir, "runs.csv")
    t0 = time.time()
    n_done = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one_config, c): c for c in configs}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                    writer.writerow({k: row.get(k) for k in CSV_FIELDS})
                except Exception as e:
                    print(f"  FAILED: {e}", file=sys.stderr)
                n_done += 1
                if n_done % 1000 == 0:
                    print(f"  {n_done}/{len(configs)}  elapsed={time.time()-t0:.1f}s")
    print(f"✓ Done {n_done}/{len(configs)} in {time.time()-t0:.1f}s  CSV→{csv_path}")


if __name__ == "__main__":
    main()
