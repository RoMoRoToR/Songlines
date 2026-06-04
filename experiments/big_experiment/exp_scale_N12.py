"""Scale-up sweep at N=12.

Same sweep design but only N=12, M in {6, 8, 10}, all 3 layouts, all 3 hazard
densities, all 8 cadences K in {1, 2, 4, 8, 16, 32, 48, 64}, and 40 seeds.

Total: 3 (M) * 3 (layout) * 3 (hazard) * 8 (K) * 40 (seeds) = 8640 runs.
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
    LAYOUTS_FULL, HAZARD_DENSITIES_FULL, SEEDS_FULL, STEP_LIMIT_FULL,
)
from experiments.big_experiment.exp_cadence_phase import CSV_FIELDS
from experiments.big_experiment.runner import RunConfig, run_one_config

# env_factory currently caps N at 8; we must extend it for N=12.
# Check before launching.
from experiments.big_experiment.env_factory import SPAWN_POSITIONS_8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/big_experiment_N12")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build config list
    configs = []
    for m in [6, 8, 10]:
        for layout in LAYOUTS_FULL:
            for arch_k in [("peer", k) for k in [1, 2, 4, 8, 16, 32, 48, 64]]:
                for h in HAZARD_DENSITIES_FULL:
                    for s in SEEDS_FULL:
                        configs.append(RunConfig(
                            n_agents=12, n_waters=m, layout=layout,
                            architecture=arch_k[0],
                            broadcast_every_k=arch_k[1],
                            hazard_density=h, seed=s,
                            step_limit=STEP_LIMIT_FULL,
                        ))
    print(f"N=12 sweep: {len(configs)} configs, workers={args.workers}")

    csv_path = os.path.join(args.out_dir, "runs.csv")
    t0 = time.time()
    n_done = 0
    n_failed = 0
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
                    n_failed += 1
                    if n_failed < 5:
                        print(f"  FAILED: {e}", file=sys.stderr)
                n_done += 1
                if n_done % 500 == 0:
                    print(f"  {n_done}/{len(configs)}  elapsed={time.time()-t0:.1f}s")
    print(f"Done {n_done}/{len(configs)} (failed={n_failed}) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
