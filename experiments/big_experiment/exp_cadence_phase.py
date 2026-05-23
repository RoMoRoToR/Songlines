"""Big experiment driver — parallel sweep with streaming CSV output.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/big_experiment/exp_cadence_phase.py \\
        --mode smoke --out_dir tmp/big_experiment_smoke
    PYTHONPATH=. .venv/bin/python experiments/big_experiment/exp_cadence_phase.py \\
        --mode full  --workers 8 --out_dir tmp/big_experiment_full
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.big_experiment.config import (
    full_configs, smoke_configs,
)
from experiments.big_experiment.runner import RunConfig, run_one_config


CSV_FIELDS = [
    "n_agents", "n_waters", "layout", "architecture",
    "broadcast_every_k", "hazard_density", "seed", "step_limit",
    "n_succeeded", "success_rate", "mean_t_succ", "p95_t_succ",
    "total_trail", "n_hazard_hits", "scarcity", "tag",
]


def _serialise_run(cfg: RunConfig) -> Dict:
    """Pickle-safe wrapper for the executor."""
    return run_one_config(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--out_dir", default="tmp/big_experiment_smoke")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--progress_every", type=int, default=50)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    configs = smoke_configs() if args.mode == "smoke" else full_configs()
    n_total = len(configs)
    print(f"Mode={args.mode}  workers={args.workers}  n_configs={n_total}")

    csv_path = os.path.join(args.out_dir, "runs.csv")
    t0 = time.time()
    n_done = 0
    n_failed = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_serialise_run, cfg): cfg for cfg in configs}
            for fut in as_completed(futures):
                cfg = futures[fut]
                try:
                    row = fut.result()
                    writer.writerow({k: row.get(k) for k in CSV_FIELDS})
                except Exception as e:
                    n_failed += 1
                    print(f"  FAILED {cfg.as_tag()}: {e}", file=sys.stderr)
                n_done += 1
                if n_done % args.progress_every == 0 or n_done == n_total:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (n_total - n_done) / rate if rate > 0 else float("inf")
                    print(f"  {n_done:>5}/{n_total}  "
                          f"elapsed={elapsed:.1f}s  rate={rate:.1f}/s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"✓ Done  n_done={n_done}  n_failed={n_failed}  "
          f"elapsed={elapsed:.1f}s  rate={n_done/elapsed:.1f}/s")
    print(f"  CSV → {csv_path}")


if __name__ == "__main__":
    main()
