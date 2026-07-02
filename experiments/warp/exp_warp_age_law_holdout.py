"""W2 hold-out — registered predictions on never-run cells (H-W3).

The discrete-time refinement of the distance law was formulated after
inspecting the original W2 grid.  This script is the out-of-sample
answer: six NEW cells (distances, trusts and cadences not present in
W2b) whose breakpoints are computed from the FIXED model and written to
disk BEFORE any episode runs.  The age grid is step 1 — finer than the
original step 2 — so a hit must be exact, not grid-padded.

Model (unchanged constants alpha=0.05, tau=0.30, conf=0.95, obs_radius=2):

    gate       = ln(trust*conf/tau) / alpha
    t_gate     = d - obs_radius - 1
    bp_exp     = gate - t_gate
    bp_prune   = 10*K - (largest broadcast tick <= t_gate)      (K <= 4)
    bp         = min(bp_exp, bp_prune);  emp = floor(bp)  (never if < 0)

Cell 5 is the sharpest: at K=2, trust=1.0, d=12 the PRUNING horizon
(bp=12) undercuts the exponential gate (bp=14.05) — the model must pick
the pruning bound, not the gate.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_age_law_holdout.py
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.warp.exp_warp_age_law import (
    OBS_RADIUS, predicted_age_max, run_traveler_episode,
)

OUT_DIR = "tmp/warp/w2_age_law"

# (k, trust, d) — none of these triples appear in the original W2b grid.
HOLDOUT_CELLS = [
    (8, 0.9, 9),
    (8, 0.7, 9),
    (8, 0.5, 15),   # predicted: never (bp < 0)
    (4, 1.0, 6),    # new cadence; pruning slack — gate must bind
    (2, 1.0, 12),   # pruning UNDERCUTS the gate — sharpest discriminator
    (2, 0.8, 8),    # pruning slack at same K — gate must bind
]


def predict(k: int, trust: float, d: int) -> dict:
    gate = predicted_age_max(trust)
    t_gate = max(0, d - OBS_RADIUS - 1)
    bp_exp = gate - t_gate
    entry = {"gate": round(gate, 2), "t_gate": t_gate,
             "bp_exponential": round(bp_exp, 2)}
    bp = bp_exp
    if k <= 4:
        t_bcast = (t_gate // k) * k
        bp_prune = 10 * k - t_bcast
        entry["bp_pruning"] = bp_prune
        entry["binding"] = "pruning" if bp_prune < bp_exp else "gate"
        bp = min(bp, bp_prune)
    else:
        entry["binding"] = "gate"
    entry["predicted_breakpoint"] = math.floor(bp) if bp >= 0 else -1
    return entry


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pred_path = os.path.join(OUT_DIR, "holdout_predictions.json")

    # ── register predictions BEFORE running anything ──────────────
    predictions = {
        f"k{k}|trust={trust}|d={d}": predict(k, trust, d)
        for (k, trust, d) in HOLDOUT_CELLS
    }
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print("Registered predictions (written before any episode):")
    for name, p in predictions.items():
        print(f"  {name:<22} bp_pred={p['predicted_breakpoint']:>3} "
              f"(gate={p['gate']}, binding={p['binding']})")

    # ── run the held-out episodes, age grid step 1 ────────────────
    print("\nRunning held-out episodes …")
    results = {}
    all_exact = True
    for (k, trust, d) in HOLDOUT_CELLS:
        name = f"k{k}|trust={trust}|d={d}"
        pred_bp = predictions[name]["predicted_breakpoint"]
        max_age = max(6, pred_bp + 6)
        succ_ages = []
        for a0 in range(0, max_age + 1):
            r = run_traveler_episode("csm", trust, a0, d, k=k)
            if r["completed"]:
                succ_ages.append(a0)
        emp_bp = max(succ_ages) if succ_ages else -1
        monotone = (succ_ages == list(range(0, emp_bp + 1))
                    if succ_ages else True)
        exact = emp_bp == pred_bp
        all_exact &= exact
        results[name] = {"empirical_breakpoint": emp_bp,
                         "predicted_breakpoint": pred_bp,
                         "exact_match": exact, "monotone_step": monotone}
        print(f"  {name:<22} emp={emp_bp:>3} pred={pred_bp:>3} "
              f"{'EXACT' if exact else 'MISS'} monotone={monotone}")

    with open(os.path.join(OUT_DIR, "holdout_results.json"), "w") as f:
        json.dump({"predictions": predictions, "results": results,
                   "all_exact": all_exact}, f, indent=2)

    print("\n" + "=" * 60)
    n_exact = sum(1 for v in results.values() if v["exact_match"])
    print(f"  [{'PASS' if all_exact else 'PARTIAL'}] hold-out: "
          f"{n_exact}/{len(results)} exact integer-breakpoint matches")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/holdout_{{predictions,results}}.json")


if __name__ == "__main__":
    main()
