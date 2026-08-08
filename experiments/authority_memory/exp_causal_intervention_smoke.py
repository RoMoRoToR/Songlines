"""Sprint 7 smoke test — does randomized-masking label collection
(``authority_memory.causal_utility``) actually recover a causal
effect it was never told, from noisy outcomes alone?

This is infrastructure validation, not the E5 evaluation itself
(``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/04_EXPERIMENTS.md`` §E5,
which compares a trained estimator against UE1/heuristics on held-out
data --- Sprint 8).  Sprint 7's job
(``07_ROADMAP_SPRINTS.md``: "Собрать utility labels через randomized
memory masking... (s, m, r, y, Z) tuples") is only to build and check
the COLLECTION mechanism.  This script is the check.

Reuses E4's exact scenario (``corruption_kit.ROLE_UTILITY``: route_A
is beneficial to scout/carrier/fast, harmful to fragile) so the two
sprints connect directly: E4 (Sprint 6) used these numbers as HAND-
SPECIFIED ground truth to test the admission gate; this script tests
whether the SAME numbers can be recovered empirically, from noisy
per-decision outcomes, without ever being told them --- exactly the
gap Sprint 8's real estimator will need to close on a substrate where
no one can just look the numbers up.

Task (synthetic, deliberately noisy --- a single decision must NOT
reveal the effect; only averaging over many randomized decisions
does, which is the entire point of the "no replay needed" claim):

    y = BASE_REWARD - COST_PER_DISTANCE * distance_to_goal
        + Z * ROLE_UTILITY[role] + Gaussian noise (sigma=1.0)

``distance_to_goal`` is a nuisance covariate: it moves the outcome
but not the treatment effect, and Z is randomized independent of it,
so it adds variance without introducing bias --- a basic sanity
property any real randomized design must have.

Registered predictions (written to disk BEFORE any label is
collected):
  E_causal.sign_accuracy   -- at the largest N tested, sign(tau_hat)
                              matches sign(ROLE_UTILITY) for all 4
                              roles.
  E_causal.within_3se       -- at the largest N, |tau_hat - true tau|
                              < 3 * standard_error, for all 4 roles.
  E_causal.se_shrinks_with_n -- for every role, standard_error at the
                              largest N tested is smaller than at the
                              smallest.

Usage::

    PYTHONPATH=. .venv/bin/python \\
        experiments/authority_memory/exp_causal_intervention_smoke.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from authority_memory.causal_utility import (collect_labels, empirical_tau,
                                             standard_error)
from experiments.authority_memory.corruption_kit import ROLE_UTILITY
from experiments.authority_memory.route_a_task import (CERTIFICATE_ID,
                                                       NOISE_SIGMA,
                                                       make_decision_fn,
                                                       make_states)

OUT_DIR = "tmp/authority_memory/e_causal_intervention_smoke"
N_SWEEP = (400, 2000, 8000, 32000)


def run_at_n(n_total: int, seed: int) -> Dict[str, Dict[str, float]]:
    states = make_states(n_total, seed=seed)
    labels = collect_labels(make_decision_fn(noise_seed=seed + 1), states,
                            CERTIFICATE_ID, seed=seed + 2)
    per_role = {}
    for role in ROLE_UTILITY:
        role_labels = [l for l in labels if l.role == role]
        tau_hat, n1, n0 = empirical_tau(role_labels)
        se = standard_error(role_labels)
        per_role[role] = {"tau_hat": tau_hat, "se": se, "n1": n1, "n0": n0,
                          "true_tau": ROLE_UTILITY[role]}
    return per_role


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e_causal_registered.json"), "w") as f:
        json.dump({
            "E_causal.sign_accuracy": {"at_n": max(N_SWEEP)},
            "E_causal.within_3se": {"at_n": max(N_SWEEP), "sigma_band": 3},
            "E_causal.se_shrinks_with_n": {"n_sweep": list(N_SWEEP)},
            "role_utility": ROLE_UTILITY, "noise_sigma": NOISE_SIGMA,
        }, f, indent=2)

    results = {n: run_at_n(n, seed=n) for n in N_SWEEP}

    max_n = max(N_SWEEP)
    min_n = min(N_SWEEP)
    sign_accuracy_ok = all(
        (results[max_n][role]["tau_hat"] > 0) == (ROLE_UTILITY[role] > 0)
        for role in ROLE_UTILITY)
    within_3se_ok = all(
        abs(results[max_n][role]["tau_hat"] - ROLE_UTILITY[role])
        < 3 * results[max_n][role]["se"]
        for role in ROLE_UTILITY)
    se_shrinks_ok = all(
        results[max_n][role]["se"] < results[min_n][role]["se"]
        for role in ROLE_UTILITY)

    verdict = {
        "E_causal.sign_accuracy": sign_accuracy_ok,
        "E_causal.within_3se": within_3se_ok,
        "E_causal.se_shrinks_with_n": se_shrinks_ok,
    }
    go_no_go = all(verdict.values())

    with open(os.path.join(args.out, "e_causal_results.json"), "w") as f:
        json.dump({"summary": results, "verdict": verdict,
                  "go_no_go": go_no_go}, f, indent=2)

    print(json.dumps(results, indent=2))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"  [{'GO' if go_no_go else 'NO-GO'}] Sprint 7 infrastructure "
         f"verdict")
    print(f"Saved: {args.out}/e_causal_results.json")


if __name__ == "__main__":
    main()
