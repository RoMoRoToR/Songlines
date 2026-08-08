"""E5 — Utility Estimator: causal vs replay-calibrated
(``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/04_EXPERIMENTS.md`` §E5).

Two-part registration, exactly as specified:

  E5.1 (deterministic grid, replay available) -- does the causal
       (Sprint 7-8) estimator lose anything relative to an exact-
       replay oracle (the UE1-equivalent for this task,
       ``route_a_task.paired_replay_tau``) when replay-based ground
       truth IS available?
  E5.2 (stochastic substrate, replay NOT available) -- does the
       causal estimator meaningfully beat heuristic/LLM-rated
       importance baselines when it is NOT available (the substrate
       UE1 itself cannot be calibrated on)?

Both halves reuse the SAME trained estimator (one ``fit_estimator``
call on unpaired randomized labels from ``route_a_task`` --- Sprint
7's collection mechanism, Sprint 8's regression) and the SAME
ROLE_UTILITY ground truth E4 (Sprint 6) used by hand. Baselines for
E5.2:

  heuristic_importance -- frequency of Z=1 among the collected
      labels, per role.  In a balanced random design this is
      STRUCTURALLY incapable of being negative (it is a fraction in
      [0,1]) --- it can never express "net harmful", only "used more
      or less often". A frequency heuristic has no mechanism to
      detect harm; this is not a strawman, it is what frequency
      literally cannot represent.
  llm_rated_importance -- a fixed, data-independent prior guess
      (LLM_RATED_IMPORTANCE below), representing a plausible
      subjective judgement made without causal grounding.  Chosen
      once, honestly, before running anything; whether it happens to
      preserve rank order while still getting the sign wrong for
      fragile is measured, not engineered.

Registered predictions (written to disk BEFORE any label is
collected):
  E5.1.matches_oracle -- causal estimator's Spearman and sign
      accuracy against the paired-replay oracle both equal 1.0 (no
      loss relative to exact replay, given enough training labels).
  E5.2.causal_beats_baselines -- causal estimator's sign accuracy
      against ROLE_UTILITY is STRICTLY GREATER than both
      heuristic_importance's and llm_rated_importance's.
  E5.2.admission_precision_gap -- causal-driven admission precision
      (``authority_memory.admission``) is STRICTLY GREATER than both
      baselines' admission precision, using each estimator's own
      tau_hat as utility_lcb.

Usage::

    PYTHONPATH=. .venv/bin/python \\
        experiments/authority_memory/exp_e5_causal_utility_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scipy.stats import spearmanr

from authority_memory.admission import decide_admission
from authority_memory.causal_utility import (collect_labels, fit_estimator,
                                             predict_tau)
from experiments.authority_memory.corruption_kit import ROLE_UTILITY
from experiments.authority_memory.route_a_task import (CERTIFICATE_ID,
                                                        make_decision_fn,
                                                        make_states,
                                                        paired_replay_tau)

OUT_DIR = "tmp/authority_memory/e5_causal_utility_eval"
N_TRAIN = 8000
TAU_E = 0.6
TAU_U = 0.0
FIXED_EVIDENCE = 0.8   # comfortably >= TAU_E, isolates the utility gate

# Fixed, data-independent subjective prior (chosen once, before any
# computation below) --- see module docstring.
LLM_RATED_IMPORTANCE = {"scout": 0.7, "carrier": 0.5, "fragile": 0.3,
                        "fast": 0.6}


def train_estimator():
    states = make_states(N_TRAIN, seed=100)
    labels = collect_labels(make_decision_fn(noise_seed=101), states,
                            CERTIFICATE_ID, seed=102)
    return fit_estimator(labels, state_keys=("distance_to_goal",)), labels


def oracle_tau_per_role() -> Dict[str, float]:
    """Averaged over three different states/seeds per role, as an
    internal-consistency check that paired replay is state-invariant
    (it must be, by construction) --- not needed for precision (a
    single paired replay is already exact), only to demonstrate that
    property explicitly."""
    roles = list(ROLE_UTILITY)
    result = {}
    for role in roles:
        vals = [paired_replay_tau({"distance_to_goal": d}, role, seed)
                for d, seed in ((1.0, 1), (5.0, 2), (9.0, 3))]
        result[role] = sum(vals) / len(vals)
    return result


def heuristic_importance_per_role(labels) -> Dict[str, float]:
    result = {}
    for role in ROLE_UTILITY:
        role_labels = [l for l in labels if l.role == role]
        result[role] = sum(l.z for l in role_labels) / len(role_labels)
    return result


def sign_accuracy(estimate: Dict[str, float],
                  truth: Dict[str, float] = ROLE_UTILITY) -> float:
    roles = list(truth)
    correct = sum((estimate[r] > 0) == (truth[r] > 0) for r in roles)
    return correct / len(roles)


def spearman_against(estimate: Dict[str, float],
                     truth: Dict[str, float] = ROLE_UTILITY) -> float:
    roles = list(truth)
    rho, _ = spearmanr([truth[r] for r in roles],
                       [estimate[r] for r in roles])
    return float(rho)


def admission_precision_recall(estimate: Dict[str, float]
                               ) -> Dict[str, float]:
    """Uses each estimator's own tau_hat as utility_lcb --- the
    practical, decision-facing sibling of Spearman/sign accuracy
    (03_METRICS.md §2): does admitting on THIS estimate produce good
    real decisions, not just a well-ranked scoreboard."""
    tp = fp = fn = tn = 0
    for role in ROLE_UTILITY:
        admitted = decide_admission(FIXED_EVIDENCE, TAU_E, estimate[role],
                                    TAU_U, True)
        truly_useful = ROLE_UTILITY[role] > 0
        if admitted and truly_useful:
            tp += 1
        elif admitted and not truly_useful:
            fp += 1
        elif not admitted and truly_useful:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp,
           "fn": fn, "tn": tn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e5_registered.json"), "w") as f:
        json.dump({
            "E5.1.matches_oracle": {"spearman": 1.0, "sign_accuracy": 1.0},
            "E5.2.causal_beats_baselines": "causal sign_accuracy > "
                                          "heuristic AND > llm_rated",
            "E5.2.admission_precision_gap": "causal precision > "
                                            "heuristic AND > llm_rated",
            "n_train": N_TRAIN, "llm_rated_importance": LLM_RATED_IMPORTANCE,
            "role_utility": ROLE_UTILITY,
        }, f, indent=2)

    estimator, labels = train_estimator()
    causal_tau = {r: predict_tau(estimator, r) for r in ROLE_UTILITY}

    # ---- E5.1: deterministic grid / replay available ----------------
    oracle_tau = oracle_tau_per_role()
    e51 = {
        "oracle_tau": oracle_tau, "causal_tau": causal_tau,
        "spearman_causal_vs_oracle": spearman_against(causal_tau,
                                                       oracle_tau),
        "sign_accuracy_causal_vs_oracle": sign_accuracy(causal_tau,
                                                        oracle_tau),
    }

    # ---- E5.2: stochastic substrate / no replay ----------------------
    heuristic_tau = heuristic_importance_per_role(labels)
    e52 = {
        "heuristic_tau": heuristic_tau,
        "llm_rated_tau": LLM_RATED_IMPORTANCE,
        "causal_tau": causal_tau,
        "sign_accuracy": {
            "heuristic": sign_accuracy(heuristic_tau),
            "llm_rated": sign_accuracy(LLM_RATED_IMPORTANCE),
            "causal": sign_accuracy(causal_tau)},
        "spearman_vs_truth": {
            "heuristic": spearman_against(heuristic_tau),
            "llm_rated": spearman_against(LLM_RATED_IMPORTANCE),
            "causal": spearman_against(causal_tau)},
        "admission": {
            "heuristic": admission_precision_recall(heuristic_tau),
            "llm_rated": admission_precision_recall(LLM_RATED_IMPORTANCE),
            "causal": admission_precision_recall(causal_tau)},
    }

    e51_ok = (e51["spearman_causal_vs_oracle"] == 1.0
             and e51["sign_accuracy_causal_vs_oracle"] == 1.0)
    e52_beats_ok = (e52["sign_accuracy"]["causal"]
                   > e52["sign_accuracy"]["heuristic"]
                   and e52["sign_accuracy"]["causal"]
                   > e52["sign_accuracy"]["llm_rated"])
    e52_precision_ok = (e52["admission"]["causal"]["precision"]
                       > e52["admission"]["heuristic"]["precision"]
                       and e52["admission"]["causal"]["precision"]
                       > e52["admission"]["llm_rated"]["precision"])

    verdict = {
        "E5.1.matches_oracle": e51_ok,
        "E5.2.causal_beats_baselines": e52_beats_ok,
        "E5.2.admission_precision_gap": e52_precision_ok,
    }
    go_no_go = all(verdict.values())

    with open(os.path.join(args.out, "e5_results.json"), "w") as f:
        json.dump({"e5_1": e51, "e5_2": e52, "verdict": verdict,
                  "go_no_go": go_no_go}, f, indent=2)

    print(json.dumps({"e5_1": e51, "e5_2": e52}, indent=2))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"  [{'GO' if go_no_go else 'NO-GO'}] Sprint 8 verdict (E5)")
    print(f"Saved: {args.out}/e5_results.json")


if __name__ == "__main__":
    main()
