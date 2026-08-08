"""E4 — Role-dependent Knowledge: the fourth experiment of the
Memory-as-Authority-Protocol frontier
(``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/04_EXPERIMENTS.md`` §E4).

Direct demonstration of the fourth go/no-go property
(``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/README.md`` §6.4):
authority is receiver-specific.  The SAME claim, with IDENTICAL
evidential admissibility (E), reaches ADMITTED for some roles and
stays PROVISIONAL (limited use only, no persistent-memory/planning
authority) for others --- because ``authority_memory.admission``
(Sprint 6) gates on causal utility PER ROLE, not on evidence alone.

This is the first experiment to exercise the real
PROVISIONAL -> ADMITTED transition (Sprint 5's E2 only exercised
QUARANTINED -> PROVISIONAL, evidence-gated; Sprint 4's E3 only
exercised the EXPIRED/REVOKED path).  Utility values are hand
specified (``corruption_kit.ROLE_UTILITY``) --- the ground truth a
learned estimator (Sprint 7-8) would eventually have to recover, not
something this experiment tries to learn.

Scenario: four roles (scout, carrier, fragile, fast --- the
predator-prey asymmetric-embodiment roster from the frozen UCSM
series) all receive the same claim "route_A is traversable" at the
SAME evidence_score.  Swept over evidence values (0.65, 0.8, 0.95,
all >= tau_E) to confirm the role/authority split is driven by
utility, not by which exact evidence value was used:

  - correct (role-gated) policy: apply_admission() with each role's
    true utility --- scout/carrier/fast reach ADMITTED (positive
    utility); fragile stays PROVISIONAL (negative utility, correctly
    withheld).
  - naive (role-blind) policy: force PROVISIONAL -> ADMITTED for
    EVERY role regardless of utility --- models an architecture that
    only checks evidence, exactly the failure the utility gate exists
    to prevent.

Regret(role, policy) = max(utility[role], 0) - realized(role, policy),
where realized = utility[role] if ADMITTED else 0.  The role-gated
policy has zero regret by construction (it always picks the better of
{act, don't act}); the naive policy incurs the FULL utility gap for
fragile (0.5) while matching the role-gated policy everywhere else.

Registered predictions (written to disk BEFORE any evidence value is
run):
  E4.same_evidence_diverging_authority -- for every evidence value
      tested, scout/carrier/fast reach ADMITTED and fragile stays
      PROVISIONAL under the role-gated policy, despite identical E.
  E4.naive_matches_on_beneficial_roles -- naive and role-gated
      policies agree (both ADMITTED) for scout/carrier/fast.
  E4.regret_gap_is_exact -- regret(fragile, naive) - regret(fragile,
      role_gated) == 0.5 exactly, for every evidence value tested.

Usage::

    PYTHONPATH=. .venv/bin/python \\
        experiments/authority_memory/exp_e4_role_dependent.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from authority_memory.admission import apply_admission
from authority_memory.authority_state import AuthorityState, transition
from experiments.authority_memory.corruption_kit import (
    ROLE_UTILITY, make_provisional_certificate_for_role)

OUT_DIR = "tmp/authority_memory/e4_role_dependent"
EVIDENCE_SWEEP = (0.65, 0.8, 0.95)
TAU_E = 0.6
TAU_U = 0.0


def run_role_gated(role: str, evidence_score: float) -> Dict[str, Any]:
    cert = make_provisional_certificate_for_role(
        role, f"cert-gated-{role}-{evidence_score}", evidence_score)
    apply_admission(cert, evidence_score=evidence_score, tau_e=TAU_E,
                    utility_lcb=ROLE_UTILITY[role], tau_u=TAU_U, role=role,
                    reason="role-gated admission", timestamp=1)
    admitted = cert.authority_state == AuthorityState.ADMITTED
    realized = ROLE_UTILITY[role] if admitted else 0.0
    return {"admitted": admitted, "authority_state": cert.authority_state.value,
           "realized_outcome": realized}


def run_naive(role: str, evidence_score: float) -> Dict[str, Any]:
    """Role-blind baseline: promotes PROVISIONAL -> ADMITTED
    unconditionally once evidence clears tau_E, bypassing the utility
    gate entirely --- models raw-history/vector-RAG/shared-graph-style
    architectures that have no receiver-specific utility check at
    all (04_EXPERIMENTS.md's E1 baseline family, this time applied to
    the ADMISSION decision rather than the evidence-authority score)."""
    cert = make_provisional_certificate_for_role(
        role, f"cert-naive-{role}-{evidence_score}", evidence_score)
    if evidence_score >= TAU_E:
        transition(cert, AuthorityState.ADMITTED,
                  evidence_score=evidence_score, utility_lcb=0.0,
                  reason="naive: evidence-only admission", timestamp=1)
    admitted = cert.authority_state == AuthorityState.ADMITTED
    realized = ROLE_UTILITY[role] if admitted else 0.0
    return {"admitted": admitted, "authority_state": cert.authority_state.value,
           "realized_outcome": realized}


def regret(role: str, realized_outcome: float) -> float:
    best = max(ROLE_UTILITY[role], 0.0)
    return best - realized_outcome


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e4_registered.json"), "w") as f:
        json.dump({
            "E4.same_evidence_diverging_authority": {
                "admitted_roles": ["scout", "carrier", "fast"],
                "provisional_only_roles": ["fragile"]},
            "E4.naive_matches_on_beneficial_roles": True,
            "E4.regret_gap_is_exact": {"fragile": 0.5},
            "evidence_sweep": list(EVIDENCE_SWEEP),
            "role_utility": ROLE_UTILITY, "tau_e": TAU_E, "tau_u": TAU_U,
        }, f, indent=2)

    results: Dict[float, Dict[str, Any]] = {}
    for e in EVIDENCE_SWEEP:
        per_role = {}
        for role in ROLE_UTILITY:
            gated = run_role_gated(role, e)
            naive = run_naive(role, e)
            per_role[role] = {
                "role_gated": gated, "naive": naive,
                "regret_role_gated": regret(role, gated["realized_outcome"]),
                "regret_naive": regret(role, naive["realized_outcome"]),
            }
        results[e] = per_role

    same_evidence_ok = all(
        results[e]["scout"]["role_gated"]["admitted"]
        and results[e]["carrier"]["role_gated"]["admitted"]
        and results[e]["fast"]["role_gated"]["admitted"]
        and not results[e]["fragile"]["role_gated"]["admitted"]
        for e in EVIDENCE_SWEEP)

    naive_matches_ok = all(
        results[e][role]["naive"]["admitted"]
        == results[e][role]["role_gated"]["admitted"]
        for e in EVIDENCE_SWEEP for role in ("scout", "carrier", "fast"))

    regret_gap_ok = all(
        results[e]["fragile"]["regret_naive"]
        - results[e]["fragile"]["regret_role_gated"] == 0.5
        for e in EVIDENCE_SWEEP)

    verdict = {
        "E4.same_evidence_diverging_authority": same_evidence_ok,
        "E4.naive_matches_on_beneficial_roles": naive_matches_ok,
        "E4.regret_gap_is_exact": regret_gap_ok,
    }
    go_no_go = all(verdict.values())

    with open(os.path.join(args.out, "e4_results.json"), "w") as f:
        json.dump({"summary": results, "verdict": verdict,
                  "go_no_go": go_no_go}, f, indent=2)

    print(json.dumps(results, indent=2))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"  [{'GO' if go_no_go else 'NO-GO'}] Sprint 6 verdict "
         f"(fourth go/no-go property, README.md §6.4)")
    print(f"Saved: {args.out}/e4_results.json")


if __name__ == "__main__":
    main()
