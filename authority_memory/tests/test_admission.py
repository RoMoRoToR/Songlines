"""Sprint 6 invariant tests --- the three-gate admission criterion
must AND correctly, and must be the only way PROVISIONAL ->
ADMITTED happens.  No pytest dependency: run

    PYTHONPATH=. python -m authority_memory.tests.test_admission
"""

from __future__ import annotations

from authority_memory.admission import (apply_admission, decide_admission,
                                        is_applicable)
from authority_memory.authority_state import AuthorityState, transition
from authority_memory.certificate import Claim, MemoryCertificate, receive

_checks = []


def check(name, cond):
    _checks.append((name, bool(cond)))


def _provisional_cert(**kwargs):
    cert, _ = receive("c1", Claim("x", "y", "z"), "A", timestamp=0,
                      **kwargs)
    transition(cert, AuthorityState.PROVISIONAL, evidence_score=0.8,
              utility_lcb=0.0, reason="setup", timestamp=0)
    return cert


# ── decide_admission: pure three-gate AND ----------------------------
def test_decide_admission_truth_table():
    check("all_three_pass", decide_admission(0.8, 0.6, 0.5, 0.0, True))
    check("evidence_gate_fails",
          not decide_admission(0.5, 0.6, 0.5, 0.0, True))
    check("utility_gate_fails",
          not decide_admission(0.8, 0.6, -0.1, 0.0, True))
    check("applicability_gate_fails",
          not decide_admission(0.8, 0.6, 0.5, 0.0, False))
    check("boundary_values_pass_inclusively",
          decide_admission(0.6, 0.6, 0.0, 0.0, True))


# ── is_applicable: empty conditions mean unrestricted ----------------
def test_is_applicable_empty_conditions_means_everyone():
    cert = _provisional_cert()
    check("no_role_conditions_is_applicable_to_anyone",
          is_applicable(cert, "fragile") and is_applicable(cert, "scout"))


def test_is_applicable_restricted_roles():
    cert = _provisional_cert(
        role_conditions={"allowed_roles": ["scout", "fast"]})
    check("allowed_role_passes", is_applicable(cert, "scout"))
    check("disallowed_role_fails", not is_applicable(cert, "fragile"))


def test_is_applicable_state_conditions():
    cert = _provisional_cert(state_conditions={"door_state": "open"})
    check("matching_state_passes",
          is_applicable(cert, "scout", {"door_state": "open"}))
    check("mismatched_state_fails",
          not is_applicable(cert, "scout", {"door_state": "closed"}))
    check("missing_state_key_fails",
          not is_applicable(cert, "scout", {}))


# ── apply_admission: only fires from PROVISIONAL, only on full pass -
def test_apply_admission_promotes_on_full_pass():
    cert = _provisional_cert()
    fired = apply_admission(cert, evidence_score=0.8, tau_e=0.6,
                            utility_lcb=0.5, tau_u=0.0, role="scout",
                            reason="test", timestamp=1)
    check("apply_admission_promotes_to_admitted",
          fired and cert.authority_state == AuthorityState.ADMITTED)


def test_apply_admission_denies_on_utility_failure():
    cert = _provisional_cert()
    fired = apply_admission(cert, evidence_score=0.8, tau_e=0.6,
                            utility_lcb=-0.5, tau_u=0.0, role="fragile",
                            reason="test", timestamp=1)
    check("apply_admission_leaves_provisional_on_utility_fail",
          not fired and cert.authority_state == AuthorityState.PROVISIONAL)


def test_apply_admission_only_from_provisional():
    cert, _ = receive("c2", Claim("x", "y", "z"), "A", timestamp=0)
    # cert is QUARANTINED, not PROVISIONAL
    fired = apply_admission(cert, evidence_score=0.9, tau_e=0.6,
                            utility_lcb=0.9, tau_u=0.0, role="scout",
                            reason="test", timestamp=1)
    check("apply_admission_no_op_outside_provisional",
          not fired and cert.authority_state == AuthorityState.QUARANTINED)


def test_apply_admission_same_evidence_different_roles():
    # The exact E4 shape: identical evidence, role-specific utility,
    # divergent authority outcome --- the fourth go/no-go property
    # (README.md §6.4): A_i(m) != A_j(m) even though E_i(m) == E_j(m).
    e = 0.8
    scout = _provisional_cert()
    fragile = _provisional_cert()
    apply_admission(scout, evidence_score=e, tau_e=0.6, utility_lcb=0.5,
                    tau_u=0.0, role="scout", reason="test", timestamp=1)
    apply_admission(fragile, evidence_score=e, tau_e=0.6, utility_lcb=-0.5,
                    tau_u=0.0, role="fragile", reason="test", timestamp=1)
    check("same_evidence_diverging_authority",
          scout.authority_state == AuthorityState.ADMITTED
          and fragile.authority_state == AuthorityState.PROVISIONAL)


def main():
    for fn in (test_decide_admission_truth_table,
              test_is_applicable_empty_conditions_means_everyone,
              test_is_applicable_restricted_roles,
              test_is_applicable_state_conditions,
              test_apply_admission_promotes_on_full_pass,
              test_apply_admission_denies_on_utility_failure,
              test_apply_admission_only_from_provisional,
              test_apply_admission_same_evidence_different_roles):
        fn()
    ok = sum(1 for _, c in _checks if c)
    for name, c in _checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"{ok}/{len(_checks)} admission checks passed")
    return 0 if ok == len(_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
