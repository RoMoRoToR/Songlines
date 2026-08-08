"""Sprint 4 invariant tests --- the decay/expiry-horizon closed form
must agree with what ``apply_staleness`` actually does step by step,
and the world-version check must fire independently of decay.  No
pytest dependency: run

    PYTHONPATH=. python -m authority_memory.tests.test_revocation
"""

from __future__ import annotations

import math

from authority_memory.authority_state import AuthorityState, transition
from authority_memory.certificate import Claim, receive
from authority_memory.revocation import (apply_staleness,
                                         apply_world_version_check, decay,
                                         expiry_horizon, is_stale,
                                         revoke_expired)

_checks = []


def check(name, cond):
    _checks.append((name, bool(cond)))


def _admitted_cert(certificate_id="c1", evidence=1.0, world_version=0):
    cert, _ = receive(certificate_id, Claim("x", "y", "z"), "A",
                      timestamp=0, world_version=world_version)
    transition(cert, AuthorityState.PROVISIONAL, evidence_score=evidence,
              utility_lcb=0.0, reason="setup", timestamp=0)
    transition(cert, AuthorityState.ADMITTED, evidence_score=evidence,
              utility_lcb=0.0, reason="setup", timestamp=0)
    return cert


# ── decay / expiry_horizon closed form ------------------------------
def test_decay_matches_exponential_formula():
    check("decay_zero_age_returns_e0", decay(0.8, 0.0, 0.05) == 0.8)
    check("decay_matches_exp_formula",
          math.isclose(decay(1.0, 10.0, 0.05), math.exp(-0.5)))


def test_expiry_horizon_roundtrips_with_decay():
    e0, rate, tau = 1.0, 0.02, 0.6
    horizon = expiry_horizon(e0, rate, tau)
    just_before = decay(e0, horizon - 0.01, rate)
    just_after = decay(e0, horizon + 0.01, rate)
    check("expiry_horizon_is_exactly_where_decay_crosses_tau",
          just_before > tau and just_after < tau)


def test_expiry_horizon_edge_cases():
    check("expiry_horizon_zero_rate_is_none", expiry_horizon(1.0, 0, 0.6)
         is None)
    check("expiry_horizon_already_below_tau_is_zero",
          expiry_horizon(0.5, 0.02, 0.6) == 0.0)


def test_is_stale_matches_decay_vs_tau():
    check("is_stale_true_past_horizon",
          is_stale(1.0, 30.0, 0.02, 0.6))
    check("is_stale_false_before_horizon",
          not is_stale(1.0, 5.0, 0.02, 0.6))


# ── apply_staleness: only fires past the horizon, exactly once -------
def test_apply_staleness_fires_only_past_horizon():
    cert = _admitted_cert(evidence=1.0)
    rate, tau = 0.02, 0.6
    horizon = expiry_horizon(1.0, rate, tau)
    fired_before = apply_staleness(cert, age=horizon - 1, rate=rate,
                                   tau_e=tau, utility_lcb=0.0,
                                   reason="test", timestamp=0)
    still_admitted = cert.authority_state == AuthorityState.ADMITTED
    fired_after = apply_staleness(cert, age=horizon + 1, rate=rate,
                                  tau_e=tau, utility_lcb=0.0,
                                  reason="test", timestamp=1)
    check("apply_staleness_does_not_fire_before_horizon",
          not fired_before and still_admitted)
    check("apply_staleness_fires_past_horizon_and_lands_expired",
          fired_after and cert.authority_state == AuthorityState.EXPIRED)


def test_apply_staleness_ignores_non_expirable_states():
    cert, _ = receive("c2", Claim("x", "y", "z"), "A", timestamp=0)
    # cert is QUARANTINED --- not in EXPIRABLE_STATES
    fired = apply_staleness(cert, age=1000, rate=0.02, tau_e=0.6,
                            utility_lcb=0.0, reason="test", timestamp=0)
    check("apply_staleness_no_op_outside_expirable_states",
          not fired and cert.authority_state == AuthorityState.QUARANTINED)


# ── world-version check fires immediately, independent of decay -----
def test_world_version_check_fires_on_mismatch_only():
    cert = _admitted_cert(evidence=1.0, world_version=0)
    same_version = apply_world_version_check(
        cert, current_world_version=0, utility_lcb=0.0, reason="test",
        timestamp=1)
    check("world_version_check_no_op_when_versions_match",
          not same_version and cert.authority_state == AuthorityState.ADMITTED)

    mismatched = apply_world_version_check(
        cert, current_world_version=1, utility_lcb=0.0, reason="test",
        timestamp=2)
    check("world_version_check_fires_immediately_on_mismatch",
          mismatched and cert.authority_state == AuthorityState.EXPIRED)


def test_world_version_check_ignores_undecayed_evidence():
    # Even at full evidence_score (no decay at all), a version
    # mismatch alone must force EXPIRED --- the whole point of this
    # trigger being independent of the decay clock.
    cert = _admitted_cert(evidence=1.0, world_version=0)
    fired = apply_world_version_check(
        cert, current_world_version=1, utility_lcb=0.0, reason="test",
        timestamp=0)
    check("world_version_check_ignores_high_evidence_score",
          fired and cert.evidence_score == 1.0
          and cert.authority_state == AuthorityState.EXPIRED)


# ── revoke_expired: the terminal step, only from the right states ---
def test_revoke_expired_only_from_revocable_states():
    cert = _admitted_cert()
    denied = revoke_expired(cert, utility_lcb=0.0, reason="test",
                            timestamp=0)
    check("revoke_expired_no_op_from_admitted",
          not denied and cert.authority_state == AuthorityState.ADMITTED)

    apply_world_version_check(cert, current_world_version=1,
                              utility_lcb=0.0, reason="test", timestamp=1)
    revoked = revoke_expired(cert, utility_lcb=0.0, reason="test",
                             timestamp=2)
    check("revoke_expired_succeeds_from_expired",
          revoked and cert.authority_state == AuthorityState.REVOKED)


def main():
    for fn in (test_decay_matches_exponential_formula,
              test_expiry_horizon_roundtrips_with_decay,
              test_expiry_horizon_edge_cases,
              test_is_stale_matches_decay_vs_tau,
              test_apply_staleness_fires_only_past_horizon,
              test_apply_staleness_ignores_non_expirable_states,
              test_world_version_check_fires_on_mismatch_only,
              test_world_version_check_ignores_undecayed_evidence,
              test_revoke_expired_only_from_revocable_states):
        fn()
    ok = sum(1 for _, c in _checks if c)
    for name, c in _checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"{ok}/{len(_checks)} revocation checks passed")
    return 0 if ok == len(_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
