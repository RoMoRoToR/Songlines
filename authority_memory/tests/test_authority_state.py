"""Sprint 1 invariant tests --- the certificate/authority-state
machine must never regress.  No pytest dependency: run

    PYTHONPATH=. python -m authority_memory.tests.test_authority_state

Scope (``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/10_CODE_LAYOUT.md``
§5): every allowed/forbidden transition of the FSM, plus basic
construction sanity for ``Claim``/``MemoryCertificate``.  Provenance
non-amplification (Theorem 1) and the LLM-boundary test are Sprint 2
/ Sprint 10 scope --- not here.
"""

from __future__ import annotations

from authority_memory.authority_state import (
    ALLOWED_TRANSITIONS, AuthorityDecision, AuthorityState,
    InvalidAuthorityTransition, TERMINAL_STATES, ValidationEvent,
    has_action_authority, has_limited_authority, transition)
from authority_memory.certificate import Claim, MemoryCertificate, receive

_checks = []


def check(name, cond):
    _checks.append((name, bool(cond)))


def _claim():
    return Claim(subject="route_17", relation="safe_for",
                object="fragile_agent", conditions={"door_state": "open"})


def _cert(state=AuthorityState.RECEIVED):
    cert = MemoryCertificate(certificate_id="c1", claim=_claim(),
                             source_agent="agent-A")
    cert.authority_state = state  # test-only: bypass the FSM to set
                                   # up an arbitrary starting state
    return cert


# ── every allowed transition succeeds and mutates state -------------
def test_all_allowed_transitions_succeed():
    ok = True
    for source, targets in ALLOWED_TRANSITIONS.items():
        for target in targets:
            cert = _cert(source)
            decision = transition(cert, target, evidence_score=0.5,
                                  utility_lcb=0.1, reason="test",
                                  timestamp=1)
            ok = ok and (
                cert.authority_state == target
                and isinstance(decision, AuthorityDecision)
                and decision.previous_state == source
                and decision.new_state == target)
    check("all_allowed_transitions_succeed_and_mutate", ok)


# ── every forbidden transition raises and leaves state untouched ----
def test_all_forbidden_transitions_raise():
    ok = True
    for source in AuthorityState:
        allowed = ALLOWED_TRANSITIONS[source]
        for target in AuthorityState:
            if target in allowed:
                continue
            cert = _cert(source)
            raised = False
            try:
                transition(cert, target, evidence_score=0.5,
                          utility_lcb=0.1, reason="test", timestamp=1)
            except InvalidAuthorityTransition:
                raised = True
            ok = ok and raised and cert.authority_state == source
    check("all_forbidden_transitions_raise_and_leave_state_untouched",
          ok)


# ── REVOKED is terminal -----------------------------------------------
def test_revoked_is_terminal():
    check("revoked_has_no_outgoing_transitions",
          AuthorityState.REVOKED in TERMINAL_STATES
          and len(ALLOWED_TRANSITIONS[AuthorityState.REVOKED]) == 0)


# ── the RECEIVED -> QUARANTINED transition is automatic on receipt --
def test_receive_lands_in_quarantined():
    cert, decision = receive("c2", _claim(), "agent-B", timestamp=3,
                             world_version=1)
    check("receive_lands_in_quarantined",
          cert.authority_state == AuthorityState.QUARANTINED
          and decision.previous_state == AuthorityState.RECEIVED
          and decision.new_state == AuthorityState.QUARANTINED
          and cert.observed_at == 3
          and cert.created_world_version == 1)


# ── action-authority predicates are exact, not "close enough" -------
def test_action_authority_predicates_are_exact():
    admitted_only = all(
        has_action_authority(s) == (s == AuthorityState.ADMITTED)
        for s in AuthorityState)
    provisional_only = all(
        has_limited_authority(s) == (s == AuthorityState.PROVISIONAL)
        for s in AuthorityState)
    check("has_action_authority_true_only_for_admitted", admitted_only)
    check("has_limited_authority_true_only_for_provisional",
          provisional_only)
    # QUARANTINED must never have either --- the central §1 invariant
    # (README.md): available-but-not-yet-authoritative.
    check("quarantined_has_no_authority_at_all",
          not has_action_authority(AuthorityState.QUARANTINED)
          and not has_limited_authority(AuthorityState.QUARANTINED))


# ── a full happy-path lifecycle round-trips correctly ----------------
def test_full_lifecycle_round_trip():
    cert = _cert(AuthorityState.RECEIVED)
    path = [AuthorityState.QUARANTINED, AuthorityState.PROVISIONAL,
           AuthorityState.ADMITTED, AuthorityState.CONTESTED,
           AuthorityState.REVOKED]
    decisions = []
    for target in path:
        decisions.append(transition(cert, target, evidence_score=0.7,
                                    utility_lcb=0.2, reason="lifecycle",
                                    timestamp=len(decisions)))
    check("full_lifecycle_round_trip",
          cert.authority_state == AuthorityState.REVOKED
          and [d.new_state for d in decisions] == path
          and all(d.certificate_id == "c1" for d in decisions))


# ── origin_ids and provenance_parents are independent sets ----------
def test_origin_and_provenance_are_independent_sets():
    cert = _cert()
    cert.origin_ids.add("obs_agentA_e1_s0")
    cert.provenance_parents.add("agent-B")
    cert.provenance_parents.add("agent-C")
    check("origin_ids_and_provenance_parents_are_independent",
          cert.origin_ids == {"obs_agentA_e1_s0"}
          and cert.provenance_parents == {"agent-B", "agent-C"}
          and cert.origin_ids != cert.provenance_parents)


# ── structural_relation is validated against the five UCSM ops ------
def test_structural_relation_validation():
    raised = False
    try:
        MemoryCertificate(certificate_id="c3", claim=_claim(),
                          source_agent="agent-A",
                          structural_relation="NOT_A_REAL_OP")
    except ValueError:
        raised = True
    valid = MemoryCertificate(certificate_id="c4", claim=_claim(),
                              source_agent="agent-A",
                              structural_relation="MERGE")
    check("structural_relation_rejects_unknown_value", raised)
    check("structural_relation_accepts_canonical_value",
          valid.structural_relation == "MERGE")


# ── ValidationEvent / AuthorityDecision are immutable audit entries -
def test_audit_records_are_frozen():
    ve = ValidationEvent(certificate_id="c1", receiver_id="agent-B",
                         local_observation={"door": "open"},
                         outcome=True, world_version=2, support=True)
    ad = AuthorityDecision(certificate_id="c1",
                           previous_state=AuthorityState.QUARANTINED,
                           new_state=AuthorityState.PROVISIONAL,
                           evidence_score=0.6, utility_lcb=0.1,
                           reason="test", timestamp=5)
    frozen_ok = True
    for obj, field, value in ((ve, "support", False),
                              (ad, "reason", "tampered")):
        try:
            setattr(obj, field, value)
            frozen_ok = False
        except (AttributeError, TypeError):
            pass
    check("validation_event_and_authority_decision_are_frozen",
          frozen_ok)


def main():
    for fn in (test_all_allowed_transitions_succeed,
              test_all_forbidden_transitions_raise,
              test_revoked_is_terminal,
              test_receive_lands_in_quarantined,
              test_action_authority_predicates_are_exact,
              test_full_lifecycle_round_trip,
              test_origin_and_provenance_are_independent_sets,
              test_structural_relation_validation,
              test_audit_records_are_frozen):
        fn()
    ok = sum(1 for _, c in _checks if c)
    for name, c in _checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"{ok}/{len(_checks)} authority-state checks passed")
    return 0 if ok == len(_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
