"""Authority Memory core — the three-gate admission criterion
(Sprint 6 slice: the gate function itself, driven by a hand-specified
``utility_lcb`` value; the LEARNED causal-utility estimator that
produces real Û values from randomized intervention data is
Sprint 7-8, ``causal_utility.py`` --- this module does not depend on
it and never will need to change when it lands, since it only
consumes a number, not a fitting procedure).

Operationalises the admission criterion from
``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/01_FORMAL_MODEL.md`` §4:

    A_i(m) = 1  <=>  E_i(m) >= tau_E  AND  LCB_i(m) >= tau_U  AND  V_i(m) = 1

Evidence answers "is this true/admissible"; utility answers "is this
useful to ME, in MY role, right now"; applicability answers "does
this even apply to someone like me".  None of the three can
substitute for another --- README.md's "utility ≠ truth" becomes an
actual AND-gate here, not a slogan.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from authority_memory.authority_state import AuthorityState, transition
from authority_memory.certificate import MemoryCertificate


def is_applicable(certificate: MemoryCertificate, role: str,
                  state: Optional[Dict[str, Any]] = None) -> bool:
    """V_i(m): does ``certificate.role_conditions``/``state_conditions``
    permit use by a receiver with this role/state?  Empty
    role_conditions/state_conditions means "no restriction declared"
    --- applicable to everyone, not to no one; an experiment that
    wants a restriction states it explicitly via
    ``role_conditions={"allowed_roles": [...]}``.
    """
    allowed_roles = certificate.role_conditions.get("allowed_roles")
    if allowed_roles is not None and role not in allowed_roles:
        return False
    state = state or {}
    for key, required in certificate.state_conditions.items():
        if state.get(key) != required:
            return False
    return True


def decide_admission(evidence_score: float, tau_e: float,
                     utility_lcb: float, tau_u: float,
                     applicable: bool) -> bool:
    """The three-gate AND --- pure, no side effects.  Callers decide
    what to do with the boolean; this function never touches the FSM
    (same "mechanism, not policy" separation as Sprint 1's
    ``transition()``, which never computes evidence_score/utility_lcb
    itself either)."""
    return evidence_score >= tau_e and utility_lcb >= tau_u and applicable


def apply_admission(certificate: MemoryCertificate, *, evidence_score: float,
                    tau_e: float, utility_lcb: float, tau_u: float,
                    role: str, state: Optional[Dict[str, Any]] = None,
                    reason: str, timestamp: int) -> bool:
    """Evaluate the three gates for ``role``/``state`` against
    ``certificate`` and, if all three pass and the certificate is
    currently PROVISIONAL, transition it to ADMITTED.  Returns True
    iff a transition was applied; a certificate that fails any gate,
    or that is not PROVISIONAL to begin with, is left untouched ---
    remaining PROVISIONAL (limited, non-persistent use only) is the
    correct outcome for a role the claim does not serve, not an
    error state.
    """
    if certificate.authority_state != AuthorityState.PROVISIONAL:
        return False
    applicable = is_applicable(certificate, role, state)
    if not decide_admission(evidence_score, tau_e, utility_lcb, tau_u,
                            applicable):
        return False
    transition(certificate, AuthorityState.ADMITTED,
              evidence_score=evidence_score, utility_lcb=utility_lcb,
              reason=reason, timestamp=timestamp)
    return True
