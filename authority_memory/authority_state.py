"""Authority Memory core — the authority state machine (Sprint 1).

``AuthorityState`` is the explicit object this frontier introduces in
place of a scalar trust weight (``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/01_FORMAL_MODEL.md``
§5).  Every transition is validated against ``ALLOWED_TRANSITIONS``
and produces an ``AuthorityDecision`` --- an append-only audit record
answering "why did this agent come to trust this testimony", the same
way the runtime's episodic store is append-only for evidence
(``songlines/record.py`` I1 invariant).  Nothing here computes E or U;
this module is the mechanism, not the policy --- the policy (when a
transition is warranted) is ``evidence.py``/``causal_utility.py``/
``admission.py`` in later sprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet


class AuthorityState(str, Enum):
    RECEIVED = "received"
    QUARANTINED = "quarantined"
    PROVISIONAL = "provisional"
    ADMITTED = "admitted"
    CONTESTED = "contested"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    REVOKED = "revoked"


# RECEIVED       -- message just arrived
# QUARANTINED    -- visible to reasoning, FORBIDDEN as durable action
#                   authority
# PROVISIONAL    -- limited use (exploration/route-proposal), does not
#                   write to persistent memory
# ADMITTED       -- participates in planning, may change persistent
#                   memory
# CONTESTED      -- incompatible evidence has appeared
# SUPERSEDED     -- a newer version of the same claim exists
# EXPIRED        -- past valid_until (the W2 distance law, ported
#                   unchanged onto the authority layer)
# REVOKED        -- terminal: authority permanently withdrawn
ALLOWED_TRANSITIONS: Dict[AuthorityState, FrozenSet[AuthorityState]] = {
    AuthorityState.RECEIVED: frozenset({AuthorityState.QUARANTINED}),
    AuthorityState.QUARANTINED: frozenset({AuthorityState.PROVISIONAL}),
    AuthorityState.PROVISIONAL: frozenset({
        AuthorityState.ADMITTED, AuthorityState.EXPIRED}),
    AuthorityState.ADMITTED: frozenset({
        AuthorityState.CONTESTED, AuthorityState.SUPERSEDED,
        AuthorityState.EXPIRED}),
    AuthorityState.CONTESTED: frozenset({AuthorityState.REVOKED}),
    AuthorityState.SUPERSEDED: frozenset({AuthorityState.REVOKED}),
    AuthorityState.EXPIRED: frozenset({AuthorityState.REVOKED}),
    AuthorityState.REVOKED: frozenset(),
}

# States in which a certificate may durably drive action / persistent
# memory writes (README.md §1: "не иметь права изменять persistent
# belief/memory structure и определять действие" outside this set).
DURABLE_ACTION_STATES: FrozenSet[AuthorityState] = frozenset({
    AuthorityState.ADMITTED})

# States in which limited, non-persistent use is allowed (exploration
# / route-proposal only --- never a write to persistent memory).
LIMITED_USE_STATES: FrozenSet[AuthorityState] = frozenset({
    AuthorityState.PROVISIONAL})

TERMINAL_STATES: FrozenSet[AuthorityState] = frozenset({
    AuthorityState.REVOKED})


class InvalidAuthorityTransition(ValueError):
    """Raised when a transition is attempted outside
    ``ALLOWED_TRANSITIONS`` --- the state machine is uncircumventable
    on the API level, the same discipline as the forbidden methods on
    ``independent_memory.IndependentAgent`` (no silent no-ops, no
    partial application: either the transition is valid or it never
    happened)."""

    def __init__(self, current: AuthorityState, attempted: AuthorityState):
        self.current = current
        self.attempted = attempted
        allowed = sorted(s.value for s in ALLOWED_TRANSITIONS[current])
        super().__init__(
            f"{current.value} -> {attempted.value} is not a permitted "
            f"authority transition (allowed from {current.value}: "
            f"{allowed})")


@dataclass(frozen=True)
class ValidationEvent:
    """The empirical check behind a receiver's own trial of a
    certificate --- "measured, not told" (S3 admission control,
    ``docs/FRONTIER_UCSM_2026-07-27.md`` §S3), formalised as its own
    logged event rather than a side effect buried inside an admission
    function."""
    certificate_id: str
    receiver_id: str
    local_observation: Any
    outcome: bool
    world_version: int
    support: bool


@dataclass(frozen=True)
class AuthorityDecision:
    """One logged authority-state transition. The audit trail that
    answers, mechanically and after the fact, "why did the agent
    believe this" --- see ``ActionRecord`` (Sprint 11+) for the
    further step from decision to action."""
    certificate_id: str
    previous_state: AuthorityState
    new_state: AuthorityState
    evidence_score: float
    utility_lcb: float
    reason: str
    timestamp: int


def transition(certificate: "MemoryCertificate",
               new_state: AuthorityState, *, evidence_score: float,
               utility_lcb: float, reason: str,
               timestamp: int) -> AuthorityDecision:
    """Validate and apply one authority-state transition, mutating
    ``certificate.authority_state``/``evidence_score`` in place and
    returning the logged ``AuthorityDecision``.

    Raises ``InvalidAuthorityTransition`` and leaves the certificate
    untouched if ``new_state`` is not reachable from the current
    state --- callers must not catch this to retry with a different
    target state as a way of "finding" a valid one; an invalid
    transition means the caller's policy logic (evidence.py /
    admission.py, later sprints) has a bug.
    """
    current = certificate.authority_state
    if new_state not in ALLOWED_TRANSITIONS[current]:
        raise InvalidAuthorityTransition(current, new_state)
    decision = AuthorityDecision(
        certificate_id=certificate.certificate_id,
        previous_state=current,
        new_state=new_state,
        evidence_score=evidence_score,
        utility_lcb=utility_lcb,
        reason=reason,
        timestamp=timestamp,
    )
    certificate.authority_state = new_state
    certificate.evidence_score = evidence_score
    return decision


def has_action_authority(state: AuthorityState) -> bool:
    """True only for states allowed to drive persistent-memory writes
    and planning (currently just ADMITTED) --- the single predicate
    every consumer of a certificate must check before acting on it."""
    return state in DURABLE_ACTION_STATES


def has_limited_authority(state: AuthorityState) -> bool:
    """True for states allowed non-persistent, provisional use
    (exploration / route-proposal) --- distinct from full action
    authority; callers must not treat this as equivalent to
    ``has_action_authority``."""
    return state in LIMITED_USE_STATES
