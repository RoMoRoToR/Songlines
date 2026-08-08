"""Authority Memory core — the certificate object (Sprint 1).

Extends the frozen Songlines Runtime v1 record type
(``m = (G, C, E, U, P, T, R, F, A)``, ``songlines/record.py``) with a
third, until-now-implicit axis: **E**, evidential admissibility.  The
runtime record answers "what does this memory item say and how
useful is it" (U x S, two independent axes, see B1 in
``docs/CLAIM_EVIDENCE_MATRIX.md``); the certificate additionally
answers "is this memory item allowed to act" --- a receiver-specific,
time-dependent, revocable decision tracked by ``authority_state``
(``authority_state.py``).

This module wraps the runtime record; it does not replace it.  Claim
content is a structured triple, not a string blob, so that provenance
(``origin_ids`` vs ``provenance_parents``, Sprint 2) and evidential
scoring have something typed to operate on --- see
``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/01_FORMAL_MODEL.md`` §§2-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from authority_memory.authority_state import (AuthorityDecision,
                                               AuthorityState, transition)

# Canonical structural_relation values --- the five UCSM formation
# operations (songlines.analogy / decide()), reused by name so the
# certificate's structural axis stays in the same vocabulary as the
# runtime it wraps, without importing the enum itself (this module
# must not depend on the runtime's internal representation).
STRUCTURAL_RELATIONS = ("MERGE", "EXCEPTION", "NEW_SCHEMA", "REPEAT",
                        "DROP")


@dataclass
class Claim:
    """Structured content of a testimony --- a subject/relation/object
    triple plus applicability conditions, never a free-text blob (the
    reason: provenance and evidential scoring need a typed target;
    an unstructured string would leave content the only unstructured
    part of an otherwise fully structured certificate)."""
    subject: str
    relation: str
    object: str
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCertificate:
    """The authority-layer wrapper around one memory item.

    ``origin_ids`` and ``provenance_parents`` are deliberately two
    different sets (Sprint 2 populates both): ``provenance_parents``
    grows with every relay (A tells B tells C); ``origin_ids`` grows
    only when a NEW independent observation supports the claim.  This
    split is the entire technical precondition for Theorem 1
    (provenance non-amplification, ``02_THEOREMS.md`` §1) --- collapse
    it into one set and non-amplification becomes unstatable.
    """
    certificate_id: str
    claim: Claim
    source_agent: str

    origin_ids: Set[str] = field(default_factory=set)
    provenance_parents: Set[str] = field(default_factory=set)

    receiver_agent: Optional[str] = None

    created_world_version: int = 0
    observed_at: int = 0
    valid_until: Optional[int] = None

    role_conditions: Dict[str, Any] = field(default_factory=dict)
    state_conditions: Dict[str, Any] = field(default_factory=dict)

    evidence_score: float = 0.0
    utility_mean: float = 0.0
    utility_std: float = 0.0

    # One of STRUCTURAL_RELATIONS, or None before formation runs.
    structural_relation: Optional[str] = None
    exception_ids: List[str] = field(default_factory=list)

    authority_state: AuthorityState = AuthorityState.RECEIVED

    def __post_init__(self) -> None:
        if (self.structural_relation is not None
                and self.structural_relation not in STRUCTURAL_RELATIONS):
            raise ValueError(
                f"structural_relation must be one of "
                f"{STRUCTURAL_RELATIONS} or None, got "
                f"{self.structural_relation!r}")


def receive(certificate_id: str, claim: Claim, source_agent: str, *,
           timestamp: int, world_version: int = 0,
           **certificate_kwargs: Any
           ) -> Tuple[MemoryCertificate, AuthorityDecision]:
    """Construct a certificate at RECEIVED and immediately apply the
    one transition that is automatic on receipt (RECEIVED ->
    QUARANTINED, ``01_FORMAL_MODEL.md`` §5) --- every certificate that
    ever entered the system by any path (Sprint 2 provenance, Sprint
    10 LLM extraction) must go through exactly this, so it lives here
    once rather than being re-implemented per producer.
    """
    cert = MemoryCertificate(
        certificate_id=certificate_id, claim=claim,
        source_agent=source_agent, created_world_version=world_version,
        observed_at=timestamp, **certificate_kwargs)
    decision = transition(
        cert, AuthorityState.QUARANTINED, evidence_score=0.0,
        utility_lcb=0.0, reason="automatic on receipt",
        timestamp=timestamp)
    return cert, decision
