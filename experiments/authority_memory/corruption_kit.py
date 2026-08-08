"""Authority Memory experiments — the corruption kit (Sprint 3 slice:
type C, provenance laundering; Sprint 4 slice: type A, staleness;
Sprint 6 slice: type D, role-dependent validity).

The full six-type kit (staleness, false testimony, provenance
laundering, role-dependent validity, context-dependent exception,
semantic aliasing) is specified in
``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/05_BENCHMARK_CORRUPTIONS.md``
§2.  Types A, C and D land here so far, because E1, E3 and E4 are the
only experiments that need them yet; the other two arrive with the
experiments that actually exercise them (E8 exceptions, E6 for
aliasing-style ablation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from authority_memory.authority_state import AuthorityState, transition
from authority_memory.certificate import Claim, MemoryCertificate, receive
from authority_memory.provenance_graph import ProvenanceGraph, RelayEdge

# 'A' is always the origin witness; the pool is who a message CAN be
# relayed to next (excluding whoever currently holds it).
AGENT_POOL = ("A", "B", "C", "D", "E", "F")


def false_claim() -> Claim:
    """The fixed, evaluator-known-false claim used throughout E1.
    Registered as false before any run (``05_BENCHMARK_CORRUPTIONS.md``
    §2.B) --- what makes the experiment honest rather than self-
    fulfilling is that authority is scored the same way regardless of
    truth value; the danger E1 demonstrates is that naive
    architectures amplify THIS false claim's authority exactly as
    readily as they would amplify a true one.
    """
    return Claim(subject="room_4", relation="contains", object="resource",
                conditions={})


@dataclass(frozen=True)
class LaunderingChain:
    """One realised relay chain for the type-C scenario: a false claim
    observed once by ``origin_agent``, then relayed ``n_hops`` times
    through agents drawn (with replacement) from ``AGENT_POOL`` --- a
    hop MAY return to an agent already earlier in the chain (the
    laundering case) purely by chance, not by construction, so
    Theorem 1 gets checked against genuinely varied topology rather
    than one hand-picked cycle."""
    claim: Claim
    origin_agent: str
    origin_id: str
    hops: List[RelayEdge]                  # length == n_hops, origin excluded
    certificates: List[MemoryCertificate]  # length == n_hops + 1
    graph: ProvenanceGraph


def build_chain(seed: int, n_hops: int = 5,
                world_version: int = 0) -> LaunderingChain:
    """Build one randomised relay chain of exactly ``n_hops`` hops
    starting from a single origin observation by agent 'A'. Reuses
    ``ProvenanceGraph.observe``/``relay`` (Sprint 2) directly --- the
    chain IS a real provenance DAG, not a hand-rolled stand-in.
    """
    rng = random.Random(seed)
    graph = ProvenanceGraph()
    claim = false_claim()
    origin_agent = "A"
    origin_id = f"obs-A-seed{seed}"
    cert, _ = graph.observe(f"cert-seed{seed}-hop0", claim, origin_agent,
                            origin_id, world_version=world_version,
                            observed_at=0)
    certificates = [cert]
    holder = origin_agent
    for hop in range(1, n_hops + 1):
        candidates = [a for a in AGENT_POOL if a != holder]
        receiver = rng.choice(candidates)
        cert, _ = graph.relay(
            cert, sender=holder, receiver=receiver,
            new_certificate_id=f"cert-seed{seed}-hop{hop}", timestamp=hop)
        certificates.append(cert)
        holder = receiver
    lineage = graph.lineage(certificates[-1].certificate_id)
    hops = lineage[1:]   # drop the origin edge; keep the n_hops relay edges
    return LaunderingChain(claim=claim, origin_agent=origin_agent,
                           origin_id=origin_id, hops=hops,
                           certificates=certificates, graph=graph)


# ── type A: staleness --------------------------------------------------
def stale_claim() -> Claim:
    """The fixed claim used throughout E3: true when observed, then
    silently becomes false when the world changes ---
    ``05_BENCHMARK_CORRUPTIONS.md`` §2.A ("door X is open... world
    changes... B later needs door X"). Unlike ``false_claim()`` (E1,
    known false from the start), this one starts TRUE --- staleness
    is about a claim that used to be right and stopped being told
    so, not about a claim that was always wrong."""
    return Claim(subject="bridge_north", relation="state", object="open",
                conditions={})


@dataclass(frozen=True)
class StalenessScenario:
    """One type-A scenario: a claim observed once at t=0 under
    ``world_version_before``; the world silently changes at
    ``t_invalidated`` to ``world_version_after`` without telling any
    agent directly --- whether/when a given architecture NOTICES this
    is exactly what E3 measures."""
    claim: Claim
    origin_agent: str
    t_invalidated: int
    world_version_before: int
    world_version_after: int


def build_staleness_scenario(t_invalidated: int, world_version_before: int = 0,
                             world_version_after: int = 1
                             ) -> StalenessScenario:
    return StalenessScenario(claim=stale_claim(), origin_agent="A",
                             t_invalidated=t_invalidated,
                             world_version_before=world_version_before,
                             world_version_after=world_version_after)


def make_admitted_certificate(scenario: StalenessScenario,
                              certificate_id: str,
                              evidence_at_t0: float = 1.0
                              ) -> MemoryCertificate:
    """Construct a certificate for ``scenario`` and force it through
    RECEIVED -> QUARANTINED -> PROVISIONAL -> ADMITTED. Sprint 4 tests
    REVOCATION, not promotion (that is Sprint 5's contract, already
    closed for the real admission path); this certificate simply
    needs to START at ADMITTED with a known ``evidence_at_t0``, so the
    intermediate transitions use placeholder ``utility_lcb`` values
    rather than a calibrated admission decision.
    """
    cert, _ = receive(certificate_id, scenario.claim, scenario.origin_agent,
                      timestamp=0,
                      world_version=scenario.world_version_before)
    transition(cert, AuthorityState.PROVISIONAL,
              evidence_score=evidence_at_t0, utility_lcb=0.0,
              reason="test setup", timestamp=0)
    transition(cert, AuthorityState.ADMITTED, evidence_score=evidence_at_t0,
              utility_lcb=0.0, reason="test setup", timestamp=0)
    return cert


# ── type D: role-dependent validity -----------------------------------
def role_dependent_claim() -> Claim:
    """The fixed claim used throughout E4: 'route A is traversable' ---
    ``05_BENCHMARK_CORRUPTIONS.md`` §2.D (a path safe for one role,
    unsafe for another). Evidential admissibility (E) is IDENTICAL
    across every role that observes this claim; only the causal
    utility of ACTING on it differs by role --- E4 exists to show
    that authority (A) tracks the second thing, not just the first."""
    return Claim(subject="route_A", relation="state", object="traversable",
                conditions={})


# Ground-truth causal utility of using route_A, by role --- hand
# specified for this synthetic scenario (the LEARNED estimator that
# would produce numbers like these from intervention data is Sprint
# 7-8, causal_utility.py; E4 tests whether the ADMISSION GATE uses a
# role-specific utility correctly, not whether one can be learned).
ROLE_UTILITY = {"scout": 0.5, "carrier": 0.2, "fragile": -0.5, "fast": 0.3}


def make_provisional_certificate_for_role(role: str, certificate_id: str,
                                          evidence_score: float,
                                          world_version: int = 0
                                          ) -> MemoryCertificate:
    """Construct a per-role belief certificate already at PROVISIONAL
    with the SAME ``evidence_score`` across every role (the doc's "E
    identical between roles"). Receiver-specificity is modelled the
    same way Sprint 5 modelled it: one certificate INSTANCE per
    receiver (here, per role), not one shared object with a
    role-keyed field --- authority genuinely differs per instance,
    not per a lookup inside a single instance.
    """
    cert, _ = receive(certificate_id, role_dependent_claim(), "A",
                      timestamp=0, world_version=world_version,
                      receiver_agent=role)
    transition(cert, AuthorityState.PROVISIONAL,
              evidence_score=evidence_score, utility_lcb=0.0,
              reason="test setup", timestamp=0)
    return cert
