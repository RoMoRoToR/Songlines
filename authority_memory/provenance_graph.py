"""Authority Memory core — the provenance DAG (Sprint 2).

Two things a naive shared/vector memory conflates: "how many messages
assert X" and "how many INDEPENDENT observations support X".  A
chain ``A -> B -> C -> D`` produces three messages and one piece of
evidence.  This module is what keeps those separate, mechanically:

- ``observe()`` registers a brand-new primary observation (a new
  ``EvidenceOrigin``) and returns a certificate whose ``origin_ids``
  is exactly that one origin.
- ``relay()`` transports an EXISTING certificate's claim from a
  sender to a receiver as a new certificate instance with the SAME
  ``origin_ids`` (no new evidence created) and ``provenance_parents``
  extended by the sender (the transport lineage grows; the evidence
  set does not).

This is the entire technical precondition for Theorem 1 (provenance
non-amplification, ``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/02_THEOREMS.md``
§1): if every relay only ever touches ``provenance_parents``, and
``evidence_score``/authority is ever only a function of
``origin_ids``, then retransmission depth cannot inflate authority by
construction --- not because some threshold happens to hold, but
because there is no code path that lets it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from authority_memory.authority_state import AuthorityDecision
from authority_memory.certificate import Claim, MemoryCertificate, receive


@dataclass(frozen=True)
class EvidenceOrigin:
    """One primary observation --- the only thing that can ever grow
    ``n_eff`` (``metrics.py``).  Registered exactly once per
    ``origin_id``; re-observing the same origin_id is a caller bug,
    not a no-op (fail loud, same discipline as
    ``InvalidAuthorityTransition``)."""
    origin_id: str
    observing_agent: str
    world_version: int
    observed_at: int


@dataclass(frozen=True)
class RelayEdge:
    """One hop of transport: either the origin observation itself
    (``is_origin=True``, ``predecessor_certificate_id=None``) or a
    relay from ``sender`` to ``receiver`` of the certificate that was
    ``predecessor_certificate_id``.  The append-only log of these
    edges is the DAG; ``ProvenanceGraph.lineage`` walks it."""
    certificate_id: str
    predecessor_certificate_id: Optional[str]
    sender: str
    receiver: str
    timestamp: int
    is_origin: bool


class DuplicateOriginError(ValueError):
    """Raised by ``register_origin``/``observe`` when ``origin_id``
    was already registered --- re-observing under the same id would
    silently look like the same evidence being counted twice were it
    allowed to proceed."""


class ProvenanceGraph:
    """Registry of evidence origins plus the transport log between
    agents.  Stateful by design (it is the thing that must stay
    consistent across an entire episode/benchmark run, see
    ``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/05_BENCHMARK_CORRUPTIONS.md``);
    one instance per world/benchmark run, not per certificate.
    """

    def __init__(self) -> None:
        self._origins: Dict[str, EvidenceOrigin] = {}
        self._edges_by_certificate: Dict[str, RelayEdge] = {}

    # ── origins ──────────────────────────────────────────────────
    def register_origin(self, origin_id: str, observing_agent: str,
                        world_version: int,
                        observed_at: int) -> EvidenceOrigin:
        if origin_id in self._origins:
            raise DuplicateOriginError(
                f"origin_id {origin_id!r} is already registered "
                f"(by {self._origins[origin_id].observing_agent!r})")
        origin = EvidenceOrigin(origin_id, observing_agent,
                                world_version, observed_at)
        self._origins[origin_id] = origin
        return origin

    def origin(self, origin_id: str) -> EvidenceOrigin:
        return self._origins[origin_id]

    # ── certificate lifecycle ───────────────────────────────────
    def observe(self, certificate_id: str, claim: Claim,
               observing_agent: str, origin_id: str, world_version: int,
               observed_at: int, **certificate_kwargs: Any
               ) -> Tuple[MemoryCertificate, AuthorityDecision]:
        """A fresh, independent observation. Registers ``origin_id``
        and returns a certificate at QUARANTINED with
        ``origin_ids == {origin_id}`` and empty ``provenance_parents``
        --- this agent is the primary witness, not a relay."""
        self.register_origin(origin_id, observing_agent, world_version,
                             observed_at)
        cert, decision = receive(
            certificate_id, claim, observing_agent, timestamp=observed_at,
            world_version=world_version, origin_ids={origin_id},
            **certificate_kwargs)
        self._edges_by_certificate[certificate_id] = RelayEdge(
            certificate_id=certificate_id,
            predecessor_certificate_id=None, sender=observing_agent,
            receiver=observing_agent, timestamp=observed_at,
            is_origin=True)
        return cert, decision

    def relay(self, certificate: MemoryCertificate, sender: str,
             receiver: str, new_certificate_id: str, timestamp: int,
             **certificate_kwargs: Any
             ) -> Tuple[MemoryCertificate, AuthorityDecision]:
        """Retransmission: ``sender`` (who currently holds
        ``certificate``) tells ``receiver`` about its claim. Returns a
        NEW certificate instance for the receiver carrying the SAME
        ``origin_ids`` (Theorem 1 --- copying a set forward is not the
        same operation as creating a new one) and
        ``provenance_parents`` extended by ``sender``.
        """
        new_provenance = set(certificate.provenance_parents) | {sender}
        new_cert, decision = receive(
            new_certificate_id, certificate.claim, sender,
            timestamp=timestamp,
            world_version=certificate.created_world_version,
            origin_ids=set(certificate.origin_ids),
            provenance_parents=new_provenance, receiver_agent=receiver,
            **certificate_kwargs)
        self._edges_by_certificate[new_certificate_id] = RelayEdge(
            certificate_id=new_certificate_id,
            predecessor_certificate_id=certificate.certificate_id,
            sender=sender, receiver=receiver, timestamp=timestamp,
            is_origin=False)
        return new_cert, decision

    # ── queries ──────────────────────────────────────────────────
    def hop_count(self, certificate: MemoryCertificate) -> int:
        """Number of relays this certificate instance has travelled
        through since its origin --- the x-axis of the social-
        amplification curve (E1, Figure 2)."""
        return len(certificate.provenance_parents)

    def is_laundering(self, certificate: MemoryCertificate,
                      receiver: str) -> bool:
        """True if ``receiver`` already appears in this certificate's
        transport lineage --- corruption type C, provenance
        laundering (``05_BENCHMARK_CORRUPTIONS.md`` §2.C): a claim
        returning to an agent already in its own relay chain. Purely
        diagnostic --- ``relay()`` never grows ``origin_ids`` anyway,
        so laundering cannot inflate authority regardless of whether
        it is flagged; this lets an experiment or test assert that a
        cycle actually occurred."""
        return receiver in certificate.provenance_parents

    def lineage(self, certificate_id: str) -> List[RelayEdge]:
        """Walk the transport chain backward from ``certificate_id``
        to its origin observation, returning edges in
        origin-to-here order. Multi-parent merges (a certificate
        formed by combining two independently-relayed instances) are
        out of scope for Sprint 2 --- structural assimilation across
        certificates is the ``schema.py`` axis (Sprint 9)."""
        chain: List[RelayEdge] = []
        current: Optional[str] = certificate_id
        while current is not None:
            edge = self._edges_by_certificate.get(current)
            if edge is None:
                break
            chain.append(edge)
            current = edge.predecessor_certificate_id
        chain.reverse()
        return chain
