"""Authority Memory core — evaluation vocabulary (Sprint 2 slice:
n_eff and PAF only).

The full vocabulary is specified in
``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/03_METRICS.md`` (FAR,
Authority Precision/Recall, Revocation Latency, PAF, n_eff).  Only
``n_eff`` and ``provenance_amplification_factor`` land in Sprint 2 ---
they are the two metrics ``provenance_graph.py`` alone makes
computable.  FAR needs ``ActionRecord`` (Sprint 11+), Authority
Precision/Recall and Revocation Latency need evaluator-side ground
truth and the revocation mechanism (``evidence.py``/``revocation.py``,
Sprint 4+) --- adding stub functions for those now would not be
testable against anything real, so they are added when their sprint
lands, not before.
"""

from __future__ import annotations

from typing import Iterable, Set, Union

from authority_memory.certificate import MemoryCertificate


def n_eff(certificates: Union[MemoryCertificate,
                              Iterable[MemoryCertificate]]) -> int:
    """Effective independent support: the size of the union of
    ``origin_ids`` across one or more certificates asserting the same
    claim (``03_METRICS.md`` §5) --- NOT a count of certificates or
    messages.  A chain ``A -> B -> C -> D`` passed as a single
    certificate (or as the four instances produced along the way)
    gives ``n_eff == 1``; three agents independently observing the
    same claim give ``n_eff == 3`` regardless of how many times each
    is subsequently relayed.
    """
    if isinstance(certificates, MemoryCertificate):
        certificates = (certificates,)
    origins: Set[str] = set()
    for cert in certificates:
        origins |= cert.origin_ids
    return len(origins)


def provenance_amplification_factor(authority_at_hop_k: float,
                                    authority_at_origin: float) -> float:
    """PAF = authority after k retransmissions / authority at origin
    (``03_METRICS.md`` §4).  A correct provenance-aware system keeps
    PAF close to 1 as k grows (Theorem 1); PAF > 1 means pure
    retransmission --- with no new origin --- inflated evidential
    authority, which is exactly the failure mode this whole layer
    exists to prevent.  This function does not compute authority
    itself (that is ``evidence.py``, later sprints); it only forms
    the ratio from two authority scores an experiment supplies.
    """
    if authority_at_origin == 0:
        raise ValueError(
            "authority_at_origin must be nonzero to compute PAF")
    return authority_at_hop_k / authority_at_origin
