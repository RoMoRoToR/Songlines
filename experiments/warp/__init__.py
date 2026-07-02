"""Semantic Warp — provenance-conditioned completion measurement layer.

W* is NOT a fifth Q/R/M/C stage: it is a provenance predicate on M*
events.  This package only reads memory-layer state and annotates the
existing M*-locks with foreignness phi; no planner or memory invariant
is touched.

See docs/FRONTIER_SEMANTIC_WARP_2026-07-02.md for the research design.
"""

from experiments.warp.warp_types import MStarEvent, WarpEpisodeLog
from experiments.warp.warp_instrumentation import (
    ProvenanceLedger,
    candidate_provenance,
)

__all__ = [
    "MStarEvent",
    "WarpEpisodeLog",
    "ProvenanceLedger",
    "candidate_provenance",
]
