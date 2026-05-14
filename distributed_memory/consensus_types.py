"""Datatypes for the distributed memory consensus layer.

The collective view is *derived* from per-agent local views via the
``ConsensusLayer``.  These types describe the fused result.

Key concepts
------------
``AgentMemoryView``
    A handle to one agent's local memory (event store, concept graph, field).
    Exported by ``AgentMemory.snapshot()`` for the consensus layer to read.

``DistributedConcept``
    A concept that has been aligned across one or more agents.  Stores who
    contributed, what each agent locally believed, and a trust-weighted
    fusion of those beliefs.

``AgentDisagreement``
    Two or more agents observed the same location but assigned incompatible
    dominant tags (e.g., one agent says ``water_source``, another says
    ``hazard_edge``).  Surfaced for explicit downstream resolution.

``ConsensusReport``
    Output of a single ``ConsensusLayer.merge()`` cycle: distributed
    concepts, disagreements, and aggregate statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class AgentMemoryView:
    """Read-only snapshot of one agent's local memory state."""

    agent_id: str
    trust: float
    n_events: int
    n_local_concepts: int
    last_observed_seq: int
    local_concept_ids: List[str] = field(default_factory=list)
    # Concept summaries: cid -> {dominant_tag, centroid_xy, semantic_profile, ...}
    local_concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "trust": self.trust,
            "n_events": self.n_events,
            "n_local_concepts": self.n_local_concepts,
            "last_observed_seq": self.last_observed_seq,
            "local_concept_ids": list(self.local_concept_ids),
            "local_concepts": {
                cid: dict(c) for cid, c in self.local_concepts.items()
            },
        }


@dataclass
class AgentContribution:
    """One agent's contribution to a distributed concept."""

    agent_id: str
    local_concept_id: str
    trust: float
    local_dominant_tag: str
    local_support: int
    local_confidence: float
    local_freshness: float


@dataclass
class DistributedConcept:
    """A concept after cross-agent consensus.

    The fields prefixed with ``consensus_`` are trust-weighted aggregations
    across all contributing agents.  ``contributions`` exposes the raw
    per-agent view so callers can drill down.
    """

    consensus_id: str
    centroid_xy: Tuple[float, float]
    consensus_dominant_tag: str
    consensus_profile: Dict[str, float] = field(default_factory=dict)
    consensus_confidence: float = 0.0
    inter_agent_agreement: float = 1.0
    contributions: List[AgentContribution] = field(default_factory=list)
    disagreement_flags: List[str] = field(default_factory=list)

    @property
    def n_agents(self) -> int:
        return len(self.contributions)

    @property
    def contributing_agent_ids(self) -> List[str]:
        return [c.agent_id for c in self.contributions]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_id": self.consensus_id,
            "centroid_xy": list(self.centroid_xy),
            "consensus_dominant_tag": self.consensus_dominant_tag,
            "consensus_profile": dict(self.consensus_profile),
            "consensus_confidence": round(self.consensus_confidence, 5),
            "inter_agent_agreement": round(self.inter_agent_agreement, 5),
            "n_agents": self.n_agents,
            "contributions": [
                {
                    "agent_id": c.agent_id,
                    "local_concept_id": c.local_concept_id,
                    "trust": round(c.trust, 4),
                    "local_dominant_tag": c.local_dominant_tag,
                    "local_support": c.local_support,
                    "local_confidence": round(c.local_confidence, 4),
                    "local_freshness": round(c.local_freshness, 4),
                }
                for c in self.contributions
            ],
            "disagreement_flags": list(self.disagreement_flags),
        }


@dataclass
class AgentDisagreement:
    """Two agents disagree on the dominant tag at a shared location."""

    consensus_id: str
    centroid_xy: Tuple[float, float]
    agent_a: str
    agent_b: str
    tag_a: str
    tag_b: str
    severity: float  # in [0, 1], higher = more severe

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_id": self.consensus_id,
            "centroid_xy": list(self.centroid_xy),
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "tag_a": self.tag_a,
            "tag_b": self.tag_b,
            "severity": round(self.severity, 4),
        }


@dataclass
class ConsensusReport:
    """Output of one ``ConsensusLayer.merge()`` cycle."""

    distributed_concepts: List[DistributedConcept] = field(default_factory=list)
    disagreements: List[AgentDisagreement] = field(default_factory=list)
    n_agents: int = 0
    n_aligned: int = 0       # consensus concepts with > 1 contributing agent
    n_isolated: int = 0      # consensus concepts with exactly 1 agent
    avg_agreement: float = 1.0

    def by_tag(self, tag: str) -> List[DistributedConcept]:
        return [c for c in self.distributed_concepts if c.consensus_dominant_tag == tag]

    def top_k(self, tag: str, k: int = 3) -> List[DistributedConcept]:
        items = self.by_tag(tag)
        items.sort(key=lambda c: c.consensus_confidence, reverse=True)
        return items[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "n_aligned": self.n_aligned,
            "n_isolated": self.n_isolated,
            "n_disagreements": len(self.disagreements),
            "avg_agreement": round(self.avg_agreement, 5),
            "distributed_concepts": [c.to_dict() for c in self.distributed_concepts],
            "disagreements": [d.to_dict() for d in self.disagreements],
        }
