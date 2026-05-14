"""Typed contracts for the Phase 1 collective memory substrate.

This module is intentionally dependency-light: it only imports from the
standard library so the existing single-agent stack continues to work
even if the collective layer is never loaded.

Design notes (Phase 1 — collective blackboard, no semantic field):
- ``CollectiveEvent`` is the smallest publishable unit. Every event carries
  full provenance (who, when, in which env/episode/step, with what
  monotonic global sequence number).
- ``BeliefRecord`` is the per-observation atom that backs every fused
  query result. Aggregation lives in :mod:`collective_memory`.
- Place identity in Phase 1 is just ``(env_id, place_key)`` where
  ``place_key`` is whatever the publishing agent supplies (typically a
  grid cell tuple). Phase 2 (place_alignment) will canonicalise this.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EventType(str, Enum):
    PLACE_OBSERVED = "place_observed"
    CONCEPT_CONFIRMED = "concept_confirmed"
    TRANSITION_VALIDATED = "transition_validated"
    HAZARD_CHANGED = "hazard_changed"
    RESOURCE_DEPLETED = "resource_depleted"
    ROUTE_FAILED = "route_failed"
    ROUTE_SUCCEEDED = "route_succeeded"
    INTENT_COMMITTED = "intent_committed"
    INTENT_RELEASED = "intent_released"


@dataclass(frozen=True)
class AgentSignature:
    agent_id: str
    role: str = "explorer"
    trust: float = 1.0
    metadata: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class EventProvenance:
    agent_id: str
    episode_id: int
    step_idx: int
    env_id: str
    wall_clock_seq: int


@dataclass
class CollectiveEvent:
    event_id: int
    event_type: EventType
    provenance: EventProvenance
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class BeliefRecord:
    agent_id: str
    episode_id: int
    step_idx: int
    confidence: float
    freshness: float
    wall_clock_seq: int


@dataclass
class PlaceBeliefAggregate:
    place_key: Tuple[Any, ...]
    env_id: str
    tag_records: Dict[str, List[BeliefRecord]] = field(default_factory=dict)
    transition_records: Dict[Tuple[Any, ...], List[BeliefRecord]] = field(default_factory=dict)
    last_seen_seq: int = -1


@dataclass
class CollectiveQuery:
    intent_type: str
    target_tag: str
    score_weights: Dict[str, float] = field(default_factory=dict)
    penalty_weights: Dict[str, float] = field(default_factory=dict)
    requesting_agent_id: str = "unknown"
    env_id: Optional[str] = None
    min_supporting_agents: int = 1
    min_fused_score: float = 0.0
    exclude_self: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectiveQueryResult:
    place_key: Tuple[Any, ...]
    env_id: str
    fused_score: float
    contributing_agents: List[str]
    contributing_event_seqs: List[int]
    used_other_agent_knowledge: bool
    target_tag: str
    per_tag_fused: Dict[str, float] = field(default_factory=dict)


@dataclass
class CollectiveDecisionExplanation:
    place_key: Tuple[Any, ...]
    env_id: str
    target_tag: str
    fused_score: float
    contributing_agents: List[str]
    per_tag_fused: Dict[str, float]
    contributing_event_seqs: List[int]
    used_other_agent_knowledge: bool
    requesting_agent_id: str


def query_from_predicate(
    intent_type: str,
    target_predicate: Any,
    requesting_agent_id: str,
    env_id: Optional[str] = None,
    exclude_self: bool = False,
    min_supporting_agents: int = 1,
    min_fused_score: float = 0.0,
) -> CollectiveQuery:
    """Adapter that converts a single-agent ``SemanticTargetPredicate`` (from
    :mod:`songline_drive.types`) into a ``CollectiveQuery`` without importing
    the heavy module — duck-typed."""
    target_tag = getattr(target_predicate, "tag_name", "")
    score_weights = dict(getattr(target_predicate, "score_weights", {}) or {})
    penalty_weights = dict(getattr(target_predicate, "penalty_weights", {}) or {})
    if not score_weights:
        score_weights = {target_tag: 1.0}
    return CollectiveQuery(
        intent_type=str(intent_type),
        target_tag=str(target_tag),
        score_weights=score_weights,
        penalty_weights=penalty_weights,
        requesting_agent_id=str(requesting_agent_id),
        env_id=env_id,
        exclude_self=bool(exclude_self),
        min_supporting_agents=int(min_supporting_agents),
        min_fused_score=float(min_fused_score),
    )
