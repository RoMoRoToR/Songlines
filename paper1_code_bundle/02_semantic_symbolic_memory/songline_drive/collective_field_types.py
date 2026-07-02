"""Phase 4 — collective semantic field types.

All types are pure dataclasses with no runtime dependencies.

``FieldMode`` is the new independent configuration axis (§1 of Phase 4 spec).
It does NOT mix with existing axes: milestone_mode, graph_update_mode,
final_exit_mode, intent_mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────── configuration axis


class FieldMode:
    """Independent configuration axis for Phase 4 semantic field.

    none        — field disabled; all retrieval via Phase 1/2/3 stack
    descriptive — field computed and logged; not used in retrieval
    read_only   — field participates in concept-recall reranking
    coordinated — field additionally manages reservation / occupancy / role pressure
    """

    NONE = "none"
    DESCRIPTIVE = "descriptive"
    READ_ONLY = "read_only"
    COORDINATED = "coordinated"

    _VALID = frozenset({NONE, DESCRIPTIVE, READ_ONLY, COORDINATED})

    @classmethod
    def validate(cls, mode: str) -> str:
        if mode not in cls._VALID:
            raise ValueError(
                f"Unknown field mode {mode!r}; valid={sorted(cls._VALID)}"
            )
        return mode


# ─────────────────────────────────────────────────────── per-channel state


@dataclass
class FieldChannelState:
    """Activation state for one (concept, channel) pair.

    ``activation`` is the primary field value used for query reranking.
    ``activation_fast`` / ``activation_slow`` are EMA-smoothed variants at
    two time constants; novelty shows as fast↑ > slow↑.

    ``support_pressure`` stores the channel affinity I(k,c) so that
    ``explain_score()`` can report it without recomputing.
    """

    channel: str
    activation: float = 0.0
    activation_fast: float = 0.0
    activation_slow: float = 0.0
    freshness: float = 0.0
    belief_strength: float = 0.0
    conflict_pressure: float = 0.0
    novelty_pressure: float = 0.0
    occupancy_pressure: float = 0.0
    reservation_pressure: float = 0.0
    support_pressure: float = 0.0      # channel affinity I(k, c)
    last_update_seq: int = -1


# ──────────────────────────────────────────────────── per-concept field state


@dataclass
class FieldCellState:
    """Full field state for one concept across all channels."""

    concept_id: str
    channels: Dict[str, FieldChannelState] = field(default_factory=dict)
    base_confidence: float = 0.0
    base_freshness: float = 0.0
    base_purity: float = 0.0
    base_conflict: float = 0.0
    support_count: int = 0
    supporting_agents: List[str] = field(default_factory=list)
    centroid_xy: Optional[Tuple[float, float]] = None


# ──────────────────────────────────────────────────────────── query types


@dataclass
class FieldQuery:
    """A channel-specific query into the semantic field."""

    channel: str                   # e.g. "water_source"
    requesting_agent_id: str
    env_id: Optional[str] = None
    top_k: int = 3
    min_activation: float = 0.0
    current_seq: int = 0
    field_weight: float = 0.30    # weight of field activation in combined score


@dataclass
class FieldQueryResult:
    """Single result from a field query."""

    concept_id: str
    channel: str
    activation: float
    concept_score: float                            # raw concept recall score
    field_score: float                              # field activation (= activation)
    combined_score: float                           # weighted combination
    centroid_xy: Optional[Tuple[float, float]]
    member_places: List[Tuple[str, Tuple[Any, ...]]]
    supporting_agents: List[str]
    explanation: Dict[str, float]                   # debug breakdown


# ──────────────────────────────────────────────── coordination types (Phase 4c)


@dataclass
class FieldReservation:
    """Soft reservation on a (concept, channel) placed by an agent (Phase 4c)."""

    concept_id: str
    channel: str
    agent_id: str
    reserved_at_seq: int
    expires_at_seq: int


@dataclass
class AgentFieldFootprint:
    """Intent pressure that an agent exerts on a (concept, channel) (Phase 4c)."""

    agent_id: str
    channel: str
    concept_id: str
    intent_pressure: float
    seq: int
