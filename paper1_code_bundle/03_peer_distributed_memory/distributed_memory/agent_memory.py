"""AgentMemory — per-agent isolated memory.

Wraps the existing Phase 1/2/3/4 primitives so each agent has its own
private:

  - ``CollectiveMemory`` (event store)
  - ``ConceptRecallLayer`` (concept graph + recall)
  - Optional ``SemanticField`` (Phase 4)

Agents do **not** share these objects.  Cross-agent integration happens
exclusively via ``ConsensusLayer.merge()`` on snapshots exported by
``AgentMemory.snapshot()``.

This file holds no business logic that depends on other agents — it is
purely a single-agent wrapper.  See ``consensus_layer.py`` for fusion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_field_types import FieldMode
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import AgentSignature
from songline_drive.concept_recall import ConceptRecallLayer, ConceptRecallResult
from songline_drive.field_adapter import FieldAdapter
from songline_drive.place_alignment import PlaceAlignmentEngine
from songline_drive.semantic_field import SemanticField

from distributed_memory.consensus_types import AgentMemoryView


class AgentMemory:
    """One agent's private memory stack.

    Parameters
    ----------
    agent_id : str
        Stable identifier for this agent.
    role : str
        Free-form role label (e.g., ``"scout"``).
    env_id : str
        Environment identifier; written into every observation.
    alignment_engine : PlaceAlignmentEngine, optional
        Override the default alignment engine.
    enable_field : bool
        If True, also build a per-agent ``SemanticField`` (Phase 4).
    field_mode : str
        ``FieldMode`` value when ``enable_field=True``.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        role: str = "scout",
        env_id: str = "default",
        trust: float = 1.0,
        alignment_engine: Optional[PlaceAlignmentEngine] = None,
        decay_engine: Optional[TemporalDecayEngine] = None,
        conflict_rules: Optional[ConflictRuleSet] = None,
        enable_field: bool = False,
        field_mode: str = FieldMode.READ_ONLY,
        field_channels: Optional[List[str]] = None,
        recency_lambda: float = 0.97,
        convergence_min_score: float = 0.4,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.env_id = env_id
        self.trust = float(trust)

        self.memory = CollectiveMemory(
            recency_lambda=recency_lambda,
            convergence_min_score=convergence_min_score,
        )
        self.memory.register_agent(AgentSignature(agent_id, role=role, trust=trust))

        self.alignment_engine = alignment_engine or PlaceAlignmentEngine(
            semantic_threshold=0.45,
            spatial_radius=4.0,
            tag_match_bonus=0.45,
            min_confidence=0.05,
        )
        self.decay_engine = decay_engine or TemporalDecayEngine()
        self.conflict_rules = conflict_rules or ConflictRuleSet.songlines_default()

        self.recall = ConceptRecallLayer(
            self.alignment_engine,
            only_dominant_tag=True,
            min_concept_support=1,
            decay_engine=self.decay_engine,
            conflict_rules=self.conflict_rules,
        )

        self.field: Optional[SemanticField] = None
        self.adapter: Optional[FieldAdapter] = None
        if enable_field:
            self.field = SemanticField(
                channels=field_channels,
                mode=field_mode,
                lambda_decay=0.95,
                alpha_belief=0.60,
                eta_conflict=0.30,
                xi_occupancy=0.20,
                gamma_diffusion=0.10,
                diffusion_steps=1,
            )
            self.adapter = FieldAdapter(
                self.field, self.recall, field_weight=0.35, mode=field_mode,
            )

        self._last_observed_seq: int = -1

    # ──────────────────────────────────────────────────── observe / refresh

    def observe(
        self,
        place_key: Tuple[Any, ...],
        semantic_tags: Dict[str, float],
        *,
        episode_id: int,
        step_idx: int,
        confidence: float = 1.0,
        node_freshness: float = 1.0,
    ) -> None:
        """Record a single observation in this agent's private event log."""
        self.memory.publish_event(
            "place_observed",
            self.agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=self.env_id,
            payload={
                "place_key": list(place_key),
                "semantic_tags": dict(semantic_tags),
                "node_freshness": float(node_freshness),
            },
            confidence=float(confidence),
        )
        self._last_observed_seq = self.memory._next_seq - 1  # noqa: SLF001

    def refresh_local(self, current_seq: Optional[int] = None) -> None:
        """Rebuild local concept graph + (optionally) local field."""
        if self.adapter is not None:
            self.adapter.refresh(self.memory, current_seq=current_seq)
        else:
            self.recall.refresh(self.memory)

    # ──────────────────────────────────────────────────── local queries

    def local_query(
        self,
        target_tag: str,
        top_k: int = 5,
        current_seq: Optional[int] = None,
    ) -> List[ConceptRecallResult]:
        """Pure local recall — uses only this agent's observations."""
        seq = (
            current_seq
            if current_seq is not None
            else self.memory._next_seq  # noqa: SLF001
        )
        return self.recall.query(
            target_tag=target_tag,
            requesting_agent_id=self.agent_id,
            env_id=self.env_id,
            top_k=top_k,
            current_seq=seq,
        )

    # ──────────────────────────────────────────────────── snapshot for consensus

    def snapshot(self) -> AgentMemoryView:
        """Export this agent's local view for the consensus layer."""
        graph = self.recall.graph
        local_concepts: Dict[str, Dict[str, Any]] = {}
        local_concept_ids: List[str] = []

        if graph is not None:
            for cid, concept in graph.concepts.items():
                local_concept_ids.append(cid)
                local_concepts[cid] = {
                    "concept_id": cid,
                    "dominant_tag": concept.dominant_tag,
                    "semantic_profile": dict(concept.semantic_profile),
                    "centroid_xy": (
                        tuple(concept.centroid_xy)
                        if concept.centroid_xy is not None
                        else None
                    ),
                    "support_count": len(concept.member_place_keys),
                    "supporting_agents": sorted(concept.supporting_agents),
                    "confidence": concept.confidence,
                    "freshness": concept.freshness,
                    "conflict_score": concept.conflict_score,
                }

        return AgentMemoryView(
            agent_id=self.agent_id,
            trust=self.trust,
            n_events=len(self.memory.events),
            n_local_concepts=len(local_concepts),
            last_observed_seq=self._last_observed_seq,
            local_concept_ids=local_concept_ids,
            local_concepts=local_concepts,
        )

    # ──────────────────────────────────────────────────── diagnostics

    def stats(self) -> Dict[str, Any]:
        graph = self.recall.graph
        return {
            "agent_id": self.agent_id,
            "trust": self.trust,
            "n_events": len(self.memory.events),
            "n_local_concepts": len(graph.concepts) if graph else 0,
            "has_field": self.field is not None,
        }
