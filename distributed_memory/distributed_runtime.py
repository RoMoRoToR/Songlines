"""DistributedRuntime — orchestrates a population of per-agent memories
plus a consensus layer.

Typical usage::

    runtime = DistributedRuntime(env_id="my-env")
    runtime.spawn_agent("scout-A", trust=1.0)
    runtime.spawn_agent("scout-B", trust=0.9)

    # Agents observe (typically driven by an external simulator)
    runtime.observe("scout-A", place_key=(0, 0),
                    semantic_tags={"water_source": 0.95})

    # Refresh all local graphs + run consensus
    report = runtime.tick()

    # Query
    top = report.top_k("water_source", k=3)
    local = runtime.local_query("scout-A", "water_source", top_k=3)
    collective = runtime.collective_query("water_source", top_k=3,
                                          requesting_agent="scout-A")

The runtime keeps the most recent ``ConsensusReport`` so ``collective_query``
can answer immediately without re-merging.  Call ``tick()`` to refresh.

Backward-compatibility note: this runtime does **not** touch any shared
``CollectiveMemory`` from Phase 1.  Each agent has its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_field_types import FieldMode
from songline_drive.concept_recall import ConceptRecallResult
from songline_drive.place_alignment import PlaceAlignmentEngine

from distributed_memory.agent_memory import AgentMemory
from distributed_memory.consensus_layer import ConsensusLayer
from distributed_memory.consensus_types import (
    ConsensusReport,
    DistributedConcept,
)
from distributed_memory.trust_model import TrustModel


class DistributedRuntime:
    """Multi-agent runtime with per-agent memory + consensus layer."""

    def __init__(
        self,
        *,
        env_id: str = "default",
        trust_model: Optional[TrustModel] = None,
        consensus_radius: float = 4.0,
        consensus_profile_threshold: float = 0.5,
        enable_field: bool = False,
        field_mode: str = FieldMode.READ_ONLY,
        field_channels: Optional[List[str]] = None,
        alignment_engine_factory=None,
        decay_engine_factory=None,
        conflict_rules: Optional[ConflictRuleSet] = None,
    ) -> None:
        self.env_id = env_id
        self.trust_model = trust_model or TrustModel()
        self.conflict_rules = conflict_rules or ConflictRuleSet.songlines_default()
        self.consensus = ConsensusLayer(
            trust_model=self.trust_model,
            consensus_radius=consensus_radius,
            profile_similarity_threshold=consensus_profile_threshold,
            conflict_rules=self.conflict_rules,
        )
        self.enable_field = bool(enable_field)
        self.field_mode = field_mode
        self.field_channels = field_channels

        self._agents: Dict[str, AgentMemory] = {}
        self._last_report: Optional[ConsensusReport] = None
        self._tick_count = 0

        self._alignment_factory = alignment_engine_factory or (
            lambda: PlaceAlignmentEngine(
                semantic_threshold=0.45,
                spatial_radius=4.0,
                tag_match_bonus=0.45,
                min_confidence=0.05,
            )
        )
        self._decay_factory = decay_engine_factory or (lambda: TemporalDecayEngine())

    # ──────────────────────────────────────────────────── agent lifecycle

    def spawn_agent(
        self,
        agent_id: str,
        *,
        trust: float = 1.0,
        role: str = "scout",
    ) -> AgentMemory:
        if agent_id in self._agents:
            raise ValueError(f"agent_id already registered: {agent_id}")
        self.trust_model.register(agent_id, trust=trust)

        agent = AgentMemory(
            agent_id,
            role=role,
            env_id=self.env_id,
            trust=trust,
            alignment_engine=self._alignment_factory(),
            decay_engine=self._decay_factory(),
            conflict_rules=self.conflict_rules,
            enable_field=self.enable_field,
            field_mode=self.field_mode,
            field_channels=self.field_channels,
        )
        self._agents[agent_id] = agent
        return agent

    def agent(self, agent_id: str) -> AgentMemory:
        return self._agents[agent_id]

    def all_agents(self) -> List[AgentMemory]:
        return list(self._agents.values())

    # ──────────────────────────────────────────────────── observe / tick

    def observe(
        self,
        agent_id: str,
        place_key: Tuple[Any, ...],
        semantic_tags: Dict[str, float],
        *,
        episode_id: int = 0,
        step_idx: int = 0,
        confidence: float = 1.0,
        node_freshness: float = 1.0,
    ) -> None:
        self._agents[agent_id].observe(
            place_key=place_key,
            semantic_tags=semantic_tags,
            episode_id=episode_id,
            step_idx=step_idx,
            confidence=confidence,
            node_freshness=node_freshness,
        )

    def tick(self) -> ConsensusReport:
        """Refresh all local graphs, then run consensus.

        Returns the new ``ConsensusReport`` and stores it as ``last_report``.
        """
        self._tick_count += 1
        for agent in self._agents.values():
            agent.refresh_local()
        views = [agent.snapshot() for agent in self._agents.values()]
        # Keep the per-snapshot trust in sync with the model
        for view in views:
            view.trust = self.trust_model.get(view.agent_id)
        report = self.consensus.merge(views)
        self._last_report = report
        return report

    # ──────────────────────────────────────────────────── queries

    def local_query(
        self,
        agent_id: str,
        target_tag: str,
        top_k: int = 5,
    ) -> List[ConceptRecallResult]:
        return self._agents[agent_id].local_query(target_tag, top_k=top_k)

    def collective_query(
        self,
        target_tag: str,
        top_k: int = 5,
        requesting_agent: Optional[str] = None,
    ) -> List[DistributedConcept]:
        """Query the latest ConsensusReport for distributed concepts.

        Call ``tick()`` first to refresh.  ``requesting_agent`` is currently
        used only for diagnostics — the consensus view is shared.
        """
        if self._last_report is None:
            raise RuntimeError("Call tick() before collective_query().")
        return self._last_report.top_k(target_tag, k=top_k)

    # ──────────────────────────────────────────────────── diagnostics

    @property
    def last_report(self) -> Optional[ConsensusReport]:
        return self._last_report

    def stats(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "n_agents": len(self._agents),
            "tick_count": self._tick_count,
            "trust": self.trust_model.all_trusts(),
            "last_report": (
                self._last_report.to_dict() if self._last_report else None
            ),
        }
