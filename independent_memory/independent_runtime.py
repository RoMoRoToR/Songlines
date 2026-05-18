"""IndependentRuntime — orchestrates fully isolated agents.

This runtime intentionally does NOT have any cross-agent operation:

  - No ``tick()`` that triggers a global merge (because there is no merge)
  - No ``collective_query()`` method (because there is no collective)
  - No bus, no consensus layer, no aggregation report

It exists purely so the user can spawn N agents through a single object
and step them in a loop.  Each ``tick()`` is equivalent to calling
``refresh_local()`` on every agent — no information crosses the boundary
between agents.

The runtime is essentially a list of agents with a step counter.  It is
included for **API symmetry** with ``DistributedRuntime`` and
``PeerRuntime`` so the three variants of multi-agent memory have the
same surface area in experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.concept_recall import ConceptRecallResult
from songline_drive.place_alignment import PlaceAlignmentEngine

from independent_memory.independent_agent import IndependentAgent


class IndependentRuntime:
    """Spawn and step isolated agents.  Performs no aggregation, ever."""

    def __init__(
        self,
        *,
        env_id: str = "default",
        conflict_rules: Optional[ConflictRuleSet] = None,
        alignment_factory=None,
        decay_factory=None,
    ) -> None:
        self.env_id = env_id
        self.conflict_rules = conflict_rules or ConflictRuleSet.songlines_default()
        self._alignment_factory = alignment_factory or (
            lambda: PlaceAlignmentEngine(
                semantic_threshold=0.45, spatial_radius=4.0,
                tag_match_bonus=0.45, min_confidence=0.05,
            )
        )
        self._decay_factory = decay_factory or (lambda: TemporalDecayEngine())
        self._agents: Dict[str, IndependentAgent] = {}
        self._tick_count = 0

    # ──────────────────────────────────────────────────── agent lifecycle

    def spawn_agent(self, agent_id: str, *, role: str = "scout") -> IndependentAgent:
        if agent_id in self._agents:
            raise ValueError(f"agent already registered: {agent_id}")
        agent = IndependentAgent(
            agent_id, env_id=self.env_id, role=role,
            alignment_engine=self._alignment_factory(),
            decay_engine=self._decay_factory(),
            conflict_rules=self.conflict_rules,
        )
        self._agents[agent_id] = agent
        return agent

    def agent(self, agent_id: str) -> IndependentAgent:
        return self._agents[agent_id]

    def all_agents(self) -> List[IndependentAgent]:
        return list(self._agents.values())

    # ──────────────────────────────────────────────────── observe / tick

    def observe(
        self, agent_id: str, place_key, semantic_tags,
        *, episode_id: int = 0, step_idx: int = 0,
        confidence: float = 1.0, node_freshness: float = 1.0,
    ) -> None:
        self._agents[agent_id].observe(
            place_key, semantic_tags,
            episode_id=episode_id, step_idx=step_idx,
            confidence=confidence, node_freshness=node_freshness,
        )

    def tick(self) -> None:
        """One global step.  Each agent independently refreshes its local graph.

        There is NO inter-agent communication.  This method's only purpose
        is to advance every agent's local state by one step.
        """
        self._tick_count += 1
        for agent in self._agents.values():
            agent.refresh_local()
            agent.tick_step()

    # ──────────────────────────────────────────────────── queries

    def local_query(
        self, agent_id: str, target_tag: str, top_k: int = 5,
    ) -> List[ConceptRecallResult]:
        return self._agents[agent_id].local_query(target_tag, top_k=top_k)

    # NOTE: there is no `collective_query` here.  Adding one would
    # require aggregation, which would violate the "independent" contract.

    # ──────────────────────────────────────────────────── diagnostics

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def stats(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "tick_count": self._tick_count,
            "n_agents": len(self._agents),
            "aggregation": "none",
            "communication": "none",
            "agents": {aid: a.stats() for aid, a in self._agents.items()},
        }
