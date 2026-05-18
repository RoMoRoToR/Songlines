"""PeerRuntime — minimal orchestrator that DOES NOT aggregate.

Unlike ``DistributedRuntime`` in ``distributed_memory/``, this runtime:

  - Does NOT own a ConsensusLayer
  - Does NOT cache a "last report" — there is no single report
  - Does NOT have a single trust model — each agent has its own
  - Does NOT pull snapshots from agents to merge centrally

It only:
  - Holds references to agents (a dict, that's it)
  - Holds a single BroadcastBus (a passive transport)
  - Schedules periodic broadcasts every K ticks
  - Calls each agent's own ``process_inbox_and_merge``

You can verify the absence of centralisation by deleting this file:
every agent still works in isolation, and gossip still happens if you
manually wire ``agent.broadcast_now()`` and ``agent.process_inbox_and_merge()``
on the bus.  This class is purely scheduling sugar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.place_alignment import PlaceAlignmentEngine

from peer_memory.broadcast_bus import BroadcastBus
from peer_memory.peer_agent import PeerAgent
from peer_memory.peer_types import PeerMergeReport, PeerView


class PeerRuntime:
    """Scheduling layer for periodic-broadcast peer-to-peer memory.

    Parameters
    ----------
    env_id : str
    broadcast_every_k : int
        Broadcast cadence.  Every ``k`` ticks, all agents broadcast.
    initial_trust : float
        Default trust each agent assigns to a new peer.
    """

    def __init__(
        self,
        *,
        env_id: str = "default",
        broadcast_every_k: int = 5,
        initial_trust: float = 0.7,
        consensus_radius: float = 4.0,
        profile_similarity_threshold: float = 0.5,
        conflict_rules: Optional[ConflictRuleSet] = None,
        alignment_factory=None,
        decay_factory=None,
    ) -> None:
        self.env_id = env_id
        self.broadcast_every_k = int(broadcast_every_k)
        self.initial_trust = float(initial_trust)
        self.consensus_radius = float(consensus_radius)
        self.profile_similarity_threshold = float(profile_similarity_threshold)
        self.conflict_rules = conflict_rules or ConflictRuleSet.songlines_default()
        self._alignment_factory = alignment_factory or (
            lambda: PlaceAlignmentEngine(
                semantic_threshold=0.45, spatial_radius=4.0,
                tag_match_bonus=0.45, min_confidence=0.05,
            )
        )
        self._decay_factory = decay_factory or (lambda: TemporalDecayEngine())

        self.bus = BroadcastBus()
        self._agents: Dict[str, PeerAgent] = {}
        self._tick_count = 0
        self._broadcast_history: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────── agent lifecycle

    def spawn_agent(self, agent_id: str, *, role: str = "scout") -> PeerAgent:
        if agent_id in self._agents:
            raise ValueError(f"agent already registered: {agent_id}")
        agent = PeerAgent(
            agent_id, self.bus,
            env_id=self.env_id, role=role,
            initial_trust=self.initial_trust,
            consensus_radius=self.consensus_radius,
            profile_similarity_threshold=self.profile_similarity_threshold,
            alignment_engine=self._alignment_factory(),
            decay_engine=self._decay_factory(),
            conflict_rules=self.conflict_rules,
        )
        self._agents[agent_id] = agent
        return agent

    def agent(self, agent_id: str) -> PeerAgent:
        return self._agents[agent_id]

    def all_agents(self) -> List[PeerAgent]:
        return list(self._agents.values())

    # ──────────────────────────────────────────────────── observe

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

    # ──────────────────────────────────────────────────── tick

    def tick(self) -> Dict[str, PeerMergeReport]:
        """One global step.

        Order of operations (no shared state mutation between agents):
          1. Each agent refreshes its local graph
          2. Each agent advances its internal step counter
          3. If (tick_count % broadcast_every_k == 0), each agent broadcasts
          4. Each agent processes its inbox and re-merges
        """
        self._tick_count += 1

        for agent in self._agents.values():
            agent.refresh_local()
            agent.tick_step()

        n_delivered = 0
        if self._tick_count % self.broadcast_every_k == 0:
            for agent in self._agents.values():
                n_delivered += agent.broadcast_now()

        reports: Dict[str, PeerMergeReport] = {}
        for agent in self._agents.values():
            reports[agent.agent_id] = agent.process_inbox_and_merge(
                at_step=self._tick_count
            )

        self._broadcast_history.append({
            "tick": self._tick_count,
            "broadcasted": self._tick_count % self.broadcast_every_k == 0,
            "n_delivered": n_delivered,
            "pending_messages": self.bus.pending_message_counts(),
        })
        return reports

    # ──────────────────────────────────────────────────── queries

    def peer_query(self, agent_id: str, target_tag: str, top_k: int = 5):
        """Query a specific agent's PRIVATE peer_view.

        Note: this is intentionally per-agent — there is NO global query.
        Different agents may return different results for the same tag.
        """
        return self._agents[agent_id].peer_query(target_tag, top_k=top_k)

    def local_query(self, agent_id: str, target_tag: str, top_k: int = 5):
        return self._agents[agent_id].local_query(target_tag, top_k=top_k)

    # ──────────────────────────────────────────────────── diagnostics

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def broadcast_history(self) -> List[Dict[str, Any]]:
        return list(self._broadcast_history)

    def stats(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "tick_count": self._tick_count,
            "broadcast_every_k": self.broadcast_every_k,
            "n_agents": len(self._agents),
            "n_broadcasts_done": self.bus.n_broadcasts,
            "agents": {aid: a.stats() for aid, a in self._agents.items()},
        }
