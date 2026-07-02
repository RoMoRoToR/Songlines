"""PeerAgent — autonomous agent with private memory, trust, and merge logic.

Composes:
  - ``AgentMemory`` (from ``distributed_memory/``) for the private
    event store and concept graph
  - ``AsymmetricTrust`` for this agent's view of its peers
  - ``PeerView`` for the current merged belief
  - A reference to a ``BroadcastBus`` for delivery

Lifecycle each tick (driven by ``PeerRuntime``):

  1. observe(...) — record local observations
  2. refresh_local() — rebuild own graph
  3. (every K steps) broadcast() — push own snapshot to bus
  4. process_inbox() — drain pending peer messages
  5. merge() — produce a fresh PeerView from own + received
  6. plan / act — query PeerView for decisions

No method on PeerAgent touches another PeerAgent or any shared
ConsensusLayer.  All inter-agent contact is through the bus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.place_alignment import PlaceAlignmentEngine

from distributed_memory.agent_memory import AgentMemory
from distributed_memory.consensus_types import (
    AgentMemoryView,
    DistributedConcept,
)

from peer_memory.broadcast_bus import BroadcastBus
from peer_memory.peer_merge import local_merge
from peer_memory.peer_trust import AsymmetricTrust
from peer_memory.peer_types import (
    BroadcastMessage,
    PeerMergeReport,
    PeerView,
)


class PeerAgent:
    """An autonomous agent in a peer-to-peer collective.

    Parameters
    ----------
    agent_id : str
    bus : BroadcastBus
        Shared transport for messages.  The bus does NOT compute anything;
        it just delivers messages between agents.
    env_id : str
    initial_trust : float
        Default trust assigned to a new peer when first seen.
    consensus_radius, profile_similarity_threshold
        Parameters of this agent's local merge.
    """

    def __init__(
        self,
        agent_id: str,
        bus: BroadcastBus,
        *,
        env_id: str = "default",
        role: str = "scout",
        initial_trust: float = 0.7,
        consensus_radius: float = 4.0,
        profile_similarity_threshold: float = 0.5,
        alignment_engine: Optional[PlaceAlignmentEngine] = None,
        decay_engine: Optional[TemporalDecayEngine] = None,
        conflict_rules: Optional[ConflictRuleSet] = None,
    ) -> None:
        self.agent_id = agent_id
        self.env_id = env_id
        self.bus = bus
        self._step_counter = 0

        # Private memory layer
        self.memory = AgentMemory(
            agent_id, role=role, env_id=env_id, trust=1.0,
            alignment_engine=alignment_engine,
            decay_engine=decay_engine,
            conflict_rules=conflict_rules,
        )

        # Private trust table
        self.trust = AsymmetricTrust(agent_id, default_trust=initial_trust)

        # Merge parameters
        self.consensus_radius = float(consensus_radius)
        self.profile_similarity_threshold = float(profile_similarity_threshold)
        self.conflict_rules = conflict_rules or ConflictRuleSet.songlines_default()

        # Bus registration
        self.bus.register(agent_id)

        # Private merged view (starts empty)
        self._peer_view: PeerView = PeerView(owner_id=agent_id)
        self._merge_reports: List[PeerMergeReport] = []

        # Last-known snapshot per peer.  Persists across ticks so that
        # merging on a tick with no new inbox messages still includes the
        # peer information from previous broadcasts.
        self._last_known: Dict[str, BroadcastMessage] = {}

    # ──────────────────────────────────────────────────── observe

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
        self.memory.observe(
            place_key, semantic_tags,
            episode_id=episode_id, step_idx=step_idx,
            confidence=confidence, node_freshness=node_freshness,
        )

    def refresh_local(self) -> None:
        self.memory.refresh_local()

    # ──────────────────────────────────────────────────── broadcast

    def broadcast_now(self) -> int:
        """Send this agent's current snapshot to all peers.  Returns recipient count."""
        snapshot = self.memory.snapshot()
        msg = BroadcastMessage(
            sender_id=self.agent_id,
            sent_at_step=self._step_counter,
            snapshot=snapshot,
        )
        return self.bus.broadcast(self.agent_id, msg)

    # ──────────────────────────────────────────────────── receive + merge

    def process_inbox_and_merge(self, at_step: Optional[int] = None) -> PeerMergeReport:
        """Drain inbox, refresh per-peer cache, run local merge, store new PeerView.

        Newer messages from a peer overwrite older ones in the per-peer
        cache.  The merge uses the latest known snapshot for every peer
        we have ever heard from — not only peers who broadcast this tick.
        """
        step = self._step_counter if at_step is None else at_step
        new_messages = self.bus.inbox(self.agent_id).drain()
        for msg in new_messages:
            self._last_known[msg.sender_id] = msg

        peer_messages = list(self._last_known.values())

        view, report = local_merge(
            own_snapshot=self.memory.snapshot(),
            peer_messages=peer_messages,
            trust=self.trust,
            consensus_radius=self.consensus_radius,
            profile_similarity_threshold=self.profile_similarity_threshold,
            conflict_rules=self.conflict_rules,
            at_step=step,
        )
        # Override n_peer_messages_used to reflect only NEW messages,
        # so callers can detect "fresh delivery this tick".
        view.n_peer_messages_used = len(new_messages)
        report.n_peer_messages = len(new_messages)
        self._peer_view = view
        self._merge_reports.append(report)
        return report

    # ──────────────────────────────────────────────────── queries

    def local_query(self, target_tag: str, top_k: int = 5):
        """Pure local — uses ONLY this agent's own observations."""
        return self.memory.local_query(target_tag, top_k=top_k)

    def peer_query(self, target_tag: str, top_k: int = 5) -> List[DistributedConcept]:
        """Query this agent's PRIVATE merged view (own + received peers)."""
        return self._peer_view.top_k(target_tag, k=top_k)

    @property
    def peer_view(self) -> PeerView:
        return self._peer_view

    @property
    def merge_reports(self) -> List[PeerMergeReport]:
        return list(self._merge_reports)

    # ──────────────────────────────────────────────────── trust updates

    def report_outcome_from_peer(self, peer_id: str, was_correct: bool) -> float:
        """Update local trust in ``peer_id`` based on observed outcome.

        Example: a peer claimed water at location X.  This agent went
        there and found water → was_correct=True.
        """
        return self.trust.update_from_outcome(peer_id, was_correct)

    # ──────────────────────────────────────────────────── time / state

    def tick_step(self) -> None:
        """Advance this agent's internal step counter by one."""
        self._step_counter += 1

    @property
    def step_counter(self) -> int:
        return self._step_counter

    def stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "step_counter": self._step_counter,
            "trust_table": self.trust.snapshot(),
            "n_local_concepts": self.memory.snapshot().n_local_concepts,
            "n_peer_view_concepts": len(self._peer_view.distributed_concepts),
            "contributing_peers": list(self._peer_view.contributing_peer_ids),
            "n_merges": len(self._merge_reports),
        }
