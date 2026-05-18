"""Datatypes for peer-to-peer memory exchange.

Key principle: no global "consensus" exists.  Each agent maintains
**its own** ``PeerView`` (a merged graph) which may differ from any
other agent's view.

``BroadcastMessage``
    What flows over the wire: a snapshot of one agent's local graph,
    timestamped, with the sender's id.  Reuses ``AgentMemoryView`` from
    ``distributed_memory`` to avoid duplicating schema work.

``PeerInbox``
    Per-agent buffer of messages received since last drain.  Receivers
    decide when to drain and merge.

``PeerView``
    One agent's *current* belief state, formed by merging its own
    local graph with whatever it most recently received from peers.
    Each agent has its own.  Snapshot at this layer is what the agent
    queries when making decisions.

``PeerMergeReport``
    Diagnostic-only — what happened during the most recent local merge
    (which peers contributed, what conflicts were detected).  Not used
    by the protocol itself, only for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from distributed_memory.consensus_types import (
    AgentContribution,
    AgentDisagreement,
    AgentMemoryView,
    DistributedConcept,
)


@dataclass
class BroadcastMessage:
    """A single message flowing peer-to-peer over the broadcast bus."""

    sender_id: str
    sent_at_step: int
    snapshot: AgentMemoryView

    def __repr__(self) -> str:
        return (
            f"BroadcastMessage(from={self.sender_id!r}, "
            f"step={self.sent_at_step}, "
            f"n_concepts={self.snapshot.n_local_concepts})"
        )


@dataclass
class PeerInbox:
    """Per-agent buffer of received messages, drained by the agent on merge."""

    owner_id: str
    messages: List[BroadcastMessage] = field(default_factory=list)

    def push(self, msg: BroadcastMessage) -> None:
        self.messages.append(msg)

    def drain(self) -> List[BroadcastMessage]:
        out = list(self.messages)
        self.messages.clear()
        return out

    def __len__(self) -> int:
        return len(self.messages)


@dataclass
class PeerView:
    """One agent's merged belief — its own local graph + peer snapshots.

    Lives inside the agent.  No two agents share this object.  Different
    agents may have different views because:
      - their local graphs differ (they observed different things)
      - their trust toward each peer may differ (asymmetric)
      - they may have received different subsets of broadcasts
    """

    owner_id: str
    formed_at_step: int = -1
    distributed_concepts: List[DistributedConcept] = field(default_factory=list)
    disagreements: List[AgentDisagreement] = field(default_factory=list)
    n_peer_messages_used: int = 0
    contributing_peer_ids: List[str] = field(default_factory=list)

    def by_tag(self, tag: str) -> List[DistributedConcept]:
        return [c for c in self.distributed_concepts
                if c.consensus_dominant_tag == tag]

    def top_k(self, tag: str, k: int = 3) -> List[DistributedConcept]:
        items = self.by_tag(tag)
        items.sort(key=lambda c: c.consensus_confidence, reverse=True)
        return items[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "formed_at_step": self.formed_at_step,
            "n_peer_messages_used": self.n_peer_messages_used,
            "contributing_peer_ids": list(self.contributing_peer_ids),
            "n_distributed_concepts": len(self.distributed_concepts),
            "n_disagreements": len(self.disagreements),
            "distributed_concepts": [c.to_dict() for c in self.distributed_concepts],
            "disagreements": [d.to_dict() for d in self.disagreements],
        }


@dataclass
class PeerMergeReport:
    """Diagnostic snapshot of one local merge cycle."""

    owner_id: str
    at_step: int
    n_local_concepts: int
    n_peer_messages: int
    peer_ids: List[str]
    n_clusters_formed: int
    n_disagreements: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "at_step": self.at_step,
            "n_local_concepts": self.n_local_concepts,
            "n_peer_messages": self.n_peer_messages,
            "peer_ids": list(self.peer_ids),
            "n_clusters_formed": self.n_clusters_formed,
            "n_disagreements": self.n_disagreements,
        }
