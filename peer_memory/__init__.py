"""Peer-to-peer memory — no central aggregator.

Each agent has its own ``AgentMemory`` (private graph) plus its own
``AsymmetricTrust`` table.  Inter-agent contact happens only through a
passive ``BroadcastBus`` (in-process message router with no business
logic).  Every agent computes its OWN merged ``PeerView`` from its own
snapshot plus messages received from peers.

Two agents starting from identical snapshots may produce different
``PeerView``s because trust is asymmetric.  There is no single
"consensus report" — explicitly absent by design.

Compare to ``distributed_memory/`` (Variant C): same per-agent
isolation, but with a central ``ConsensusLayer`` that aggregates.
This package replaces that central layer with peer-to-peer gossip.
"""

from peer_memory.broadcast_bus import BroadcastBus
from peer_memory.peer_agent import PeerAgent
from peer_memory.peer_merge import local_merge
from peer_memory.peer_runtime import PeerRuntime
from peer_memory.peer_trust import AsymmetricTrust
from peer_memory.peer_types import (
    BroadcastMessage,
    PeerInbox,
    PeerMergeReport,
    PeerView,
)

__all__ = [
    "AsymmetricTrust",
    "BroadcastBus",
    "BroadcastMessage",
    "PeerAgent",
    "PeerInbox",
    "PeerMergeReport",
    "PeerRuntime",
    "PeerView",
    "local_merge",
]
