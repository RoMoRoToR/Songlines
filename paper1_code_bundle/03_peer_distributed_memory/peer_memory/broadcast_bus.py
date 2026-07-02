"""BroadcastBus — a passive message router for peer-to-peer communication.

**The bus is NOT a coordinator.**  It has no business logic.  It does
not merge, aggregate, persist or interpret messages.  It is the
in-process analogue of a WiFi link layer: deliver bytes from one inbox
to another, that's all.

What it does
------------
- ``register(agent_id)``: create an empty ``PeerInbox`` for the agent
- ``broadcast(sender_id, message)``: put a copy of ``message`` into the
  inbox of every other registered agent
- ``inbox(agent_id)``: return the agent's inbox (the agent drains it
  on its own schedule)

What it explicitly does NOT do
------------------------------
- Compute any merged / consensus / aggregated view
- Store history (beyond per-agent buffers)
- Decide who talks to whom (that is the gossip protocol's job)
- Cache or deduplicate snapshots

This keeps the architecture honestly peer-to-peer: removing the bus
would break only the *delivery* of messages.  Agents would still
function in isolation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from peer_memory.peer_types import BroadcastMessage, PeerInbox


class BroadcastBus:
    """Passive in-process message router."""

    def __init__(self) -> None:
        self._inboxes: Dict[str, PeerInbox] = {}
        self._n_broadcasts: int = 0

    # ─────────────────────────────────────────────────── registry

    def register(self, agent_id: str) -> PeerInbox:
        if agent_id in self._inboxes:
            return self._inboxes[agent_id]
        inbox = PeerInbox(owner_id=agent_id)
        self._inboxes[agent_id] = inbox
        return inbox

    def unregister(self, agent_id: str) -> None:
        self._inboxes.pop(agent_id, None)

    def known_agents(self) -> List[str]:
        return list(self._inboxes.keys())

    # ─────────────────────────────────────────────────── delivery

    def broadcast(self, sender_id: str, message: BroadcastMessage) -> int:
        """Deliver ``message`` to every other registered agent's inbox.

        Returns the number of recipients.  Does NOT deliver to the sender.
        """
        if message.sender_id != sender_id:
            raise ValueError(
                f"sender_id mismatch: {sender_id} vs {message.sender_id}"
            )
        delivered = 0
        for aid, inbox in self._inboxes.items():
            if aid == sender_id:
                continue
            inbox.push(message)
            delivered += 1
        self._n_broadcasts += 1
        return delivered

    def inbox(self, agent_id: str) -> PeerInbox:
        if agent_id not in self._inboxes:
            raise KeyError(f"agent_id not registered: {agent_id}")
        return self._inboxes[agent_id]

    # ─────────────────────────────────────────────────── diagnostics

    @property
    def n_broadcasts(self) -> int:
        return self._n_broadcasts

    def pending_message_counts(self) -> Dict[str, int]:
        return {aid: len(inbox) for aid, inbox in self._inboxes.items()}
