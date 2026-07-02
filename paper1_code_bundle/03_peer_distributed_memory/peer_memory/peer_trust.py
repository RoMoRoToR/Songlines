"""Asymmetric pairwise trust — each agent has its own opinion of each peer.

Differs from ``distributed_memory.TrustModel`` (which is a global scalar
per agent) in two ways:

  1. **Asymmetric**: ``trust[A][B]`` need not equal ``trust[B][A]``.
  2. **Private**: each agent owns its own ``AsymmetricTrust`` instance.
     No two agents share this object.  There is no global table.

EMA-based outcome updates so trust shifts gradually when peers
contradict (or align with) the local agent's experience.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class AsymmetricTrust:
    """One agent's private trust table over the other agents it knows.

    Parameters
    ----------
    owner_id : str
        Identifier of the agent who owns this table.
    default_trust : float
        Initial trust assigned to a peer the first time it's encountered.
    trust_min, trust_max : float
        Hard clamps.  Trust never falls below ``trust_min`` so a peer
        is not permanently silenced.
    update_rate : float
        EMA step size for outcome updates.
    """

    def __init__(
        self,
        owner_id: str,
        *,
        default_trust: float = 0.7,
        trust_min: float = 0.10,
        trust_max: float = 1.00,
        update_rate: float = 0.10,
    ) -> None:
        self.owner_id = owner_id
        self.default_trust = float(default_trust)
        self.trust_min = float(trust_min)
        self.trust_max = float(trust_max)
        self.update_rate = float(update_rate)
        self._trust: Dict[str, float] = {}
        self._history: List[Dict] = []

    # ──────────────────────────────────────────────────────── read

    def trust_in(self, peer_id: str) -> float:
        """Return this agent's trust in ``peer_id``."""
        if peer_id == self.owner_id:
            return self.trust_max  # always trust self
        return self._trust.get(peer_id, self.default_trust)

    def known_peers(self) -> List[str]:
        return list(self._trust.keys())

    def snapshot(self) -> Dict[str, float]:
        return dict(self._trust)

    # ──────────────────────────────────────────────────────── write

    def set(self, peer_id: str, trust: float) -> None:
        if peer_id == self.owner_id:
            return  # don't set trust in self
        self._trust[peer_id] = self._clamp(trust)

    def update_from_outcome(
        self,
        peer_id: str,
        was_correct: bool,
        weight: float = 1.0,
    ) -> float:
        """EMA shift toward trust_max (correct) or trust_min (incorrect)."""
        if peer_id == self.owner_id:
            return self.trust_max
        target = self.trust_max if was_correct else self.trust_min
        current = self.trust_in(peer_id)
        step = self.update_rate * weight
        new = current + step * (target - current)
        new = self._clamp(new)
        self._trust[peer_id] = new
        self._history.append({
            "peer_id": peer_id, "was_correct": was_correct,
            "weight": weight, "from": current, "to": new,
        })
        return new

    def history(self) -> List[Dict]:
        return list(self._history)

    # ──────────────────────────────────────────────────────── internal

    def _clamp(self, value: float) -> float:
        return max(self.trust_min, min(self.trust_max, float(value)))
