"""Trust model — per-agent reliability scores used in consensus aggregation.

Trust is a scalar in ``[trust_min, trust_max]`` (default ``[0.1, 1.0]``).
It serves two purposes:

1. Weighting in ``ConsensusLayer.merge()`` — a high-trust agent's belief
   about a concept's tag pulls the consensus profile harder.
2. Outcome-driven update — when an agent's prediction is validated
   (or refuted) the trust shifts by an EMA-style step.

The model is intentionally lightweight: no bayesian machinery, no graph
of pairwise trusts.  One scalar per agent.  Callers are free to layer
something richer on top.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class TrustModel:
    """Scalar trust per agent with bounded EMA updates.

    Parameters
    ----------
    default_trust:
        Starting trust for any agent not explicitly set.
    trust_min, trust_max:
        Hard clamps on trust.  Trust never falls below ``trust_min`` so
        no agent is permanently silenced by a few bad outcomes.
    update_rate:
        Step size for outcome-driven updates.  ``0.10`` means a single
        validated outcome shifts trust ~10% of the way toward the target.
    """

    def __init__(
        self,
        default_trust: float = 0.8,
        trust_min: float = 0.10,
        trust_max: float = 1.00,
        update_rate: float = 0.10,
    ) -> None:
        self.default_trust = float(default_trust)
        self.trust_min = float(trust_min)
        self.trust_max = float(trust_max)
        self.update_rate = float(update_rate)
        self._trust: Dict[str, float] = {}
        self._history: List[Dict] = []

    def get(self, agent_id: str) -> float:
        return self._trust.get(agent_id, self.default_trust)

    def set(self, agent_id: str, trust: float) -> None:
        self._trust[agent_id] = self._clamp(trust)

    def register(self, agent_id: str, trust: Optional[float] = None) -> None:
        if agent_id not in self._trust:
            self._trust[agent_id] = self._clamp(
                self.default_trust if trust is None else trust
            )

    def update_from_outcome(
        self,
        agent_id: str,
        was_correct: bool,
        weight: float = 1.0,
    ) -> float:
        """EMA-style update toward 1.0 (correct) or trust_min (incorrect).

        ``weight`` lets callers attenuate uncertain outcomes (e.g.,
        partial validation).  Returns the new trust.
        """
        target = self.trust_max if was_correct else self.trust_min
        current = self.get(agent_id)
        step = self.update_rate * weight
        new = current + step * (target - current)
        new = self._clamp(new)
        self._trust[agent_id] = new
        self._history.append({
            "agent_id": agent_id,
            "was_correct": was_correct,
            "weight": weight,
            "from": current,
            "to": new,
        })
        return new

    def all_trusts(self) -> Dict[str, float]:
        return dict(self._trust)

    def history(self) -> List[Dict]:
        return list(self._history)

    def _clamp(self, value: float) -> float:
        return max(self.trust_min, min(self.trust_max, float(value)))
