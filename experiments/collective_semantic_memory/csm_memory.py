"""Minimal Collective Semantic Memory (CSM) — Phase 2 deliverable.

This is the first instantiation of the framework's title concept. The four
peer-broadcast architectures evaluated in the main benchmark sit at the
corners of a richer design space defined by three explicit rules:

    MERGE     — how peer evidence about the same place is combined.
    TRUST     — how conflicting or stale peer evidence is weighted.
    STALENESS — how consolidated structure decays or is invalidated.

The minimal CSM uses peer-broadcast at K=8 (the best fixed-cadence
peer configuration on the main sweep) and overlays:

  • Trust:     per-peer trust evolves as an EMA on retrieval-success
               consistency. Peers whose broadcasts have led to correct
               material-locks gain weight.
  • Staleness: each broadcast snapshot carries a tick stamp; weight
               decays exponentially with age (rate alpha = 0.05/tick).
  • Merge:     trust-weighted, staleness-discounted majority vote on
               cell tag at each candidate place, with a confidence
               threshold below which the place is dropped.

The runtime exposes the standard observe/tick/query interface so it is a
drop-in replacement for the four existing architectures in
experiments/big_experiment/memory_factory.py.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


WATER_TAG = "water_source"


@dataclass
class _Snapshot:
    """A peer's broadcast snapshot at one moment."""
    sender: str
    tick: int
    places: Dict[Tuple[int, int], Dict[str, float]]  # xy -> tag -> confidence


@dataclass
class _AgentState:
    aid: str
    own_evidence: Dict[Tuple[int, int], Dict[str, float]] = field(default_factory=dict)
    peer_snapshots: List[_Snapshot] = field(default_factory=list)
    trust: Dict[str, float] = field(default_factory=dict)  # peer_aid -> trust
    last_locked: Tuple[int, int] | None = None


class CSMMemory:
    """Collective Semantic Memory with explicit merge / trust / staleness."""

    def __init__(
        self,
        agent_ids: List[str],
        env_id: str = "default",
        broadcast_every_k: int = 8,
        staleness_alpha: float = 0.05,
        trust_ema: float = 0.10,
        initial_trust: float = 0.7,
        merge_threshold: float = 0.30,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.env_id = env_id
        self.broadcast_every_k = int(broadcast_every_k)
        self.staleness_alpha = float(staleness_alpha)
        self.trust_ema = float(trust_ema)
        self.initial_trust = float(initial_trust)
        self.merge_threshold = float(merge_threshold)
        self._tick = 0
        self._states: Dict[str, _AgentState] = {
            aid: _AgentState(aid=aid) for aid in agent_ids
        }
        for s in self._states.values():
            for peer in agent_ids:
                if peer != s.aid:
                    s.trust[peer] = self.initial_trust

    # ── adapter API ─────────────────────────────────────────────────

    def observe(self, agent_id: str, cells: List[Dict[str, Any]], tick: int) -> None:
        s = self._states[agent_id]
        for cell in cells:
            tag = cell.get("tag")
            if tag in (None, "wall", "safe_neutral"):
                continue
            xy = tuple(cell["xy"])
            d = s.own_evidence.setdefault(xy, {})
            d[tag] = max(d.get(tag, 0.0), 0.95)

    def tick(self, tick_idx: int) -> None:
        self._tick = tick_idx
        if self.broadcast_every_k > 0 and tick_idx % self.broadcast_every_k == 0:
            # All agents broadcast their current own_evidence snapshots.
            for sender_id in self.agent_ids:
                sender = self._states[sender_id]
                snapshot = _Snapshot(
                    sender=sender_id, tick=tick_idx,
                    places={xy: dict(tags) for xy, tags in sender.own_evidence.items()},
                )
                for receiver_id in self.agent_ids:
                    if receiver_id == sender_id:
                        continue
                    self._states[receiver_id].peer_snapshots.append(snapshot)
            # Prune very-old snapshots from each receiver (memory budget).
            cutoff = tick_idx - 10 * self.broadcast_every_k
            for s in self._states.values():
                s.peer_snapshots = [snp for snp in s.peer_snapshots if snp.tick >= cutoff]

    def query(self, agent_id: str) -> List[Tuple[float, float]]:
        s = self._states[agent_id]
        merged = self._merge(s)
        candidates = [
            xy for xy, score in merged.items() if score >= self.merge_threshold
        ]
        # Update trust after this query: peers whose snapshots support the
        # current top candidate gain trust (Hebbian-style EMA).
        if candidates:
            top = max(candidates, key=lambda xy: merged[xy])
            s.last_locked = top
            self._update_trust_from_top(s, top)
        return [(float(xy[0]), float(xy[1])) for xy in candidates]

    # ── merge / trust / staleness internals ─────────────────────────

    def _merge(self, s: _AgentState) -> Dict[Tuple[int, int], float]:
        """Trust-weighted, staleness-discounted score per place."""
        score: Dict[Tuple[int, int], float] = defaultdict(float)
        # Own evidence at full weight.
        for xy, tags in s.own_evidence.items():
            w = tags.get(WATER_TAG, 0.0)
            if w > 0:
                score[xy] += w
        # Peer evidence trust×staleness-weighted.
        for snp in s.peer_snapshots:
            age = max(0, self._tick - snp.tick)
            stale = math.exp(-self.staleness_alpha * age)
            trust = s.trust.get(snp.sender, self.initial_trust)
            for xy, tags in snp.places.items():
                w = tags.get(WATER_TAG, 0.0)
                if w > 0:
                    score[xy] += trust * stale * w
        return dict(score)

    def _update_trust_from_top(self, s: _AgentState, top_xy: Tuple[int, int]) -> None:
        """EMA-update peer trust: peers whose recent snapshots supported
        the current top candidate get +; others get small decay."""
        beta = self.trust_ema
        for snp in s.peer_snapshots[-len(self.agent_ids) :]:  # last broadcast round
            sender = snp.sender
            supported = top_xy in snp.places and snp.places[top_xy].get(WATER_TAG, 0.0) > 0
            target = 1.0 if supported else 0.4
            current = s.trust.get(sender, self.initial_trust)
            s.trust[sender] = (1 - beta) * current + beta * target
