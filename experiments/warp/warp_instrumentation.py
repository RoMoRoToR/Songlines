"""Provenance extraction — phi at the moment of an M*-lock.

phi(m) = foreign evidence mass / total evidence mass, where the masses
are THE SAME weights the architecture's merge uses (design §3.1):

    peer / centralized :  trust · log1p(support)      (peer_merge.local_merge)
    csm                :  trust · exp(−α·age) · conf  (CSMMemory._merge)
    shared             :  per-agent observation counts (all trust = 1)
    independent        :  all mass is own; phi ≡ 0

Nothing in the memory layers is rewritten: for peer/centralized the
per-source weights are reconstructed exactly from ``AgentContribution``
(trust, local_support) that ``local_merge`` already exposes; for CSM the
snapshot pool is read directly.  This module only READS state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

XY_TOL = 0.6  # same closeness tolerance as the Q/R/M/C runner


def _xy_close(a, b, tol: float = XY_TOL) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ────────────────────────────────────────────── observation ledger


@dataclass
class ProvenanceLedger:
    """Who observed which water cell, when.

    Fed by the warp runner on every ``memory.observe`` call.  Used for:
      - the strict-W* cross-check (``self_ever_observed``)
      - foreign-source age for architectures without explicit snapshots
      - phi for the shared bus (where all trust weights are 1)
    """

    water_tag: str = "water_source"
    # (x, y) -> agent_id -> {"count": int, "first_tick": int, "last_tick": int}
    cells: Dict[Tuple[int, int], Dict[str, Dict[str, int]]] = field(
        default_factory=dict)

    def record(self, agent_id: str, cells: List[Dict[str, Any]],
               tick: int) -> None:
        for cell in cells:
            if cell.get("tag") != self.water_tag:
                continue
            xy = (int(cell["xy"][0]), int(cell["xy"][1]))
            per_agent = self.cells.setdefault(xy, {})
            rec = per_agent.setdefault(
                agent_id, {"count": 0, "first_tick": tick, "last_tick": tick})
            rec["count"] += 1
            rec["last_tick"] = tick

    def _matching(self, target_xy) -> List[Tuple[Tuple[int, int], Dict]]:
        return [(xy, per_agent) for xy, per_agent in self.cells.items()
                if _xy_close(xy, target_xy)]

    def self_observed(self, agent_id: str, target_xy) -> bool:
        return any(agent_id in per_agent
                   for _, per_agent in self._matching(target_xy))

    def counts_by_agent(self, target_xy) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _, per_agent in self._matching(target_xy):
            for aid, rec in per_agent.items():
                out[aid] = out.get(aid, 0) + rec["count"]
        return out

    def freshest_foreign_age(self, agent_id: str, target_xy,
                             tick: int) -> Optional[float]:
        ages = [tick - rec["last_tick"]
                for _, per_agent in self._matching(target_xy)
                for aid, rec in per_agent.items() if aid != agent_id]
        return float(min(ages)) if ages else None


# ────────────────────────────────────────────── provenance result


@dataclass
class CandidateProvenance:
    phi: float
    own_mass: float
    foreign_mass: float
    per_source_mass: Dict[str, float]
    source_snapshot_age: Optional[float]
    co_recipients: int


def _from_masses(per_source: Dict[str, float], agent_id: str,
                 snapshot_age: Optional[float],
                 co_recipients: int) -> CandidateProvenance:
    own = per_source.get(agent_id, 0.0)
    foreign = sum(v for k, v in per_source.items() if k != agent_id)
    total = own + foreign
    phi = (foreign / total) if total > 0 else 0.0
    return CandidateProvenance(
        phi=phi, own_mass=own, foreign_mass=foreign,
        per_source_mass=dict(per_source),
        source_snapshot_age=snapshot_age,
        co_recipients=co_recipients,
    )


# ────────────────────────────────────────────── per-architecture


def _peer_provenance(adapter, agent_id: str, target_xy,
                     tick: int) -> Optional[CandidateProvenance]:
    agent = adapter.rt.agent(agent_id)
    concept = None
    for c in agent.peer_view.distributed_concepts:
        if c.centroid_xy is not None and _xy_close(c.centroid_xy, target_xy):
            if concept is None or c.consensus_confidence > concept.consensus_confidence:
                concept = c
    if concept is None:
        return None

    # Exact reconstruction of local_merge's aggregation weight:
    # w_combined = trust · log1p(support)
    per_source: Dict[str, float] = {}
    for contrib in concept.contributions:
        w = contrib.trust * math.log1p(max(1, contrib.local_support))
        per_source[contrib.agent_id] = per_source.get(contrib.agent_id, 0.0) + w

    # Age of the freshest foreign snapshot that contributed.  PeerRuntime's
    # internal tick counter runs one ahead of the runner's 0-based tick, so
    # age is computed against the runtime clock and clamped at 0.
    rt_tick = getattr(adapter.rt, "tick_count", tick)
    ages = []
    for peer_id, msg in getattr(agent, "_last_known", {}).items():
        if peer_id != agent_id and peer_id in per_source:
            ages.append(max(0, rt_tick - msg.sent_at_step))
    snapshot_age = float(min(ages)) if ages else None

    n = len(adapter.rt.all_agents())
    return _from_masses(per_source, agent_id, snapshot_age,
                        co_recipients=max(0, n - 2))


def _centralized_provenance(adapter, agent_id: str, target_xy,
                            tick: int) -> Optional[CandidateProvenance]:
    report = adapter.rt.last_report
    if report is None:
        return None
    concept = None
    for c in report.distributed_concepts:
        if c.centroid_xy is not None and _xy_close(c.centroid_xy, target_xy):
            if concept is None or c.consensus_confidence > concept.consensus_confidence:
                concept = c
    if concept is None:
        return None
    per_source: Dict[str, float] = {}
    for contrib in concept.contributions:
        w = contrib.trust * math.log1p(max(1, contrib.local_support))
        per_source[contrib.agent_id] = per_source.get(contrib.agent_id, 0.0) + w
    n = len(concept.contributions)
    # Central merge recomputes every tick — evidence age is not snapshot-bound.
    return _from_masses(per_source, agent_id, None,
                        co_recipients=max(0, len(per_source)))


def _csm_provenance(adapter, agent_id: str, target_xy,
                    tick: int) -> Optional[CandidateProvenance]:
    mem = adapter.mem
    state = mem._states[agent_id]
    from experiments.collective_semantic_memory.csm_memory import WATER_TAG

    target_cell = (int(round(target_xy[0])), int(round(target_xy[1])))
    per_source: Dict[str, float] = {}

    own_w = state.own_evidence.get(target_cell, {}).get(WATER_TAG, 0.0)
    if own_w > 0:
        per_source[agent_id] = own_w

    freshest_age: Optional[float] = None
    for snp in state.peer_snapshots:
        w = snp.places.get(target_cell, {}).get(WATER_TAG, 0.0)
        if w <= 0:
            continue
        age = max(0, mem._tick - snp.tick)
        stale = math.exp(-mem.staleness_alpha * age)
        trust = state.trust.get(snp.sender, mem.initial_trust)
        mass = trust * stale * w
        per_source[snp.sender] = per_source.get(snp.sender, 0.0) + mass
        if snp.sender != agent_id:
            if freshest_age is None or age < freshest_age:
                freshest_age = float(age)

    if not per_source:
        return None
    n = len(mem.agent_ids)
    return _from_masses(per_source, agent_id, freshest_age,
                        co_recipients=max(0, n - 2))


def _shared_provenance(adapter, agent_id: str, target_xy, tick: int,
                       ledger: ProvenanceLedger,
                       n_agents: int) -> Optional[CandidateProvenance]:
    counts = ledger.counts_by_agent(target_xy)
    if not counts:
        return None
    per_source = {aid: float(c) for aid, c in counts.items()}
    age = ledger.freshest_foreign_age(agent_id, target_xy, tick)
    return _from_masses(per_source, agent_id, age,
                        co_recipients=max(0, n_agents - 1))


def _independent_provenance(agent_id: str) -> CandidateProvenance:
    return CandidateProvenance(
        phi=0.0, own_mass=1.0, foreign_mass=0.0,
        per_source_mass={agent_id: 1.0},
        source_snapshot_age=None, co_recipients=0,
    )


# ────────────────────────────────────────────── dispatch


def candidate_provenance(
    memory_adapter, agent_id: str, target_xy,
    ledger: ProvenanceLedger, tick: int,
    n_agents: int = 0,
) -> CandidateProvenance:
    """Provenance of the candidate the planner locked, at lock time.

    Falls back to the observation ledger when the architecture-specific
    view cannot resolve the concept (e.g. centroid drifted past the
    matching tolerance) — the ledger is a superset of what every merge
    consumed, so the fallback is conservative.
    """
    name = getattr(memory_adapter, "name", "unknown")
    result: Optional[CandidateProvenance] = None
    if name == "independent":
        result = _independent_provenance(agent_id)
    elif name == "peer":
        result = _peer_provenance(memory_adapter, agent_id, target_xy, tick)
    elif name == "centralized":
        result = _centralized_provenance(memory_adapter, agent_id, target_xy, tick)
    elif name == "csm":
        result = _csm_provenance(memory_adapter, agent_id, target_xy, tick)
    elif name == "shared":
        result = _shared_provenance(memory_adapter, agent_id, target_xy,
                                    tick, ledger, n_agents)

    if result is None:
        counts = ledger.counts_by_agent(target_xy)
        if counts:
            per_source = {aid: float(c) for aid, c in counts.items()}
            result = _from_masses(
                per_source, agent_id,
                ledger.freshest_foreign_age(agent_id, target_xy, tick),
                co_recipients=max(0, len(counts) - 1))
        else:
            result = _independent_provenance(agent_id)
    return result


def mask_foreign_targets(
    targets: List[Tuple[float, float]], agent_id: str,
    ledger: ProvenanceLedger,
) -> List[Tuple[float, float]]:
    """Counterfactual mask (W1): zero out foreign contribution at merge.

    A candidate survives the mask iff the agent itself has observed a
    matching water cell — i.e. exactly the evidence the agent would
    still hold with all foreign snapshots removed.
    """
    return [t for t in targets if ledger.self_observed(agent_id, t)]
