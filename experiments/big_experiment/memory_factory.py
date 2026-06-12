"""Memory layer factory + adapter for the universal runner.

For each architecture, we expose:
  - ``setup(agent_ids)`` — build the memory layer
  - ``observe(agent_id, obs, tick)`` — publish observations from this agent
  - ``tick(tick_idx)`` — internal refresh / broadcast
  - ``query(agent_id) -> List[(x, y)]`` — water targets visible to this agent

This lets the universal runner be variant-agnostic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import AgentSignature
from songline_drive.concept_recall import ConceptRecallLayer
from songline_drive.place_alignment import PlaceAlignmentEngine

from independent_memory import IndependentRuntime
from distributed_memory import DistributedRuntime
from peer_memory import PeerRuntime

WATER_TAG = "water_source"


def _make_alignment_engine() -> PlaceAlignmentEngine:
    return PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=2.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )


def _make_recall_layer() -> ConceptRecallLayer:
    return ConceptRecallLayer(
        _make_alignment_engine(),
        only_dominant_tag=False, min_concept_support=1,
        decay_engine=TemporalDecayEngine(),
        conflict_rules=ConflictRuleSet.songlines_default(),
    )


# ─────────────────────────────────────────────────────── INDEPENDENT


class _IndependentAdapter:
    name = "independent"

    def __init__(self, agent_ids: List[str], env_id: str):
        self.rt = IndependentRuntime(env_id=env_id)
        for aid in agent_ids:
            self.rt.spawn_agent(aid)

    def observe(self, agent_id, cells, tick):
        for cell in cells:
            tag = cell["tag"]
            if tag in ("wall", "safe_neutral"):
                continue
            self.rt.observe(agent_id, cell["xy"], {tag: 0.95},
                            episode_id=1, step_idx=tick, confidence=0.95)

    def tick(self, tick_idx):
        self.rt.tick()

    def query(self, agent_id) -> List[Tuple[float, float]]:
        results = self.rt.local_query(agent_id, WATER_TAG, top_k=5)
        return [r.centroid_xy for r in results if r.centroid_xy is not None]


# ─────────────────────────────────────────────────────── SHARED (songline_drive)


class _SharedAdapter:
    name = "shared"

    def __init__(self, agent_ids: List[str], env_id: str):
        self.env_id = env_id
        self.collective = CollectiveMemory(recency_lambda=0.97)
        for aid in agent_ids:
            self.collective.register_agent(
                AgentSignature(aid, role="agent", trust=1.0))
        self.recall = _make_recall_layer()
        self.graph = None

    def observe(self, agent_id, cells, tick):
        # Build a synthetic place_observed event with the most "informative" tag
        # from the visible cells (we publish each non-wall cell separately so the
        # shared graph correctly accumulates).
        for cell in cells:
            tag = cell["tag"]
            if tag in ("wall",):
                continue
            self.collective.publish_event(
                "place_observed", agent_id,
                episode_id=1, step_idx=tick, env_id=self.env_id,
                payload={"place_key": list(cell["xy"]),
                         "semantic_tags": {tag: 0.95},
                         "node_freshness": 1.0},
                confidence=0.95,
            )

    def tick(self, tick_idx):
        self.graph = self.recall.refresh(self.collective)

    def query(self, agent_id) -> List[Tuple[float, float]]:
        if self.graph is None:
            return []
        return [node.centroid_xy
                for node in self.graph.concepts.values()
                if node.dominant_tag == WATER_TAG and node.centroid_xy is not None]


# ─────────────────────────────────────────────────────── CENTRALIZED


class _CentralizedAdapter:
    name = "centralized"

    def __init__(self, agent_ids: List[str], env_id: str):
        self.rt = DistributedRuntime(env_id=env_id, consensus_radius=2.5)
        for aid in agent_ids:
            self.rt.spawn_agent(aid)

    def observe(self, agent_id, cells, tick):
        for cell in cells:
            tag = cell["tag"]
            if tag in ("wall", "safe_neutral"):
                continue
            self.rt.observe(agent_id, cell["xy"], {tag: 0.95},
                            episode_id=1, step_idx=tick, confidence=0.95)

    def tick(self, tick_idx):
        self.rt.tick()

    def query(self, agent_id) -> List[Tuple[float, float]]:
        report = self.rt.last_report
        if report is None:
            return []
        return [c.centroid_xy for c in report.distributed_concepts
                if c.consensus_dominant_tag == WATER_TAG]


# ─────────────────────────────────────────────────────── PEER


class _PeerAdapter:
    name = "peer"

    def __init__(self, agent_ids: List[str], env_id: str, *, broadcast_every_k: int):
        self.rt = PeerRuntime(
            env_id=env_id, broadcast_every_k=broadcast_every_k,
            consensus_radius=2.5,
        )
        for aid in agent_ids:
            self.rt.spawn_agent(aid)
        self.k = broadcast_every_k

    def observe(self, agent_id, cells, tick):
        for cell in cells:
            tag = cell["tag"]
            if tag in ("wall", "safe_neutral"):
                continue
            self.rt.observe(agent_id, cell["xy"], {tag: 0.95},
                            episode_id=1, step_idx=tick, confidence=0.95)

    def tick(self, tick_idx):
        self.rt.tick()

    def query(self, agent_id) -> List[Tuple[float, float]]:
        results = self.rt.peer_query(agent_id, WATER_TAG, top_k=5)
        return [c.centroid_xy for c in results]


# ─────────────────────────────────────────────────────── factory


class _CSMAdapter:
    """Minimal Collective Semantic Memory: peer-broadcast (K=8) + trust + staleness."""
    name = "csm"

    def __init__(self, agent_ids: List[str], env_id: str, *,
                 broadcast_every_k: int = 8):
        from experiments.collective_semantic_memory.csm_memory import CSMMemory
        self.mem = CSMMemory(
            agent_ids=agent_ids, env_id=env_id,
            broadcast_every_k=broadcast_every_k,
        )

    def observe(self, agent_id, cells, tick):
        self.mem.observe(agent_id, cells, tick)

    def tick(self, tick_idx):
        self.mem.tick(tick_idx)

    def query(self, agent_id) -> List[Tuple[float, float]]:
        return self.mem.query(agent_id)


def build_memory(
    architecture: str, agent_ids: List[str], env_id: str,
    *, broadcast_every_k: int = 4,
):
    if architecture == "independent":
        return _IndependentAdapter(agent_ids, env_id)
    if architecture == "shared":
        return _SharedAdapter(agent_ids, env_id)
    if architecture == "centralized":
        return _CentralizedAdapter(agent_ids, env_id)
    if architecture == "peer":
        return _PeerAdapter(agent_ids, env_id, broadcast_every_k=broadcast_every_k)
    if architecture == "csm":
        return _CSMAdapter(agent_ids, env_id, broadcast_every_k=broadcast_every_k)
    raise ValueError(f"Unknown architecture: {architecture}")
