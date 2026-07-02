"""Phase 2 — canonical shared place concepts.

A SharedConceptNode is what the team collectively believes is a single
recurring *kind of place* — e.g. ``water_source_cluster_0``,
``hazard_band_2``, ``post_hazard_rejoin_corridor_4``. Membership is
across-agents and across-episodes: scout-A's water observation at
``(5,3)`` and scout-B's water observation at ``(5,4)`` should land in
the same concept once they look similar enough.

Phase 2 deliberately stays a thin layer on top of Phase 1 event bus:
the canonical concept graph is *derived* from ``CollectiveMemory``
state, not written to instead of it. The Phase 1 ``place_beliefs``
remain the system of record; concepts are a consolidated view.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PlaceKey = Tuple[str, Tuple[Any, ...]]


@dataclass
class SharedConceptNode:
    concept_id: str
    dominant_tag: str
    semantic_profile: Dict[str, float] = field(default_factory=dict)
    member_place_keys: List[PlaceKey] = field(default_factory=list)
    supporting_agents: Set[str] = field(default_factory=set)
    episode_signatures: Set[Tuple[str, int]] = field(default_factory=set)
    support_count: int = 0
    freshness: float = 0.0
    confidence: float = 0.0
    conflict_score: float = 0.0
    centroid_xy: Optional[Tuple[float, float]] = None
    radius_xy: float = 0.0
    last_seen_seq: int = -1

    def member_count(self) -> int:
        return len(self.member_place_keys)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "dominant_tag": self.dominant_tag,
            "semantic_profile": dict(self.semantic_profile),
            "member_place_keys": [
                {"env_id": env, "place_key": list(key)}
                for env, key in self.member_place_keys
            ],
            "supporting_agents": sorted(self.supporting_agents),
            "episode_signatures": [
                {"agent_id": a, "episode_id": e}
                for a, e in sorted(self.episode_signatures)
            ],
            "support_count": self.support_count,
            "freshness": self.freshness,
            "confidence": self.confidence,
            "conflict_score": self.conflict_score,
            "centroid_xy": list(self.centroid_xy) if self.centroid_xy is not None else None,
            "radius_xy": self.radius_xy,
            "last_seen_seq": self.last_seen_seq,
        }


@dataclass
class SharedConceptEdge:
    src_concept_id: str
    dst_concept_id: str
    support: int = 0
    supporting_agents: Set[str] = field(default_factory=set)
    last_seen_seq: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_concept_id": self.src_concept_id,
            "dst_concept_id": self.dst_concept_id,
            "support": self.support,
            "supporting_agents": sorted(self.supporting_agents),
            "last_seen_seq": self.last_seen_seq,
        }


@dataclass
class ConceptQueryResult:
    concept_id: str
    dominant_tag: str
    score: float
    supporting_agents: List[str]
    member_count: int
    per_tag_contribution: Dict[str, float] = field(default_factory=dict)
    used_other_agent_knowledge: bool = False
    centroid_xy: Optional[Tuple[float, float]] = None


class SharedConceptGraph:
    """Derived collective view over ``CollectiveMemory.place_beliefs``.

    Owns concepts (place clusters), edges (concept-to-concept transitions)
    and the place→concept index. Built by ``PlaceAlignmentEngine``;
    queried independently of the event bus.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, SharedConceptNode] = {}
        self.edges: Dict[Tuple[str, str], SharedConceptEdge] = {}
        self.place_to_concept: Dict[PlaceKey, str] = {}
        self._tag_counter: Dict[str, int] = {}

    # ------------------------------------------------------------------ build

    def new_concept_id(self, dominant_tag: str) -> str:
        tag_slug = str(dominant_tag).replace("/", "_") or "concept"
        idx = self._tag_counter.get(tag_slug, 0)
        self._tag_counter[tag_slug] = idx + 1
        return f"{tag_slug}_cluster_{idx}"

    def register_concept(self, concept: SharedConceptNode) -> None:
        self.concepts[concept.concept_id] = concept
        for place_key in concept.member_place_keys:
            self.place_to_concept[place_key] = concept.concept_id

    def attach_place(
        self,
        concept_id: str,
        place_key: PlaceKey,
        agent_ids: Iterable[str] = (),
        episode_signatures: Iterable[Tuple[str, int]] = (),
    ) -> None:
        concept = self.concepts[concept_id]
        if place_key not in concept.member_place_keys:
            concept.member_place_keys.append(place_key)
        concept.supporting_agents.update(agent_ids)
        concept.episode_signatures.update(episode_signatures)
        self.place_to_concept[place_key] = concept_id

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        agent_id: str,
        seq: int,
    ) -> None:
        key = (src_id, dst_id)
        edge = self.edges.get(key)
        if edge is None:
            edge = SharedConceptEdge(src_concept_id=src_id, dst_concept_id=dst_id)
            self.edges[key] = edge
        edge.support += 1
        edge.supporting_agents.add(agent_id)
        edge.last_seen_seq = max(edge.last_seen_seq, int(seq))

    # ------------------------------------------------------------------ views

    def concepts_by_tag(self, dominant_tag: str) -> List[SharedConceptNode]:
        return [c for c in self.concepts.values() if c.dominant_tag == dominant_tag]

    def concepts_supported_by(self, agent_id: str) -> List[SharedConceptNode]:
        return [c for c in self.concepts.values() if agent_id in c.supporting_agents]

    def neighbors(self, concept_id: str) -> List[SharedConceptEdge]:
        return [e for (src, dst), e in self.edges.items() if src == concept_id]

    def concept_for_place(self, place_key: PlaceKey) -> Optional[SharedConceptNode]:
        concept_id = self.place_to_concept.get(place_key)
        if concept_id is None:
            return None
        return self.concepts.get(concept_id)

    def stats(self) -> Dict[str, Any]:
        per_tag: Dict[str, int] = {}
        for concept in self.concepts.values():
            per_tag[concept.dominant_tag] = per_tag.get(concept.dominant_tag, 0) + 1
        member_total = sum(c.member_count() for c in self.concepts.values())
        return {
            "n_concepts": len(self.concepts),
            "n_edges": len(self.edges),
            "n_member_places": member_total,
            "concepts_per_tag": per_tag,
        }

    # ---------------------------------------------------------------- queries

    def query_concepts(
        self,
        *,
        target_tag: str,
        score_weights: Optional[Dict[str, float]] = None,
        penalty_weights: Optional[Dict[str, float]] = None,
        requesting_agent_id: Optional[str] = None,
        exclude_self: bool = False,
        min_support_count: int = 1,
        min_supporting_agents: int = 1,
        only_dominant_tag: bool = False,
        top_k: int = 5,
    ) -> List[ConceptQueryResult]:
        """Query concepts scored against ``target_tag``.

        Parameters
        ----------
        only_dominant_tag:
            When ``True``, only concepts whose ``dominant_tag`` equals
            ``target_tag`` are considered. This eliminates noise-contaminated
            concepts that have a tiny cross-tag signal but are semantically
            something else, giving higher-precision results at the cost of
            recall on genuinely ambiguous places.
        """
        score_weights = dict(score_weights or {target_tag: 1.0})
        penalty_weights = dict(penalty_weights or {})
        results: List[ConceptQueryResult] = []
        for concept in self.concepts.values():
            if only_dominant_tag and concept.dominant_tag != target_tag:
                continue
            if exclude_self and requesting_agent_id is not None:
                others = concept.supporting_agents - {requesting_agent_id}
                if not others:
                    continue
                supporting_size = len(others)
            else:
                supporting_size = len(concept.supporting_agents)
            if concept.support_count < min_support_count:
                continue
            if supporting_size < min_supporting_agents:
                continue
            score = 0.0
            per_tag: Dict[str, float] = {}
            for tag, weight in score_weights.items():
                value = float(concept.semantic_profile.get(tag, 0.0))
                if value <= 0.0:
                    continue
                per_tag[tag] = value
                score += float(weight) * value
            for tag, weight in penalty_weights.items():
                value = float(concept.semantic_profile.get(tag, 0.0))
                if value <= 0.0:
                    continue
                per_tag[f"-{tag}"] = value
                score -= float(weight) * value
            if score <= 0.0:
                continue
            used_other = bool(
                requesting_agent_id is not None
                and (concept.supporting_agents - {requesting_agent_id})
            )
            results.append(
                ConceptQueryResult(
                    concept_id=concept.concept_id,
                    dominant_tag=concept.dominant_tag,
                    score=score,
                    supporting_agents=sorted(concept.supporting_agents),
                    member_count=concept.member_count(),
                    per_tag_contribution=per_tag,
                    used_other_agent_knowledge=used_other,
                    centroid_xy=concept.centroid_xy,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: max(0, int(top_k))]

    # ------------------------------------------------------------ persistence

    def export(self, out_dir: str, filename_prefix: str = "collective_concepts") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        concepts_path = os.path.join(out_dir, f"{filename_prefix}_concepts.json")
        edges_path = os.path.join(out_dir, f"{filename_prefix}_edges.json")
        index_path = os.path.join(out_dir, f"{filename_prefix}_place_index.json")
        stats_path = os.path.join(out_dir, f"{filename_prefix}_stats.json")

        with open(concepts_path, "w", encoding="utf-8") as fh:
            json.dump(
                [c.to_dict() for c in self.concepts.values()],
                fh,
                ensure_ascii=False,
                indent=2,
            )
        with open(edges_path, "w", encoding="utf-8") as fh:
            json.dump(
                [e.to_dict() for e in self.edges.values()],
                fh,
                ensure_ascii=False,
                indent=2,
            )
        index_payload = {
            f"{env}::{list(key)}": cid
            for (env, key), cid in self.place_to_concept.items()
        }
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(index_payload, fh, ensure_ascii=False, indent=2)
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(self.stats(), fh, ensure_ascii=False, indent=2)

        return {
            "concepts_json": concepts_path,
            "edges_json": edges_path,
            "place_index_json": index_path,
            "stats_json": stats_path,
        }


# -------------------------------------------------------------------- helpers


def normalize_profile(profile: Dict[str, float]) -> Dict[str, float]:
    if not profile:
        return {}
    norm = math.sqrt(sum(v * v for v in profile.values()))
    if norm <= 0.0:
        return {}
    return {k: float(v) / norm for k, v in profile.items() if v > 0.0}


def cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    nl = normalize_profile(left)
    nr = normalize_profile(right)
    if not nl or not nr:
        return 0.0
    keys = set(nl.keys()) & set(nr.keys())
    return float(sum(nl[k] * nr[k] for k in keys))


def profile_entropy(profile: Dict[str, float]) -> float:
    positive = [float(v) for v in profile.values() if v > 0.0]
    total = sum(positive)
    if total <= 0.0 or len(positive) <= 1:
        return 0.0
    entropy = 0.0
    for value in positive:
        p = value / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return entropy


def euclidean_xy(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)
