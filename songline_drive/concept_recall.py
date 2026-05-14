"""Phase 2/3 — read-only collective recall layer.

``ConceptRecallLayer`` is the single integration point between the concept
graph and a live agent retrieval loop.  Phase 2 introduced the core
concept-augmented retrieval; Phase 3 adds:

- **Temporal decay filtering** (3a): concepts below ``deactivation_threshold``
  are suppressed from query results.  Optionally filter by decayed score.
- **Conflict penalty** (3b): concepts with competing tag evidence (e.g.
  water + hazard) are penalised before ranking.
- **Incremental refresh** (3c): ``refresh_incremental()`` delegates to
  ``IncrementalAlignmentEngine.update()`` instead of doing a full rebuild.

All Phase 2 public methods are fully backward-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from songline_drive.collective_concepts import (
    ConceptQueryResult,
    SharedConceptGraph,
)
from songline_drive.collective_memory import CollectiveMemory, _normalize_place_key
from songline_drive.collective_types import CollectiveQuery, CollectiveQueryResult
from songline_drive.place_alignment import IncrementalAlignmentEngine, PlaceAlignmentEngine


# ------------------------------------------------------------------ result type


@dataclass
class ConceptRecallResult:
    """Single concept returned by ``ConceptRecallLayer.query``."""

    concept_id: str
    dominant_tag: str
    score: float
    centroid_xy: Optional[Tuple[float, float]]
    member_places: List[Tuple[str, Tuple[Any, ...]]]  # [(env_id, place_key), ...]
    supporting_agents: List[str]
    is_cross_agent: bool
    per_tag_contribution: Dict[str, float] = field(default_factory=dict)

    def best_place_key(self) -> Optional[Tuple[Any, ...]]:
        """The first member place key (concept seeds are highest-support first)."""
        if self.member_places:
            return self.member_places[0][1]
        return None

    def env_id(self) -> Optional[str]:
        if self.member_places:
            return self.member_places[0][0]
        return None


# ------------------------------------------------------------------ main class


class ConceptRecallLayer:
    """Read-only bridge from SharedConceptGraph to the live retrieval loop.

    Usage::

        engine = PlaceAlignmentEngine(semantic_threshold=0.45, spatial_radius=4.0)
        recall = ConceptRecallLayer(engine)

        # after scouts finish publishing to CollectiveMemory:
        recall.refresh(collective)

        # consumer queries concept-augmented results:
        results = recall.query(
            target_tag="water_source",
            requesting_agent_id="consumer-C",
            env_id="grid-10x8",
        )

    Parameters
    ----------
    engine:
        ``PlaceAlignmentEngine`` instance (owns clustering hyperparameters).
    only_dominant_tag:
        Passed to ``SharedConceptGraph.query_concepts``; when ``True`` only
        concepts whose dominant tag equals the query tag are returned. This
        substantially improves precision by eliminating noise-contaminated
        concepts. Recommended for production retrieval; disable only for
        recall-sensitive fuzzy queries.
    min_concept_support:
        Minimum total support count across all member records for a concept
        to be returned. Filters one-shot noisy observations.
    require_cross_agent:
        When ``True``, only concepts supported by ≥2 agents are returned.
        Ensures the retrieval layer only uses collectively-validated knowledge.
    """

    def __init__(
        self,
        engine: PlaceAlignmentEngine,
        *,
        only_dominant_tag: bool = True,
        min_concept_support: int = 2,
        require_cross_agent: bool = False,
        decay_engine: Optional[Any] = None,      # TemporalDecayEngine (Phase 3a)
        conflict_rules: Optional[Any] = None,    # ConflictRuleSet (Phase 3b)
        max_conflict_penalty: float = 1.0,       # 1.0 = no filtering by conflict
    ) -> None:
        self.engine = engine
        self.only_dominant_tag = bool(only_dominant_tag)
        self.min_concept_support = int(min_concept_support)
        self.require_cross_agent = bool(require_cross_agent)
        self.decay_engine = decay_engine    # Phase 3a
        self.conflict_rules = conflict_rules  # Phase 3b
        self.max_conflict_penalty = float(max_conflict_penalty)
        self._graph: Optional[SharedConceptGraph] = None
        self._last_seq: int = -1
        self._incremental_stats: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- rebuild

    def refresh(self, collective: CollectiveMemory) -> SharedConceptGraph:
        """Full rebuild of the concept graph from ``collective``'s current state."""
        self._graph = self.engine.build(collective)
        self._last_seq = collective._next_seq  # noqa: SLF001
        # Phase 3a: write decayed freshness into each concept
        if self.decay_engine is not None:
            self.decay_engine.apply_to_graph(self._graph, self._last_seq)
        # Phase 3b: write conflict_score into each concept
        if self.conflict_rules is not None:
            self.conflict_rules.apply_to_graph(self._graph)
        return self._graph

    @property
    def graph(self) -> Optional[SharedConceptGraph]:
        return self._graph

    def refresh_incremental(
        self,
        collective: CollectiveMemory,
    ) -> Tuple[SharedConceptGraph, Dict[str, Any]]:
        """Phase 3c: incremental update instead of full rebuild.

        Requires ``self.engine`` to be an ``IncrementalAlignmentEngine``.
        Falls back to a full rebuild via ``refresh()`` if the engine does
        not support incremental updates.

        Returns ``(graph, stats)`` where ``stats`` is the update info dict
        from ``IncrementalAlignmentEngine.update()``.
        """
        if isinstance(self.engine, IncrementalAlignmentEngine):
            graph, stats = self.engine.update(collective)
            self._graph = graph
            self._last_seq = collective._next_seq  # noqa: SLF001
            self._incremental_stats = stats
            # Apply Phase 3a decay and 3b conflict to updated graph
            if self.decay_engine is not None:
                self.decay_engine.apply_to_graph(graph, collective._next_seq)  # noqa: SLF001
            if self.conflict_rules is not None:
                self.conflict_rules.apply_to_graph(graph)
            return graph, stats
        # Fallback for non-incremental engines
        self.refresh(collective)
        return self._graph, {"first_build": False, "fallback_full_rebuild": True}

    @property
    def incremental_stats(self) -> Optional[Dict[str, Any]]:
        """Stats from the most recent incremental update, or None."""
        return self._incremental_stats

    def is_stale(self, collective: CollectiveMemory) -> bool:
        """True if new events have been published since the last refresh."""
        return collective._next_seq != self._last_seq  # noqa: SLF001

    # ----------------------------------------------------------------- query

    def query(
        self,
        target_tag: str,
        requesting_agent_id: str,
        env_id: Optional[str] = None,
        *,
        score_weights: Optional[Dict[str, float]] = None,
        penalty_weights: Optional[Dict[str, float]] = None,
        top_k: int = 3,
        current_seq: Optional[int] = None,
    ) -> List[ConceptRecallResult]:
        """Query concept graph for the best concepts matching ``target_tag``.

        Phase 3 additions:
        - ``decay_engine``: stale concepts (below deactivation_threshold) are
          filtered out before ranking if ``current_seq`` is provided.
        - ``conflict_rules``: concept scores are penalised by conflict before
          final ranking.

        Returns an empty list if the graph has not been built yet or no
        concepts match.
        """
        if self._graph is None:
            return []

        min_agents = 2 if self.require_cross_agent else 1
        raw: List[ConceptQueryResult] = self._graph.query_concepts(
            target_tag=target_tag,
            score_weights=score_weights,
            penalty_weights=penalty_weights,
            requesting_agent_id=requesting_agent_id,
            exclude_self=False,
            min_support_count=self.min_concept_support,
            min_supporting_agents=min_agents,
            only_dominant_tag=self.only_dominant_tag,
            top_k=top_k * 3,  # fetch extra to allow for Phase 3 filtering
        )

        # Phase 3a: filter stale concepts
        if self.decay_engine is not None and current_seq is not None:
            raw = [
                r for r in raw
                if r.concept_id in self._graph.concepts
                and self.decay_engine.is_active(
                    self._graph.concepts[r.concept_id], current_seq
                )
            ]

        # Phase 3b: apply conflict penalty to scores, filter by max_conflict_penalty
        if self.conflict_rules is not None:
            adjusted: List[ConceptQueryResult] = []
            for r in raw:
                concept = self._graph.concepts.get(r.concept_id)
                if concept is None:
                    continue
                penalty = self.conflict_rules.conflict_penalty(concept)
                if penalty > self.max_conflict_penalty:
                    continue
                # Build adjusted result (mutate a copy of the score)
                from dataclasses import replace as _replace
                try:
                    adj_score = self.conflict_rules.adjusted_score(concept, r.score)
                    adjusted.append(_replace(r, score=adj_score))
                except TypeError:
                    # dataclass without frozen; create manually
                    from songline_drive.collective_concepts import ConceptQueryResult
                    adjusted.append(ConceptQueryResult(
                        concept_id=r.concept_id,
                        dominant_tag=r.dominant_tag,
                        score=self.conflict_rules.adjusted_score(concept, r.score),
                        supporting_agents=r.supporting_agents,
                        member_count=r.member_count,
                        per_tag_contribution=r.per_tag_contribution,
                        used_other_agent_knowledge=r.used_other_agent_knowledge,
                        centroid_xy=r.centroid_xy,
                    ))
            adjusted.sort(key=lambda x: x.score, reverse=True)
            raw = adjusted[:top_k]
        else:
            raw = raw[:top_k]

        results: List[ConceptRecallResult] = []
        for res in raw:
            node = self._graph.concepts.get(res.concept_id)
            if node is None:
                continue
            # Filter by env_id if specified
            members = list(node.member_place_keys)
            if env_id is not None:
                members = [(e, k) for e, k in members if e == env_id]
            if not members:
                continue
            results.append(
                ConceptRecallResult(
                    concept_id=res.concept_id,
                    dominant_tag=res.dominant_tag,
                    score=res.score,
                    centroid_xy=res.centroid_xy,
                    member_places=members,
                    supporting_agents=list(res.supporting_agents),
                    is_cross_agent=res.used_other_agent_knowledge,
                    per_tag_contribution=dict(res.per_tag_contribution),
                )
            )
        return results

    # --------------------------------------------------------- bridge to Phase 1

    def to_collective_results(
        self,
        recall_results: List[ConceptRecallResult],
        target_tag: str,
        requesting_agent_id: str,
    ) -> List[CollectiveQueryResult]:
        """Convert concept recall results to Phase 1 ``CollectiveQueryResult`` format.

        This bridge allows downstream planner code that expects
        ``CollectiveQueryResult`` to consume concept-level answers without
        any code change. The ``per_tag_fused`` dict carries a
        ``_concept_id`` sentinel so callers can optionally inspect which
        concept backed the result.
        """
        out: List[CollectiveQueryResult] = []
        for res in recall_results:
            for env_id, place_key in res.member_places:
                per_tag = dict(res.per_tag_contribution)
                per_tag["_concept_id"] = res.concept_id  # type: ignore[assignment]
                out.append(
                    CollectiveQueryResult(
                        place_key=place_key,
                        env_id=env_id,
                        fused_score=res.score,
                        contributing_agents=list(res.supporting_agents),
                        contributing_event_seqs=[],
                        used_other_agent_knowledge=res.is_cross_agent,
                        target_tag=target_tag,
                        per_tag_fused=per_tag,
                    )
                )
        # Sort by score desc, deduplicate place_keys
        seen: Set[Tuple[str, Tuple[Any, ...]]] = set()
        deduped: List[CollectiveQueryResult] = []
        for r in sorted(out, key=lambda x: x.fused_score, reverse=True):
            key = (r.env_id, r.place_key)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    # ----------------------------------------------- query with fallback

    def query_collective(
        self,
        collective: CollectiveMemory,
        query: CollectiveQuery,
        top_k: int = 5,
        *,
        fallback_to_raw: bool = True,
    ) -> Tuple[List[CollectiveQueryResult], str]:
        """Primary integration method for live retrieval loop.

        1. Query concept graph with ``only_dominant_tag=True``.
        2. Convert results to ``CollectiveQueryResult`` format.
        3. If no concept matched and ``fallback_to_raw`` is True, fall back
           to Phase 1 ``query_collective_nodes``.

        Returns ``(results, source)`` where ``source`` is one of:
        ``"concept_recall"`` | ``"raw_fallback"`` | ``"empty"``.
        """
        recall_results = self.query(
            target_tag=query.target_tag,
            requesting_agent_id=query.requesting_agent_id,
            env_id=query.env_id,
            score_weights=query.score_weights or None,
            penalty_weights=query.penalty_weights or None,
            top_k=top_k,
        )

        if recall_results:
            converted = self.to_collective_results(
                recall_results,
                target_tag=query.target_tag,
                requesting_agent_id=query.requesting_agent_id,
            )
            if converted:
                # Log the reads so Phase 1 metrics still capture this
                collective._reads_log.append({  # noqa: SLF001
                    "requesting_agent_id": query.requesting_agent_id,
                    "intent_type": query.intent_type,
                    "target_tag": query.target_tag,
                    "place_key": converted[0].place_key,
                    "env_id": converted[0].env_id,
                    "fused_score": converted[0].fused_score,
                    "contributing_agents": list(converted[0].contributing_agents),
                    "used_other_agent_knowledge": converted[0].used_other_agent_knowledge,
                    "wall_clock_seq": collective._next_seq,  # noqa: SLF001
                    "source": "concept_recall",
                })
                return converted[:top_k], "concept_recall"

        if fallback_to_raw:
            raw = collective.query_collective_nodes(query, top_k=top_k)
            return raw, "raw_fallback"

        return [], "empty"
