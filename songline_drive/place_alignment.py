"""Phase 2 — PlaceAlignmentEngine.

Reads ``CollectiveMemory.place_beliefs`` and builds a ``SharedConceptGraph``
by greedy clustering on three jointly-evaluated criteria:

1. **Dominant-tag agreement** — two places that share the same highest-confidence
   tag get a ``tag_match_bonus`` added to their similarity score.
2. **Semantic cosine similarity** — the fused tag-confidence profiles of the
   two places are compared; score ≥ ``semantic_threshold`` is required to merge.
3. **Spatial proximity** — if both places expose (x, y) coordinates in their
   place_key, the Euclidean distance must be ≤ ``spatial_radius``.

Design constraints:
- **Read-only** access to ``CollectiveMemory``. Phase 1 state is never mutated.
- **No external dependencies** beyond stdlib + ``collective_concepts.py``.
- **Deterministic** given the same ``CollectiveMemory`` state (sort-stable).
- Places from different ``env_id``s never share a concept by default
  (``cross_env=False``); Phase 4+ may relax this for transfer scenarios.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet

from songline_drive.collective_concepts import (
    SharedConceptGraph,
    SharedConceptNode,
    cosine_similarity,
    euclidean_xy,
    profile_entropy,
)
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import BeliefRecord


# ------------------------------------------------------------------ helpers


def _try_xy(place_key: Tuple[Any, ...]) -> Optional[Tuple[float, float]]:
    """Extract (x, y) from a grid-cell place_key; returns None if not possible."""
    if len(place_key) >= 2:
        try:
            return (float(place_key[0]), float(place_key[1]))
        except (TypeError, ValueError):
            pass
    return None


def _centroid(xys: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not xys:
        return (0.0, 0.0)
    return (
        sum(x for x, _ in xys) / len(xys),
        sum(y for _, y in xys) / len(xys),
    )


def _fuse_profile(
    tag_records: Dict[str, List[BeliefRecord]],
    recency_lambda: float,
    cur_seq: int,
) -> Dict[str, float]:
    """Recency-decayed fused confidence per tag (same formula as CollectiveMemory)."""
    profile: Dict[str, float] = {}
    for tag, records in tag_records.items():
        if not records:
            continue
        total = 0.0
        norm = 0.0
        for record in records:
            recency = recency_lambda ** max(0, cur_seq - record.wall_clock_seq)
            total += record.confidence * recency
            norm += abs(recency)
        if norm > 0.0:
            fused = total / norm
            if fused > 0.0:
                profile[tag] = fused
    return profile


def _merge_into(base: Dict[str, float], incoming: Dict[str, float]) -> None:
    """Simple average-merge: update ``base`` in-place."""
    for k in set(base) | set(incoming):
        base[k] = (base.get(k, 0.0) + incoming.get(k, 0.0)) / 2.0


# ------------------------------------------------------------------ engine


class PlaceAlignmentEngine:
    """Build a ``SharedConceptGraph`` from a ``CollectiveMemory`` snapshot.

    Parameters
    ----------
    semantic_threshold:
        Minimum combined score (cosine similarity + tag_match_bonus) for
        two places to be merged into the same concept. Lowering this creates
        larger, fuzzier concepts; raising it creates more, purer ones.
    spatial_radius:
        Maximum Euclidean distance (grid units) between a candidate place and
        the concept's current centroid to allow spatial merging.
        ``float('inf')`` disables the spatial gate entirely.
    tag_match_bonus:
        Additive bonus applied when dominant tags agree. This makes
        tag-agreement the primary clustering signal on top of cosine sim.
    min_confidence:
        Places whose fused max-confidence is below this threshold are skipped.
    cross_env:
        Allow places from different ``env_id``s to share a concept.
        Off by default; turn on for cross-layout transfer experiments.
    """

    def __init__(
        self,
        semantic_threshold: float = 0.5,
        spatial_radius: float = 3.0,
        tag_match_bonus: float = 0.4,
        min_confidence: float = 0.05,
        cross_env: bool = False,
    ) -> None:
        self.semantic_threshold = float(semantic_threshold)
        self.spatial_radius = float(spatial_radius)
        self.tag_match_bonus = float(tag_match_bonus)
        self.min_confidence = float(min_confidence)
        self.cross_env = bool(cross_env)

    # -------------------------------------------------------------- public

    def build(self, collective: CollectiveMemory) -> SharedConceptGraph:
        """Build and return a new ``SharedConceptGraph`` from current beliefs."""
        graph = SharedConceptGraph()
        cur_seq = collective._next_seq  # noqa: SLF001 — read-only diagnostic

        place_descs = self._gather_descriptors(collective, cur_seq)

        if self.cross_env:
            self._cluster_group(graph, place_descs)
        else:
            groups: Dict[str, List[Dict]] = {}
            for desc in place_descs:
                groups.setdefault(desc["env_id"], []).append(desc)
            for group in groups.values():
                self._cluster_group(graph, group)

        self._infer_edges(graph, collective)

        for concept in graph.concepts.values():
            self._finalize_concept(concept)

        return graph

    # ----------------------------------------------------------- internals

    def _gather_descriptors(
        self, collective: CollectiveMemory, cur_seq: int
    ) -> List[Dict]:
        descs: List[Dict] = []
        for (env_id, place_key), aggregate in collective.place_beliefs.items():
            profile = _fuse_profile(
                aggregate.tag_records, collective.recency_lambda, cur_seq
            )
            if not profile:
                continue
            max_conf = max(profile.values())
            if max_conf < self.min_confidence:
                continue
            dominant_tag = max(profile.items(), key=lambda kv: kv[1])[0]

            agents: Set[str] = set()
            episode_sigs: Set[Tuple[str, int]] = set()
            support_count = 0
            last_seq = -1
            for records in aggregate.tag_records.values():
                for r in records:
                    agents.add(r.agent_id)
                    episode_sigs.add((r.agent_id, r.episode_id))
                    support_count += 1
                    last_seq = max(last_seq, r.wall_clock_seq)

            descs.append({
                "env_id": env_id,
                "graph_key": (env_id, place_key),  # key type for SharedConceptGraph
                "raw_place_key": place_key,
                "profile": dict(profile),
                "dominant_tag": dominant_tag,
                "agents": agents,
                "episode_sigs": episode_sigs,
                "support_count": support_count,
                "last_seq": last_seq,
                "xy": _try_xy(place_key),
            })

        # Most-supported places become seeds (greedy: process high-support first)
        descs.sort(key=lambda d: (-d["support_count"], d["last_seq"]))
        return descs

    def _cluster_group(
        self, graph: SharedConceptGraph, descs: List[Dict]
    ) -> None:
        """Greedy single-pass clustering over one env group (or global)."""
        # Running state per concept (not stored in graph to avoid mutating it mid-loop)
        concept_profiles: Dict[str, Dict[str, float]] = {}
        concept_xys: Dict[str, List[Tuple[float, float]]] = {}

        for desc in descs:
            best_cid: Optional[str] = None
            best_score: float = -1.0

            for cid, concept in graph.concepts.items():
                sem_sim = cosine_similarity(desc["profile"], concept_profiles[cid])
                tag_bonus = (
                    self.tag_match_bonus
                    if desc["dominant_tag"] == concept.dominant_tag
                    else 0.0
                )
                combined = sem_sim + tag_bonus

                if combined < self.semantic_threshold:
                    continue

                # Spatial gate: only applies when both sides have coordinates
                if desc["xy"] is not None and concept_xys.get(cid):
                    cx, cy = _centroid(concept_xys[cid])
                    dist = euclidean_xy(desc["xy"], (cx, cy))
                    if dist > self.spatial_radius:
                        continue

                if combined > best_score:
                    best_score = combined
                    best_cid = cid

            if best_cid is not None:
                concept = graph.concepts[best_cid]
                graph.attach_place(
                    best_cid,
                    desc["graph_key"],
                    agent_ids=desc["agents"],
                    episode_signatures=desc["episode_sigs"],
                )
                concept.support_count += desc["support_count"]
                concept.last_seen_seq = max(concept.last_seen_seq, desc["last_seq"])
                _merge_into(concept_profiles[best_cid], desc["profile"])
                if desc["xy"] is not None:
                    concept_xys.setdefault(best_cid, []).append(desc["xy"])
            else:
                cid = graph.new_concept_id(desc["dominant_tag"])
                concept = SharedConceptNode(
                    concept_id=cid,
                    dominant_tag=desc["dominant_tag"],
                    semantic_profile=dict(desc["profile"]),
                    member_place_keys=[desc["graph_key"]],
                    supporting_agents=set(desc["agents"]),
                    episode_signatures=set(desc["episode_sigs"]),
                    support_count=desc["support_count"],
                    last_seen_seq=desc["last_seq"],
                )
                graph.concepts[cid] = concept
                graph.place_to_concept[desc["graph_key"]] = cid
                concept_profiles[cid] = dict(desc["profile"])
                if desc["xy"] is not None:
                    concept_xys[cid] = [desc["xy"]]

        # Sync merged profiles back into concept objects
        for cid, merged_profile in concept_profiles.items():
            graph.concepts[cid].semantic_profile = dict(merged_profile)

    def _infer_edges(
        self, graph: SharedConceptGraph, collective: CollectiveMemory
    ) -> None:
        """Add concept→concept edges derived from transition_records."""
        for (env_id, src_key), aggregate in collective.place_beliefs.items():
            src_cid = graph.place_to_concept.get((env_id, src_key))
            if src_cid is None:
                continue
            for dst_key, records in aggregate.transition_records.items():
                dst_cid = graph.place_to_concept.get((env_id, dst_key))
                if dst_cid is None or dst_cid == src_cid:
                    continue
                for record in records:
                    if record.confidence > 0.0:
                        graph.add_edge(
                            src_cid, dst_cid,
                            record.agent_id, record.wall_clock_seq,
                        )

    def _finalize_concept(self, concept: SharedConceptNode) -> None:
        """Recompute centroid, radius, confidence, and conflict_score."""
        xys: List[Tuple[float, float]] = []
        for env_id, place_key in concept.member_place_keys:
            xy = _try_xy(place_key)
            if xy is not None:
                xys.append(xy)

        if xys:
            cx = sum(x for x, _ in xys) / len(xys)
            cy = sum(y for _, y in xys) / len(xys)
            concept.centroid_xy = (cx, cy)
            if len(xys) > 1:
                concept.radius_xy = max(
                    math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in xys
                )

        if concept.semantic_profile:
            concept.confidence = max(concept.semantic_profile.values())
            concept.conflict_score = profile_entropy(concept.semantic_profile)

        # freshness: initialised to 1.0; TemporalDecayEngine.apply_to_graph()
        # will overwrite this with the decayed value in Phase 3a.
        concept.freshness = 1.0


# ──────────────────────────────────────────────────────── Phase 3c: incremental


class IncrementalAlignmentEngine(PlaceAlignmentEngine):
    """Extends ``PlaceAlignmentEngine`` with online incremental updates.

    On the first call to ``update()``, a full build is performed.  On
    subsequent calls, only places that received new events since the last
    build are re-evaluated: their profiles are recomputed and they are
    either reinforced in their current concept, reassigned to a better-
    fitting one, or used to seed a new concept.

    This keeps incremental updates O(|dirty| × |concepts|) instead of
    O(|all_places| × |concepts|), while the graph stays consistent across
    calls.

    Attributes
    ----------
    current_graph:
        The most recently built/updated ``SharedConceptGraph``.
    last_built_seq:
        The ``wall_clock_seq`` value at which the graph was last updated.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._graph: Optional[SharedConceptGraph] = None
        self._last_built_seq: int = 0
        # Snapshot of place_to_concept at last build/update for churn tracking
        self._prev_place_snap: Dict[Tuple, str] = {}

    @property
    def current_graph(self) -> Optional[SharedConceptGraph]:
        return self._graph

    @property
    def last_built_seq(self) -> int:
        return self._last_built_seq

    # ----------------------------------------------------------------- update

    def update(
        self,
        collective: "CollectiveMemory",  # type: ignore[name-defined]
    ) -> Tuple[SharedConceptGraph, Dict[str, Any]]:
        """Update the concept graph from new events.

        Returns ``(graph, stats)`` where ``stats`` contains:

        * ``first_build`` — True if this was a full rebuild
        * ``n_dirty_places`` — places with events after last build
        * ``n_reinforced`` — dirty places whose membership was unchanged
        * ``n_reassigned`` — places that moved to a different concept
        * ``n_new_concepts`` — new concepts seeded by dirty places
        * ``prev_n_concepts`` — concept count before this update
        * ``curr_n_concepts`` — concept count after this update
        """
        if self._graph is None or self._last_built_seq == 0:
            graph = super().build(collective)
            self._graph = graph
            self._last_built_seq = collective._next_seq  # noqa: SLF001
            self._prev_place_snap = dict(graph.place_to_concept)
            return graph, {
                "first_build": True,
                "n_dirty_places": 0,
                "n_reinforced": 0,
                "n_reassigned": 0,
                "n_new_concepts": len(graph.concepts),
                "prev_n_concepts": 0,
                "curr_n_concepts": len(graph.concepts),
            }

        prev_snap = dict(self._graph.place_to_concept)
        dirty = self._dirty_places(collective)

        if not dirty:
            return self._graph, {
                "first_build": False,
                "n_dirty_places": 0,
                "n_reinforced": 0,
                "n_reassigned": 0,
                "n_new_concepts": 0,
                "prev_n_concepts": len(self._graph.concepts),
                "curr_n_concepts": len(self._graph.concepts),
            }

        prev_n = len(self._graph.concepts)
        n_reinforced, n_reassigned, n_new = self._apply_dirty(
            self._graph, collective, dirty
        )
        self._last_built_seq = collective._next_seq  # noqa: SLF001
        self._prev_place_snap = prev_snap

        return self._graph, {
            "first_build": False,
            "n_dirty_places": len(dirty),
            "n_reinforced": n_reinforced,
            "n_reassigned": n_reassigned,
            "n_new_concepts": n_new,
            "prev_n_concepts": prev_n,
            "curr_n_concepts": len(self._graph.concepts),
        }

    def snapshot_membership(self) -> Dict[Tuple, str]:
        """Return a copy of the current place→concept mapping."""
        if self._graph is None:
            return {}
        return dict(self._graph.place_to_concept)

    # ----------------------------------------------------------------- internals

    def _dirty_places(
        self, collective: "CollectiveMemory"  # type: ignore[name-defined]
    ) -> Set[Tuple[str, Tuple]]:
        """Places that received events after ``_last_built_seq``."""
        dirty: Set[Tuple[str, Tuple]] = set()
        for event in collective.events:
            if event.provenance.wall_clock_seq <= self._last_built_seq:
                continue
            raw_key = event.payload.get("place_key")
            if raw_key:
                from songline_drive.collective_memory import _normalize_place_key
                dirty.add((event.provenance.env_id, _normalize_place_key(raw_key)))
        return dirty

    def _apply_dirty(
        self,
        graph: SharedConceptGraph,
        collective: "CollectiveMemory",  # type: ignore[name-defined]
        dirty: Set[Tuple[str, Tuple]],
    ) -> Tuple[int, int, int]:
        """Re-evaluate dirty places against the existing graph.

        Returns ``(n_reinforced, n_reassigned, n_new_concepts)``.
        """
        cur_seq = collective._next_seq  # noqa: SLF001
        n_reinforced = n_reassigned = n_new = 0
        affected_cids: Set[str] = set()

        for env_id, place_key in dirty:
            aggregate = collective.place_beliefs.get((env_id, place_key))
            if aggregate is None:
                continue
            profile = _fuse_profile(
                aggregate.tag_records, collective.recency_lambda, cur_seq
            )
            if not profile:
                continue
            max_conf = max(profile.values())
            if max_conf < self.min_confidence:
                continue

            dominant_tag = max(profile.items(), key=lambda kv: kv[1])[0]
            agents: Set[str] = set()
            episode_sigs: Set[Tuple[str, int]] = set()
            support_count = 0
            last_seq = -1
            for records in aggregate.tag_records.values():
                for r in records:
                    agents.add(r.agent_id)
                    episode_sigs.add((r.agent_id, r.episode_id))
                    support_count += 1
                    last_seq = max(last_seq, r.wall_clock_seq)

            graph_key = (env_id, place_key)
            xy = _try_xy(place_key)
            current_cid = graph.place_to_concept.get(graph_key)

            # Find best concept match (same logic as _cluster_group)
            best_cid: Optional[str] = None
            best_score: float = -1.0
            for cid, concept in graph.concepts.items():
                sem_sim = cosine_similarity(profile, concept.semantic_profile)
                bonus = self.tag_match_bonus if dominant_tag == concept.dominant_tag else 0.0
                combined = sem_sim + bonus
                if combined < self.semantic_threshold:
                    continue
                if xy is not None and concept.centroid_xy is not None:
                    if euclidean_xy(xy, concept.centroid_xy) > self.spatial_radius:
                        continue
                if combined > best_score:
                    best_score = combined
                    best_cid = cid

            if best_cid is not None and best_cid == current_cid:
                # Reinforce existing membership
                concept = graph.concepts[best_cid]
                _merge_into(concept.semantic_profile, profile)
                concept.support_count += support_count
                concept.last_seen_seq = max(concept.last_seen_seq, last_seq)
                concept.supporting_agents.update(agents)
                concept.episode_signatures.update(episode_sigs)
                affected_cids.add(best_cid)
                n_reinforced += 1

            elif best_cid is not None and best_cid != current_cid:
                # Reassign: detach from old concept, attach to new
                if current_cid is not None and current_cid in graph.concepts:
                    old = graph.concepts[current_cid]
                    if graph_key in old.member_place_keys:
                        old.member_place_keys.remove(graph_key)
                    affected_cids.add(current_cid)
                graph.attach_place(best_cid, graph_key, agents, episode_sigs)
                concept = graph.concepts[best_cid]
                _merge_into(concept.semantic_profile, profile)
                concept.support_count += support_count
                concept.last_seen_seq = max(concept.last_seen_seq, last_seq)
                affected_cids.add(best_cid)
                n_reassigned += 1

            else:
                # No suitable concept — create new one
                if current_cid is None:
                    cid = graph.new_concept_id(dominant_tag)
                    concept = SharedConceptNode(
                        concept_id=cid,
                        dominant_tag=dominant_tag,
                        semantic_profile=dict(profile),
                        member_place_keys=[graph_key],
                        supporting_agents=set(agents),
                        episode_signatures=set(episode_sigs),
                        support_count=support_count,
                        last_seen_seq=last_seq,
                    )
                    graph.concepts[cid] = concept
                    graph.place_to_concept[graph_key] = cid
                    affected_cids.add(cid)
                    n_new += 1
                else:
                    # No better match found → reinforce current (stability preference)
                    concept = graph.concepts.get(current_cid)
                    if concept is not None:
                        _merge_into(concept.semantic_profile, profile)
                        concept.support_count += support_count
                        concept.last_seen_seq = max(concept.last_seen_seq, last_seq)
                        affected_cids.add(current_cid)
                        n_reinforced += 1

        # Re-infer edges for dirty place transitions
        from songline_drive.collective_memory import _normalize_place_key
        for env_id, place_key in dirty:
            aggregate = collective.place_beliefs.get((env_id, place_key))
            if aggregate is None:
                continue
            src_cid = graph.place_to_concept.get((env_id, place_key))
            if src_cid is None:
                continue
            for dst_key, records in aggregate.transition_records.items():
                dst_cid = graph.place_to_concept.get((env_id, dst_key))
                if dst_cid is None or dst_cid == src_cid:
                    continue
                for record in records:
                    if record.wall_clock_seq > self._last_built_seq and record.confidence > 0.0:
                        graph.add_edge(src_cid, dst_cid, record.agent_id, record.wall_clock_seq)

        # Finalize geometry + stats for affected concepts
        for cid in affected_cids:
            if cid in graph.concepts:
                self._finalize_concept(graph.concepts[cid])

        return n_reinforced, n_reassigned, n_new
