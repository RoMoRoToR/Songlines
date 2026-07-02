"""Phase 4 — FieldAdapter: bridge between SemanticField and collective memory stack.

``FieldAdapter`` is the single integration point for Phase 4.  It wraps a
``ConceptRecallLayer`` (Phase 2/3) and a ``SemanticField`` (Phase 4) and
exposes a unified ``query()`` that degrades gracefully by ``FieldMode``:

    none        → raw Phase 1 query via ``CollectiveMemory.query_collective_nodes``
    descriptive → Phase 2/3 concept recall only; field is rebuilt but not used
    read_only   → concept recall + field reranking
    coordinated → read_only + reservation / occupancy pressure (Phase 4c)

Invariants maintained (Phase 4 spec §0):
    1. Phase 1 event bus is never written to by FieldAdapter.
    2. Phase 2 concept graph remains canonical; field is built on top.
    3. Phase 3 belief dynamics are applied by ConceptRecallLayer.refresh() first.
    4. Planner/control not touched; FieldAdapter only reorders candidates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from songline_drive.collective_field_types import FieldMode
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import CollectiveQuery, CollectiveQueryResult
from songline_drive.concept_recall import ConceptRecallLayer, ConceptRecallResult
from songline_drive.semantic_field import SemanticField


class FieldAdapter:
    """Bridge between SemanticField and the existing collective memory stack.

    Parameters
    ----------
    field:
        The ``SemanticField`` instance to manage.
    recall_layer:
        A fully configured ``ConceptRecallLayer`` (with decay_engine /
        conflict_rules set up for Phase 3).  ``refresh()`` on this layer
        applies Phase 3 dynamics before the field reads the graph.
    field_weight:
        Weight of field activation in the combined reranking score
        (Phase 4b).  0.0 = pure concept recall; 1.0 = pure field.
    mode:
        ``FieldMode`` constant; can be updated at runtime to enable
        ablation studies without changing any other code.
    """

    def __init__(
        self,
        field: SemanticField,
        recall_layer: ConceptRecallLayer,
        *,
        field_weight: float = 0.30,
        mode: str = FieldMode.DESCRIPTIVE,
    ) -> None:
        self.field = field
        self.recall_layer = recall_layer
        self.field_weight = float(field_weight)
        self.mode = FieldMode.validate(mode)
        # Keep field.mode in sync so reservation checks inside SemanticField
        # can still introspect mode if needed.
        self.field.mode = self.mode
        self._snapshots: List[Dict] = []
        self._graph: Optional[Any] = None  # last built SharedConceptGraph

    # ────────────────────────────────────────────────────────── refresh

    def refresh(
        self,
        collective: CollectiveMemory,
        current_seq: Optional[int] = None,
    ) -> Tuple[Any, SemanticField]:
        """Rebuild concept graph (Phase 2/3) then field (Phase 4).

        Returns ``(graph, field)`` for external inspection.

        Order of operations (enforces Phase 4 invariant §3):
        1. ``recall_layer.refresh()`` → applies Phase 3 decay + conflict to graph
        2. ``field.rebuild_from_concepts(graph)`` → reads pre-computed attributes
        """
        graph = self.recall_layer.refresh(collective)
        self._graph = graph
        seq = (
            current_seq
            if current_seq is not None
            else collective._next_seq  # noqa: SLF001
        )
        self.field.rebuild_from_concepts(graph, current_seq=seq)
        return graph, self.field

    def snapshot(self, label: str = "") -> Dict:
        """Capture and store a field snapshot (for metrics / visualisation)."""
        snap = self.field.to_snapshot()
        snap["label"] = label
        self._snapshots.append(snap)
        return snap

    # ────────────────────────────────────────────────────────── query

    def query(
        self,
        collective: CollectiveMemory,
        query: CollectiveQuery,
        top_k: int = 5,
        current_seq: Optional[int] = None,
    ) -> Tuple[List[CollectiveQueryResult], str]:
        """Unified query respecting FieldMode.

        Fallback chain (most → least capable):
          coordinated / read_only → concept recall + field reranking
          descriptive             → pure concept recall
          none                    → raw Phase 1 query

        In all modes except ``none``, the concept recall layer is consulted
        first; the field only reorders results.
        """
        if self.mode == FieldMode.NONE:
            raw = collective.query_collective_nodes(query, top_k=top_k)
            return raw, "raw_phase1"

        seq = (
            current_seq
            if current_seq is not None
            else collective._next_seq  # noqa: SLF001
        )

        # Fetch concept recall candidates (Phase 2/3)
        recall_results: List[ConceptRecallResult] = self.recall_layer.query(
            target_tag=query.target_tag,
            requesting_agent_id=query.requesting_agent_id,
            env_id=query.env_id,
            top_k=top_k * 3,
            current_seq=seq,
        )

        if self.mode == FieldMode.DESCRIPTIVE:
            if recall_results:
                converted = self.recall_layer.to_collective_results(
                    recall_results[:top_k],
                    target_tag=query.target_tag,
                    requesting_agent_id=query.requesting_agent_id,
                )
                return converted, "concept_recall"
            raw = collective.query_collective_nodes(query, top_k=top_k)
            return raw, "raw_fallback"

        # read_only / coordinated: rerank by field
        if not recall_results:
            raw = collective.query_collective_nodes(query, top_k=top_k)
            return raw, "raw_fallback"

        channel = query.target_tag
        reranked = self.field.rerank(
            recall_results,
            channel=channel,
            field_weight=self.field_weight,
        )

        converted = self.recall_layer.to_collective_results(
            reranked[:top_k],
            target_tag=query.target_tag,
            requesting_agent_id=query.requesting_agent_id,
        )

        source = (
            "field_coordinated"
            if self.mode == FieldMode.COORDINATED
            else "field_reranked"
        )
        return converted, source

    # ──────────────────────────────────────── reservation API (Phase 4c)

    def commit_reservation(
        self,
        agent_id: str,
        concept_id: str,
        channel: str,
        duration: int = 20,
        current_seq: int = 0,
    ) -> Optional[Any]:
        """Soft-reserve (concept, channel) for ``agent_id`` in COORDINATED mode.

        Returns the FieldReservation or None if not in COORDINATED mode.
        The reservation immediately reduces the concept's activation in the field
        so subsequent queries from other agents see the penalised value.
        """
        if self.mode != FieldMode.COORDINATED:
            return None
        return self.field.reserve(concept_id, channel, agent_id, duration, current_seq)

    def release_reservation(self, concept_id: str, agent_id: str) -> None:
        """Release a reservation made by ``agent_id`` on ``concept_id``."""
        self.field.release(concept_id, agent_id)

    def expire_reservations(self, current_seq: int) -> int:
        """Expire all reservations past their TTL. Returns count removed."""
        return self.field.expire_reservations(current_seq)

    @property
    def active_reservations(self) -> List[Any]:
        """All currently active FieldReservation objects."""
        return list(self.field._reservations.values())  # noqa: SLF001

    # ────────────────────────────────────────────────── direct field query

    def field_query(
        self,
        channel: str,
        requesting_agent_id: str,
        env_id: Optional[str] = None,
        top_k: int = 3,
        min_activation: float = 0.0,
        current_seq: int = 0,
    ) -> List[Any]:
        """Query the field directly (bypasses concept recall layer).

        Returns ``List[FieldQueryResult]`` sorted by activation descending.
        """
        return self.field.query(
            channel=channel,
            requesting_agent_id=requesting_agent_id,
            env_id=env_id,
            top_k=top_k,
            min_activation=min_activation,
            graph=self._graph,
            current_seq=current_seq,
        )

    @property
    def snapshots(self) -> List[Dict]:
        return self._snapshots

    @property
    def graph(self) -> Optional[Any]:
        return self._graph
