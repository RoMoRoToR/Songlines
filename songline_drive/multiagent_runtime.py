"""Phase 1 multi-agent orchestrator + read-only adapters.

This module deliberately stays a thin coordinator: it owns nothing
semantic itself, it just routes each agent's local observations into
the shared ``CollectiveMemory`` and proxies typed queries back.

Read-only adapters never touch the existing single-agent modules.
The ``GraphMemoryAdapter`` accepts duck-typed snapshots (so it works
whether you pass a live ``DynamicSonglineGraph`` or a JSON dict that
came out of ``DynamicSonglineGraph.export``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from songline_drive.collective_memory import CollectiveMemory, _normalize_place_key
from songline_drive.collective_types import (
    AgentSignature,
    CollectiveEvent,
    CollectiveQuery,
    CollectiveQueryResult,
    EventType,
)


class MultiAgentRuntime:
    """Routes per-agent observations and intent transitions through a
    single ``CollectiveMemory``.

    The runtime is environment-agnostic: it does not start MiniGrid /
    MiniWorld / BabyAI for you. It only enforces a clean contract
    between an existing agent loop and the shared substrate.
    """

    def __init__(self, collective: CollectiveMemory):
        self.collective = collective
        self._signatures: Dict[str, AgentSignature] = {}

    # --------------------------------------------------------------- agents

    def register(self, signature: AgentSignature) -> None:
        self.collective.register_agent(signature)
        self._signatures[signature.agent_id] = signature

    def get_signature(self, agent_id: str) -> AgentSignature:
        if agent_id not in self._signatures:
            raise KeyError(f"agent {agent_id!r} not registered with runtime")
        return self._signatures[agent_id]

    # ---------------------------------------------------------- publishing

    def publish_observation(
        self,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        place_key: Any,
        semantic_tags: Dict[str, float],
        *,
        node_freshness: float = 1.0,
        confidence: float = 1.0,
    ) -> CollectiveEvent:
        self.get_signature(agent_id)
        return self.collective.publish_event(
            event_type=EventType.PLACE_OBSERVED,
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            payload={
                "place_key": _normalize_place_key(place_key),
                "semantic_tags": dict(semantic_tags),
                "node_freshness": float(node_freshness),
            },
            confidence=float(confidence),
        )

    def confirm_concept(
        self,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        place_key: Any,
        concept_tag: str,
        confidence: float = 1.0,
    ) -> CollectiveEvent:
        self.get_signature(agent_id)
        return self.collective.publish_event(
            event_type=EventType.CONCEPT_CONFIRMED,
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            payload={
                "place_key": _normalize_place_key(place_key),
                "semantic_tags": {str(concept_tag): 1.0},
            },
            confidence=float(confidence),
        )

    def publish_transition(
        self,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        src_key: Any,
        dst_key: Any,
        confidence: float = 1.0,
    ) -> CollectiveEvent:
        self.get_signature(agent_id)
        return self.collective.publish_event(
            event_type=EventType.TRANSITION_VALIDATED,
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            payload={
                "src_key": _normalize_place_key(src_key),
                "dst_key": _normalize_place_key(dst_key),
            },
            confidence=float(confidence),
        )

    def commit_intent(
        self,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        intent_type: str,
        target_place_key: Any,
    ) -> CollectiveEvent:
        self.get_signature(agent_id)
        return self.collective.publish_event(
            event_type=EventType.INTENT_COMMITTED,
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            payload={
                "intent_type": str(intent_type),
                "target_place_key": _normalize_place_key(target_place_key),
            },
        )

    def release_intent(
        self,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        intent_type: str,
        success: bool,
        target_place_key: Any = (),
        target_tag: str = "",
    ) -> List[CollectiveEvent]:
        self.get_signature(agent_id)
        outcome_event = self.collective.publish_event(
            event_type=EventType.ROUTE_SUCCEEDED if success else EventType.ROUTE_FAILED,
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            payload={
                "intent_type": str(intent_type),
                "target_place_key": _normalize_place_key(target_place_key),
                "target_tag": str(target_tag),
            },
        )
        release_event = self.collective.publish_event(
            event_type=EventType.INTENT_RELEASED,
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            payload={"intent_type": str(intent_type)},
        )
        return [outcome_event, release_event]

    def report_hazard_change(
        self,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        place_key: Any,
        confidence: float = 1.0,
    ) -> CollectiveEvent:
        self.get_signature(agent_id)
        return self.collective.publish_event(
            event_type=EventType.HAZARD_CHANGED,
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            payload={"place_key": _normalize_place_key(place_key)},
            confidence=float(confidence),
        )

    def report_resource_depleted(
        self,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        place_key: Any,
        resource_tag: str,
        confidence: float = 1.0,
    ) -> CollectiveEvent:
        self.get_signature(agent_id)
        return self.collective.publish_event(
            event_type=EventType.RESOURCE_DEPLETED,
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            payload={
                "place_key": _normalize_place_key(place_key),
                "resource_tag": str(resource_tag),
            },
            confidence=float(confidence),
        )

    # ----------------------------------------------------------- consuming

    def query(
        self,
        agent_id: str,
        query: CollectiveQuery,
        top_k: int = 5,
    ) -> List[CollectiveQueryResult]:
        self.get_signature(agent_id)
        if query.requesting_agent_id != agent_id:
            query.requesting_agent_id = agent_id
        return self.collective.query_collective_nodes(query, top_k=top_k)

    def query_with_concept_recall(
        self,
        agent_id: str,
        query: CollectiveQuery,
        recall_layer: Any,  # ConceptRecallLayer — typed loosely to avoid circular import
        top_k: int = 5,
        *,
        fallback_to_raw: bool = True,
    ) -> Tuple[List[CollectiveQueryResult], str]:
        """Phase 2 integration: concept-augmented retrieval with Phase 1 fallback.

        Delegates to ``ConceptRecallLayer.query_collective``. Returns
        ``(results, source)`` where ``source`` is ``"concept_recall"``,
        ``"raw_fallback"``, or ``"empty"``.
        """
        self.get_signature(agent_id)
        if query.requesting_agent_id != agent_id:
            query.requesting_agent_id = agent_id
        return recall_layer.query_collective(
            self.collective,
            query,
            top_k=top_k,
            fallback_to_raw=fallback_to_raw,
        )

    def reserved_targets(self, env_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Currently-committed intent targets. Phase 1 helper that lets a
        consumer agent see which place is already being chased by a
        teammate before re-targeting."""
        out: Dict[str, Dict[str, Any]] = {}
        for agent_id, commitment in self.collective.intent_commitments.items():
            if env_id is not None and commitment.get("env_id") != env_id:
                continue
            out[agent_id] = dict(commitment)
        return out

    # --------------------------------------------------------- consolidation

    def consolidate_episode(
        self,
        agent_id: str,
        episode_id: int,
        env_id: str,
        outcome: Optional[Dict[str, Any]] = None,
        visited_place_keys: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        self.get_signature(agent_id)
        return self.collective.consolidate_episode(
            agent_id=agent_id,
            episode_id=episode_id,
            env_id=env_id,
            outcome=outcome,
            visited_place_keys=visited_place_keys,
        )


# ----------------------------------------------------------- read-only adapter


class GraphMemoryAdapter:
    """Read-only bridge from the existing single-agent graph snapshot to
    collective events.

    The adapter never imports ``DynamicSonglineGraph`` directly: it
    accepts any duck-typed object exposing ``.node_by_id``-style data
    or a plain dict from ``DynamicSonglineGraph.export(...)``. That way
    the collective layer can run alongside the current single-agent
    benchmark without touching its files.
    """

    @staticmethod
    def emit_node_observation(
        runtime: MultiAgentRuntime,
        *,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        place_key: Any,
        semantic_tag_confidence: Dict[str, float],
        node_freshness: float = 1.0,
        node_visits: int = 1,
    ) -> CollectiveEvent:
        """Convert a single graph node into a ``place_observed`` event.

        ``node_visits`` and ``node_freshness`` are folded into the event
        confidence so a brittle one-shot observation does not look the
        same as a well-confirmed node."""
        confidence = max(0.2, min(1.0, 0.5 + 0.5 * float(node_freshness)))
        if node_visits <= 0:
            confidence *= 0.5
        return runtime.publish_observation(
            agent_id=agent_id,
            episode_id=episode_id,
            step_idx=step_idx,
            env_id=env_id,
            place_key=place_key,
            semantic_tags=dict(semantic_tag_confidence),
            node_freshness=float(node_freshness),
            confidence=confidence,
        )

    @staticmethod
    def emit_from_snapshot_iter(
        runtime: MultiAgentRuntime,
        *,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        nodes: Iterable[Dict[str, Any]],
        place_key_field: str = "pose_mean",
        tag_field: str = "semantic_tag_confidence",
        freshness_field: str = "freshness",
        visits_field: str = "visits",
    ) -> List[CollectiveEvent]:
        events: List[CollectiveEvent] = []
        for node in nodes:
            raw_key = node.get(place_key_field) or node.get("pose_xy") or node.get("place_key")
            if raw_key is None:
                continue
            tags = dict(node.get(tag_field) or {})
            if not tags:
                continue
            event = GraphMemoryAdapter.emit_node_observation(
                runtime,
                agent_id=agent_id,
                episode_id=episode_id,
                step_idx=step_idx,
                env_id=env_id,
                place_key=tuple(raw_key) if isinstance(raw_key, (list, tuple)) else (raw_key,),
                semantic_tag_confidence=tags,
                node_freshness=float(node.get(freshness_field, 1.0)),
                node_visits=int(node.get(visits_field, 1)),
            )
            events.append(event)
        return events
