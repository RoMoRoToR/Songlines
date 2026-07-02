"""Phase 1 collective memory substrate.

A typed, debuggable, append-only event bus + provenance-aware fused query
layer that several agents can publish observations into and query as a
shared place-belief store.

Phase 1 deliberately does NOT include semantic-field diffusion (Phase 4),
canonical place alignment (Phase 2) or active belief revision (Phase 3).
The hooks for them are present (transition records, conflict scoring,
intent commitments), but fusion is just trust-weighted recency-decayed
average of contributing observations.

Zero modifications to existing single-agent modules. ``CollectiveMemory``
can be used standalone or driven by :mod:`multiagent_runtime` adapters.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from songline_drive.collective_types import (
    AgentSignature,
    BeliefRecord,
    CollectiveDecisionExplanation,
    CollectiveEvent,
    CollectiveQuery,
    CollectiveQueryResult,
    EventProvenance,
    EventType,
    PlaceBeliefAggregate,
)


def _normalize_place_key(raw: Any) -> Tuple[Any, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return (raw,)


class CollectiveMemory:
    """Append-only event bus + place-belief aggregator.

    Parameters
    ----------
    recency_lambda:
        Decay applied per global ``wall_clock_seq`` step when fusing
        observations. ``0.95`` means an observation 20 events old still
        carries ~36% of its original weight.
    convergence_min_score:
        Default fused-score threshold used by ``time_to_collective_convergence``
        and similar diagnostics.
    """

    def __init__(
        self,
        recency_lambda: float = 0.95,
        convergence_min_score: float = 0.5,
    ):
        self.recency_lambda = float(recency_lambda)
        self.convergence_min_score = float(convergence_min_score)
        self.agents: Dict[str, AgentSignature] = {}
        self.events: List[CollectiveEvent] = []
        self.place_beliefs: Dict[Tuple[str, Tuple[Any, ...]], PlaceBeliefAggregate] = {}
        self.intent_commitments: Dict[str, Dict[str, Any]] = {}
        self.episode_consolidations: List[Dict[str, Any]] = []
        self._reads_log: List[Dict[str, Any]] = []
        self._next_event_id: int = 0
        self._next_seq: int = 0

    # ------------------------------------------------------------------ public

    def register_agent(self, signature: AgentSignature) -> None:
        self.agents[signature.agent_id] = signature

    def publish_event(
        self,
        event_type: EventType,
        agent_id: str,
        episode_id: int,
        step_idx: int,
        env_id: str,
        payload: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> CollectiveEvent:
        self._next_seq += 1
        event = CollectiveEvent(
            event_id=self._next_event_id,
            event_type=EventType(event_type),
            provenance=EventProvenance(
                agent_id=str(agent_id),
                episode_id=int(episode_id),
                step_idx=int(step_idx),
                env_id=str(env_id),
                wall_clock_seq=self._next_seq,
            ),
            payload=dict(payload or {}),
            confidence=float(confidence),
        )
        self._next_event_id += 1
        self.events.append(event)
        self._integrate_event(event)
        return event

    def consolidate_episode(
        self,
        agent_id: str,
        episode_id: int,
        env_id: str,
        outcome: Optional[Dict[str, Any]] = None,
        visited_place_keys: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "agent_id": str(agent_id),
            "episode_id": int(episode_id),
            "env_id": str(env_id),
            "outcome": dict(outcome or {}),
            "visited_place_keys": [
                _normalize_place_key(p) for p in (visited_place_keys or [])
            ],
            "wall_clock_seq_end": self._next_seq,
        }
        self.episode_consolidations.append(record)
        return record

    # ----------------------------------------------------------------- queries

    def query_collective_nodes(
        self,
        query: CollectiveQuery,
        top_k: int = 5,
    ) -> List[CollectiveQueryResult]:
        score_weights = query.score_weights or {query.target_tag: 1.0}
        penalty_weights = query.penalty_weights or {}

        results: List[CollectiveQueryResult] = []
        for (env_id, place_key), aggregate in self.place_beliefs.items():
            if query.env_id is not None and env_id != query.env_id:
                continue

            per_tag_fused: Dict[str, float] = {}
            score = 0.0
            contributing_agents: set = set()
            contributing_event_seqs: List[int] = []

            for tag, weight in score_weights.items():
                fused, agents, seqs = self._fuse_records(
                    aggregate.tag_records.get(tag, []),
                    exclude_agent_id=query.requesting_agent_id if query.exclude_self else None,
                )
                if fused <= 0.0:
                    continue
                per_tag_fused[tag] = fused
                score += float(weight) * fused
                contributing_agents.update(agents)
                contributing_event_seqs.extend(seqs)

            for tag, weight in penalty_weights.items():
                fused, _, _ = self._fuse_records(
                    aggregate.tag_records.get(tag, []),
                    exclude_agent_id=query.requesting_agent_id if query.exclude_self else None,
                )
                if fused <= 0.0:
                    continue
                per_tag_fused[f"-{tag}"] = fused
                score -= float(weight) * fused

            if score < query.min_fused_score:
                continue
            if len(contributing_agents) < query.min_supporting_agents:
                continue

            used_other = bool(contributing_agents - {query.requesting_agent_id})
            result = CollectiveQueryResult(
                place_key=place_key,
                env_id=env_id,
                fused_score=score,
                contributing_agents=sorted(contributing_agents),
                contributing_event_seqs=sorted(set(contributing_event_seqs)),
                used_other_agent_knowledge=used_other,
                target_tag=query.target_tag,
                per_tag_fused=per_tag_fused,
            )
            results.append(result)

        results.sort(key=lambda r: r.fused_score, reverse=True)
        results = results[: max(0, int(top_k))]

        for result in results:
            self._reads_log.append({
                "requesting_agent_id": query.requesting_agent_id,
                "intent_type": query.intent_type,
                "target_tag": query.target_tag,
                "place_key": result.place_key,
                "env_id": result.env_id,
                "fused_score": result.fused_score,
                "contributing_agents": list(result.contributing_agents),
                "used_other_agent_knowledge": result.used_other_agent_knowledge,
                "wall_clock_seq": self._next_seq,
            })
        return results

    def query_collective_paths(
        self,
        env_id: str,
        src_key: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Phase 1 minimal: list outgoing transition beliefs from ``src_key``,
        sorted by fused (trust * recency) success weight."""
        key = (str(env_id), _normalize_place_key(src_key))
        aggregate = self.place_beliefs.get(key)
        if aggregate is None:
            return []
        out: List[Dict[str, Any]] = []
        for dst_key, records in aggregate.transition_records.items():
            if not records:
                continue
            fused, agents, seqs = self._fuse_records(records)
            out.append({
                "src_key": aggregate.place_key,
                "dst_key": dst_key,
                "env_id": aggregate.env_id,
                "fused_success": fused,
                "contributing_agents": sorted(agents),
                "contributing_event_seqs": sorted(set(seqs)),
            })
        out.sort(key=lambda r: r["fused_success"], reverse=True)
        return out[: max(0, int(top_k))]

    def explain_collective_decision(
        self,
        result: CollectiveQueryResult,
        requesting_agent_id: str,
    ) -> CollectiveDecisionExplanation:
        return CollectiveDecisionExplanation(
            place_key=result.place_key,
            env_id=result.env_id,
            target_tag=result.target_tag,
            fused_score=result.fused_score,
            contributing_agents=list(result.contributing_agents),
            per_tag_fused=dict(result.per_tag_fused),
            contributing_event_seqs=list(result.contributing_event_seqs),
            used_other_agent_knowledge=result.used_other_agent_knowledge,
            requesting_agent_id=str(requesting_agent_id),
        )

    # ------------------------------------------------------------------- views

    def fused_tag_score(
        self,
        env_id: str,
        place_key: Any,
        tag: str,
        exclude_agent_id: Optional[str] = None,
    ) -> Tuple[float, List[str]]:
        aggregate = self.place_beliefs.get((str(env_id), _normalize_place_key(place_key)))
        if aggregate is None:
            return 0.0, []
        fused, agents, _ = self._fuse_records(
            aggregate.tag_records.get(tag, []),
            exclude_agent_id=exclude_agent_id,
        )
        return fused, sorted(agents)

    def all_events(self) -> List[CollectiveEvent]:
        return list(self.events)

    def reads_log(self) -> List[Dict[str, Any]]:
        return list(self._reads_log)

    def stats(self) -> Dict[str, Any]:
        return {
            "n_agents": len(self.agents),
            "n_events": len(self.events),
            "n_places": len(self.place_beliefs),
            "n_episodes_consolidated": len(self.episode_consolidations),
            "n_reads_logged": len(self._reads_log),
            "n_active_intent_commitments": len(self.intent_commitments),
            "wall_clock_seq": self._next_seq,
        }

    # ----------------------------------------------------------- serialization

    def export(self, out_dir: str, filename_prefix: str = "collective") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        events_path = os.path.join(out_dir, f"{filename_prefix}_events.jsonl")
        places_path = os.path.join(out_dir, f"{filename_prefix}_places.json")
        reads_path = os.path.join(out_dir, f"{filename_prefix}_reads.jsonl")
        stats_path = os.path.join(out_dir, f"{filename_prefix}_stats.json")

        with open(events_path, "w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(json.dumps(self._event_to_dict(event), ensure_ascii=False) + "\n")

        place_dump = {}
        for (env_id, place_key), aggregate in self.place_beliefs.items():
            place_dump[f"{env_id}::{list(place_key)}"] = {
                "env_id": env_id,
                "place_key": list(place_key),
                "last_seen_seq": aggregate.last_seen_seq,
                "tag_records": {
                    tag: [self._record_to_dict(r) for r in records]
                    for tag, records in aggregate.tag_records.items()
                },
                "transition_records": {
                    str(list(dst)): [self._record_to_dict(r) for r in records]
                    for dst, records in aggregate.transition_records.items()
                },
            }
        with open(places_path, "w", encoding="utf-8") as fh:
            json.dump(place_dump, fh, ensure_ascii=False, indent=2)

        with open(reads_path, "w", encoding="utf-8") as fh:
            for entry in self._reads_log:
                fh.write(json.dumps(self._read_to_dict(entry), ensure_ascii=False) + "\n")

        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(self.stats(), fh, ensure_ascii=False, indent=2)

        return {
            "events_jsonl": events_path,
            "places_json": places_path,
            "reads_jsonl": reads_path,
            "stats_json": stats_path,
        }

    # ----------------------------------------------------------------- internals

    def _trust(self, agent_id: str) -> float:
        signature = self.agents.get(agent_id)
        if signature is None:
            return 1.0
        return float(signature.trust)

    def _integrate_event(self, event: CollectiveEvent) -> None:
        env_id = event.provenance.env_id
        agent_id = event.provenance.agent_id
        trust = self._trust(agent_id)
        seq = event.provenance.wall_clock_seq

        if event.event_type in (EventType.PLACE_OBSERVED, EventType.CONCEPT_CONFIRMED):
            place_key = _normalize_place_key(event.payload.get("place_key"))
            aggregate = self.place_beliefs.setdefault(
                (env_id, place_key),
                PlaceBeliefAggregate(place_key=place_key, env_id=env_id),
            )
            tags = event.payload.get("semantic_tags") or {}
            for tag, raw_conf in tags.items():
                record = BeliefRecord(
                    agent_id=agent_id,
                    episode_id=event.provenance.episode_id,
                    step_idx=event.provenance.step_idx,
                    confidence=max(0.0, float(raw_conf)) * event.confidence * trust,
                    freshness=float(event.payload.get("node_freshness", 1.0)),
                    wall_clock_seq=seq,
                )
                aggregate.tag_records.setdefault(str(tag), []).append(record)
            aggregate.last_seen_seq = max(aggregate.last_seen_seq, seq)

        elif event.event_type == EventType.TRANSITION_VALIDATED:
            src_key = _normalize_place_key(event.payload.get("src_key"))
            dst_key = _normalize_place_key(event.payload.get("dst_key"))
            aggregate = self.place_beliefs.setdefault(
                (env_id, src_key),
                PlaceBeliefAggregate(place_key=src_key, env_id=env_id),
            )
            record = BeliefRecord(
                agent_id=agent_id,
                episode_id=event.provenance.episode_id,
                step_idx=event.provenance.step_idx,
                confidence=event.confidence * trust,
                freshness=1.0,
                wall_clock_seq=seq,
            )
            aggregate.transition_records.setdefault(dst_key, []).append(record)
            aggregate.last_seen_seq = max(aggregate.last_seen_seq, seq)

        elif event.event_type == EventType.HAZARD_CHANGED:
            place_key = _normalize_place_key(event.payload.get("place_key"))
            aggregate = self.place_beliefs.setdefault(
                (env_id, place_key),
                PlaceBeliefAggregate(place_key=place_key, env_id=env_id),
            )
            record = BeliefRecord(
                agent_id=agent_id,
                episode_id=event.provenance.episode_id,
                step_idx=event.provenance.step_idx,
                confidence=event.confidence * trust,
                freshness=1.0,
                wall_clock_seq=seq,
            )
            aggregate.tag_records.setdefault("hazard_edge", []).append(record)
            aggregate.last_seen_seq = max(aggregate.last_seen_seq, seq)

        elif event.event_type == EventType.RESOURCE_DEPLETED:
            place_key = _normalize_place_key(event.payload.get("place_key"))
            aggregate = self.place_beliefs.setdefault(
                (env_id, place_key),
                PlaceBeliefAggregate(place_key=place_key, env_id=env_id),
            )
            depleted_tag = str(event.payload.get("resource_tag", "water_source"))
            record = BeliefRecord(
                agent_id=agent_id,
                episode_id=event.provenance.episode_id,
                step_idx=event.provenance.step_idx,
                confidence=-abs(event.confidence) * trust,
                freshness=1.0,
                wall_clock_seq=seq,
            )
            aggregate.tag_records.setdefault(depleted_tag, []).append(record)
            aggregate.last_seen_seq = max(aggregate.last_seen_seq, seq)

        elif event.event_type == EventType.ROUTE_SUCCEEDED:
            place_key = _normalize_place_key(event.payload.get("target_place_key"))
            aggregate = self.place_beliefs.setdefault(
                (env_id, place_key),
                PlaceBeliefAggregate(place_key=place_key, env_id=env_id),
            )
            target_tag = str(event.payload.get("target_tag", ""))
            if target_tag:
                record = BeliefRecord(
                    agent_id=agent_id,
                    episode_id=event.provenance.episode_id,
                    step_idx=event.provenance.step_idx,
                    confidence=event.confidence * trust,
                    freshness=1.0,
                    wall_clock_seq=seq,
                )
                aggregate.tag_records.setdefault(target_tag, []).append(record)
            aggregate.last_seen_seq = max(aggregate.last_seen_seq, seq)

        elif event.event_type == EventType.ROUTE_FAILED:
            place_key = _normalize_place_key(event.payload.get("target_place_key"))
            aggregate = self.place_beliefs.setdefault(
                (env_id, place_key),
                PlaceBeliefAggregate(place_key=place_key, env_id=env_id),
            )
            target_tag = str(event.payload.get("target_tag", ""))
            if target_tag:
                record = BeliefRecord(
                    agent_id=agent_id,
                    episode_id=event.provenance.episode_id,
                    step_idx=event.provenance.step_idx,
                    confidence=-0.5 * abs(event.confidence) * trust,
                    freshness=1.0,
                    wall_clock_seq=seq,
                )
                aggregate.tag_records.setdefault(target_tag, []).append(record)
            aggregate.last_seen_seq = max(aggregate.last_seen_seq, seq)

        elif event.event_type == EventType.INTENT_COMMITTED:
            self.intent_commitments[agent_id] = {
                "intent_type": event.payload.get("intent_type"),
                "target_place_key": _normalize_place_key(event.payload.get("target_place_key")),
                "env_id": env_id,
                "since_seq": seq,
                "episode_id": event.provenance.episode_id,
            }

        elif event.event_type == EventType.INTENT_RELEASED:
            self.intent_commitments.pop(agent_id, None)

    def _fuse_records(
        self,
        records: Iterable[BeliefRecord],
        exclude_agent_id: Optional[str] = None,
    ) -> Tuple[float, List[str], List[int]]:
        records = list(records)
        if not records:
            return 0.0, [], []
        cur_seq = self._next_seq
        total = 0.0
        norm = 0.0
        agents: List[str] = []
        seqs: List[int] = []
        for record in records:
            if exclude_agent_id is not None and record.agent_id == exclude_agent_id:
                continue
            recency = self.recency_lambda ** max(0, cur_seq - record.wall_clock_seq)
            total += record.confidence * recency
            norm += abs(recency)
            agents.append(record.agent_id)
            seqs.append(record.wall_clock_seq)
        if norm <= 0.0:
            return 0.0, agents, seqs
        return total / norm, agents, seqs

    def _event_to_dict(self, event: CollectiveEvent) -> Dict[str, Any]:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "agent_id": event.provenance.agent_id,
            "episode_id": event.provenance.episode_id,
            "step_idx": event.provenance.step_idx,
            "env_id": event.provenance.env_id,
            "wall_clock_seq": event.provenance.wall_clock_seq,
            "confidence": event.confidence,
            "payload": event.payload,
        }

    def _record_to_dict(self, record: BeliefRecord) -> Dict[str, Any]:
        return {
            "agent_id": record.agent_id,
            "episode_id": record.episode_id,
            "step_idx": record.step_idx,
            "confidence": record.confidence,
            "freshness": record.freshness,
            "wall_clock_seq": record.wall_clock_seq,
        }

    def _read_to_dict(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(entry)
        out["place_key"] = list(entry.get("place_key", ()))
        return out
