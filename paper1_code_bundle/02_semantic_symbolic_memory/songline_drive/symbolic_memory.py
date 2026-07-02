import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from songline_drive.types import (
    ConceptCluster,
    ConceptRecord,
    EpisodeRecord,
    EpisodeStepRecord,
    ManeuverPlan,
    OutcomeWindow,
    PlannerQuery,
    QueryDebugRecord,
    RelationRecord,
    SemanticTargetPredicate,
    SymbolicNode,
    SymbolicTransition,
)


class SymbolicMemory:
    def __init__(self, graph):
        self.graph = graph
        self.active_episode: Optional[EpisodeRecord] = None
        self.episode_records: List[EpisodeRecord] = []
        self.query_debug_records: List[QueryDebugRecord] = []
        self.relation_tag_names = {
            "near_water",
            "adjacent_hazard",
            "post_hazard_goal_rejoin",
            "open_safe_rest_zone",
        }
        self.relation_target_concepts = {
            "near_water": "water_source",
            "adjacent_hazard": "hazard_edge",
            "post_hazard_goal_rejoin": "goal_region",
            "open_safe_rest_zone": "safe_rest_zone",
        }
        self.last_concept_query_debug: Dict[str, Any] = {}
        self.symbolic_nodes: Dict[int, SymbolicNode] = {}
        self.graph_node_to_symbolic_node: Dict[int, int] = {}
        self.symbolic_transitions: Dict[int, Dict[int, SymbolicTransition]] = {}
        self.relation_records: Dict[str, Dict[int, RelationRecord]] = {}
        self.concept_records: Dict[str, ConceptRecord] = {}
        self._pending_pre_step: Dict[int, Dict[str, Any]] = {}

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def start_episode(self, episode_id: int, env_id: str = "", task_mode: str = "default"):
        self.active_episode = EpisodeRecord(
            episode_id=int(episode_id),
            env_id=str(env_id),
            task_mode=str(task_mode),
        )

    def _agent_state_snapshot(self, agent_state) -> Dict[str, Any]:
        if agent_state is None:
            return {}
        return {
            "thirst": float(getattr(agent_state, "thirst", 0.0)),
            "energy": float(getattr(agent_state, "energy", 0.0)),
            "risk_budget": float(getattr(agent_state, "risk_budget", 0.0)),
            "task_phase": str(getattr(agent_state, "task_phase", "")),
            "previous_task_phase": str(getattr(agent_state, "previous_task_phase", "")),
            "active_intent": (
                None
                if getattr(agent_state, "active_intent", None) is None
                else str(getattr(agent_state.active_intent, "value", agent_state.active_intent))
            ),
            "active_intent_reason": str(getattr(agent_state, "active_intent_reason", "")),
        }

    def _compute_freshness(self, now_step: int, last_seen_step: int) -> float:
        if hasattr(self.graph, "_compute_freshness"):
            return float(self.graph._compute_freshness(now_step, last_seen_step))
        age = max(0.0, float(now_step - last_seen_step))
        return float(np.exp(-age / 50.0))

    def _compute_confidence(self, visits: int) -> float:
        if hasattr(self.graph, "_compute_confidence"):
            return float(self.graph._compute_confidence(visits))
        return float(1.0 - np.exp(-max(0, int(visits)) / 5.0))

    def _ema_update(self, old: float, value: float, alpha: float) -> float:
        return ((1.0 - alpha) * float(old)) + (alpha * float(value))

    def _ensure_query(self, planner_query: Optional[PlannerQuery] = None, predicate: Optional[SemanticTargetPredicate] = None):
        if planner_query is not None:
            if not planner_query.required_tags:
                required_tags = dict(planner_query.target_predicate.required_tag_thresholds)
                required_tags.setdefault(str(planner_query.target_predicate.tag_name), float(planner_query.target_predicate.min_confidence))
                planner_query.required_tags = required_tags
            if not planner_query.preferred_tags:
                planner_query.preferred_tags = dict(
                    planner_query.target_predicate.score_weights or {planner_query.target_predicate.tag_name: 0.5}
                )
            if not planner_query.penalty_tags:
                planner_query.penalty_tags = dict(planner_query.target_predicate.penalty_weights or {})
            if "min_freshness" not in planner_query.temporal_constraints:
                planner_query.temporal_constraints["min_freshness"] = float(
                    planner_query.target_predicate.metadata.get("min_freshness", 0.0)
                )
            return planner_query
        if predicate is None:
            raise ValueError("Either planner_query or predicate must be provided")
        return PlannerQuery(
            intent_type=getattr(predicate.metadata, "intent_type", None) or "legacy_intent",
            target_predicate=predicate,
            required_tags={
                **dict(predicate.required_tag_thresholds),
                str(predicate.tag_name): float(predicate.min_confidence),
            },
            preferred_tags=dict(predicate.score_weights or {predicate.tag_name: 0.5}),
            penalty_tags=dict(predicate.penalty_weights or {}),
            temporal_constraints={"min_freshness": float(predicate.metadata.get("min_freshness", 0.0))},
        )

    def _dominant_semantic_tag(self, semantic_tags: Dict[str, float], token_label: Optional[str]) -> str:
        priority_weight = {
            "water_source": 1.60,
            "water_candidate": 1.40,
            "safe_rest_zone": 1.60,
            "rest_candidate": 1.40,
            "hazard_recovery_route": 1.45,
            "hazard_edge": 1.35,
            "goal_region": 1.20,
            "safe_exit": 0.85,
            "room_center": 0.80,
            "corridor": 0.80,
        }
        semantic_candidates = []
        for tag_name, value in dict(semantic_tags or {}).items():
            if str(tag_name) in self.relation_tag_names:
                continue
            weighted_value = float(value) * float(priority_weight.get(str(tag_name), 1.0))
            semantic_candidates.append((str(tag_name), float(weighted_value), float(value)))
        semantic_candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
        if semantic_candidates and semantic_candidates[0][1] > 0.20:
            return semantic_candidates[0][0]
        return "unknown_place" if token_label is None else str(token_label)

    def _semantic_similarity(self, left: Dict[str, float], right: Dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        tag_names = sorted(set(left.keys()) | set(right.keys()))
        if not tag_names:
            return 0.0
        left_vec = np.asarray([float(left.get(tag_name, 0.0)) for tag_name in tag_names], dtype=np.float64)
        right_vec = np.asarray([float(right.get(tag_name, 0.0)) for tag_name in tag_names], dtype=np.float64)
        denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(left_vec, right_vec) / denom)

    def _match_symbolic_node(
        self,
        dominant_tag: str,
        semantic_tags: Dict[str, float],
        pose_xy,
    ) -> Optional[int]:
        pose_arr = None if pose_xy is None else np.asarray(pose_xy, dtype=np.float64)
        best_score = None
        best_node_id = None
        for symbolic_node_id, node in self.symbolic_nodes.items():
            if str(node.dominant_tag) != str(dominant_tag):
                continue
            score = self._semantic_similarity(node.semantic_tag_confidence, semantic_tags)
            if pose_arr is not None and node.pose_mean is not None:
                distance = float(np.linalg.norm(pose_arr - np.asarray(node.pose_mean, dtype=np.float64)))
                if distance > 2.5:
                    continue
                if str(dominant_tag) in {"safe_exit", "room_center", "corridor"} and distance > 1.25:
                    continue
                score -= 0.15 * distance
            if best_score is None or score > best_score:
                best_score = score
                best_node_id = int(symbolic_node_id)
        if best_node_id is None:
            return None
        if best_score is None:
            return None
        if best_score >= 0.60:
            return int(best_node_id)
        if pose_arr is not None:
            node = self.symbolic_nodes[int(best_node_id)]
            if node.pose_mean is not None:
                distance = float(np.linalg.norm(pose_arr - np.asarray(node.pose_mean, dtype=np.float64)))
                if distance <= 1.5 and best_score >= 0.35:
                    return int(best_node_id)
        return None

    def _refresh_symbolic_node_stats(self, symbolic_node_id: int):
        node = self.symbolic_nodes[int(symbolic_node_id)]
        graph_members = [int(graph_node_id) for graph_node_id in node.member_graph_node_ids if int(graph_node_id) in self.graph.nodes]
        if not graph_members:
            return
        pose_means = []
        pose_vars = []
        utilities = []
        utility_fast = []
        utility_slow = []
        graph_confidence = []
        graph_freshness = []
        last_seen = []
        for graph_node_id in graph_members:
            snapshot = self.graph.symbolic_node_snapshot(int(graph_node_id))
            if snapshot.get("pose_mean") is not None:
                pose_means.append(np.asarray(snapshot["pose_mean"], dtype=np.float64))
            if snapshot.get("pose_var") is not None:
                pose_vars.append(np.asarray(snapshot["pose_var"], dtype=np.float64))
            utilities.append(float(self.graph.node_utility(int(graph_node_id))))
            utility_fast.append(float(snapshot.get("utility_fast", 0.0)))
            utility_slow.append(float(snapshot.get("utility_slow", 0.0)))
            graph_confidence.append(float(snapshot.get("confidence", 0.0)))
            graph_freshness.append(float(snapshot.get("freshness", 0.0)))
            last_seen.append(int(snapshot.get("last_seen_step", -1)))
        node.pose_mean = None if not pose_means else tuple(float(v) for v in np.mean(pose_means, axis=0))
        node.pose_var = None if not pose_vars else tuple(float(v) for v in np.mean(pose_vars, axis=0))
        node.utility_cached = float(np.mean(utilities)) if utilities else 0.0
        node.utility_fast = float(np.mean(utility_fast)) if utility_fast else 0.0
        node.utility_slow = float(np.mean(utility_slow)) if utility_slow else 0.0
        if graph_confidence:
            node.confidence = max(float(node.confidence), float(np.mean(graph_confidence)))
        if graph_freshness:
            node.freshness = float(np.mean(graph_freshness))
        if last_seen:
            node.last_seen_step = max(last_seen)

    def _refresh_concept_record(self, concept_tag: str, step_idx: int):
        member_nodes = [
            node
            for node in self.symbolic_nodes.values()
            if str(node.dominant_tag) == str(concept_tag)
        ]
        if not member_nodes:
            self.concept_records.pop(str(concept_tag), None)
            return
        record = self.concept_records.get(str(concept_tag))
        if record is None:
            record = ConceptRecord(concept_tag=str(concept_tag))
            self.concept_records[str(concept_tag)] = record
        record.member_symbolic_node_ids = [int(node.symbolic_node_id) for node in member_nodes]
        record.support_count = int(sum(max(1, int(node.visits)) for node in member_nodes))
        record.last_seen_step = int(step_idx)
        semantic_profile: Dict[str, float] = {}
        relation_profile: Dict[str, float] = {}
        freshness_vals = []
        confidence_vals = []
        for node in member_nodes:
            for tag_name, value in dict(node.semantic_tag_confidence).items():
                semantic_profile[str(tag_name)] = semantic_profile.get(str(tag_name), 0.0) + float(value)
            for relation_name, value in dict(node.relation_confidence).items():
                relation_profile[str(relation_name)] = relation_profile.get(str(relation_name), 0.0) + float(value)
            freshness_vals.append(float(node.freshness))
            confidence_vals.append(float(node.confidence))
        denom = float(max(1, len(member_nodes)))
        record.semantic_profile = {key: float(value / denom) for key, value in semantic_profile.items()}
        record.relation_profile = {key: float(value / denom) for key, value in relation_profile.items()}
        record.freshness_mean = float(np.mean(freshness_vals)) if freshness_vals else 0.0
        record.confidence_mean = float(np.mean(confidence_vals)) if confidence_vals else 0.0

    def _update_symbolic_node(
        self,
        graph_node_id: Optional[int],
        token_label: Optional[str],
        semantic_tags: Dict[str, float],
        pose_xy,
        step_idx: int,
    ) -> Optional[int]:
        if graph_node_id is None:
            return None
        graph_node_id = int(graph_node_id)
        existing = self.graph_node_to_symbolic_node.get(graph_node_id)
        if existing is not None:
            symbolic_node_id = int(existing)
        else:
            dominant_tag = self._dominant_semantic_tag(semantic_tags, token_label)
            symbolic_node_id = self._match_symbolic_node(
                dominant_tag=dominant_tag,
                semantic_tags=semantic_tags,
                pose_xy=pose_xy,
            )
            if symbolic_node_id is None:
                symbolic_node_id = int(len(self.symbolic_nodes))
                self.symbolic_nodes[symbolic_node_id] = SymbolicNode(
                    symbolic_node_id=int(symbolic_node_id),
                    label=str(dominant_tag),
                    dominant_tag=str(dominant_tag),
                )
        self.graph_node_to_symbolic_node[graph_node_id] = int(symbolic_node_id)
        node = self.symbolic_nodes[int(symbolic_node_id)]
        if graph_node_id not in node.member_graph_node_ids:
            node.member_graph_node_ids.append(int(graph_node_id))
        phrase_signature = [] if token_label is None else [str(token_label)]
        if phrase_signature and phrase_signature not in node.phrase_signatures:
            node.phrase_signatures.append(list(phrase_signature))
        prev_last_seen = int(node.last_seen_step)
        node.visits += 1
        node.last_seen_step = int(step_idx)
        node.freshness = 1.0 if prev_last_seen < 0 else self._compute_freshness(step_idx, prev_last_seen)
        node.confidence = max(float(node.confidence), self._compute_confidence(node.visits))
        for tag_name, value in dict(semantic_tags or {}).items():
            tag_value = float(value)
            if tag_value <= 0.0:
                continue
            node.semantic_tag_counts[str(tag_name)] = int(node.semantic_tag_counts.get(str(tag_name), 0)) + 1
            count = int(node.semantic_tag_counts[str(tag_name)])
            prev = float(node.semantic_tag_confidence.get(str(tag_name), 0.0))
            node.semantic_tag_confidence[str(tag_name)] = ((prev * float(count - 1)) + tag_value) / float(count)
            if str(tag_name) in self.relation_tag_names:
                node.relation_confidence[str(tag_name)] = float(node.semantic_tag_confidence[str(tag_name)])
        self._refresh_symbolic_node_stats(int(symbolic_node_id))
        self._refresh_concept_record(node.dominant_tag, step_idx=step_idx)
        return int(symbolic_node_id)

    def _update_relation_records(
        self,
        symbolic_node_id: Optional[int],
        semantic_tags: Dict[str, float],
        step_idx: int,
    ):
        if symbolic_node_id is None:
            return
        symbolic_node_id = int(symbolic_node_id)
        for relation_name in sorted(self.relation_tag_names):
            relation_value = float(dict(semantic_tags or {}).get(str(relation_name), 0.0))
            if relation_value <= 0.0:
                continue
            relation_bucket = self.relation_records.setdefault(str(relation_name), {})
            relation_record = relation_bucket.get(int(symbolic_node_id))
            if relation_record is None:
                relation_record = RelationRecord(
                    relation_name=str(relation_name),
                    symbolic_node_id=int(symbolic_node_id),
                    target_concept_tag=self.relation_target_concepts.get(str(relation_name)),
                )
                relation_bucket[int(symbolic_node_id)] = relation_record
            prev_last_seen = int(relation_record.last_seen_step)
            relation_record.support_count += 1
            relation_record.last_seen_step = int(step_idx)
            relation_record.freshness = (
                1.0 if prev_last_seen < 0 else self._compute_freshness(step_idx, prev_last_seen)
            )
            relation_record.confidence = (
                (float(relation_record.confidence) * float(relation_record.support_count - 1)) + relation_value
            ) / float(relation_record.support_count)
            self.symbolic_nodes[int(symbolic_node_id)].relation_confidence[str(relation_name)] = float(
                relation_record.confidence
            )

    def _update_symbolic_transition(
        self,
        src_symbolic_node_id: Optional[int],
        dst_symbolic_node_id: Optional[int],
        step_idx: int,
        transition_success: Optional[float],
        transition_risk: Optional[float],
        transition_cost: Optional[float],
        context_label: Optional[str] = None,
    ):
        if src_symbolic_node_id is None or dst_symbolic_node_id is None:
            return
        if int(src_symbolic_node_id) == int(dst_symbolic_node_id):
            return
        alpha_fast = float(getattr(self.graph, "alpha_fast", 0.15))
        alpha_slow = float(getattr(self.graph, "alpha_slow", 0.02))
        bucket = self.symbolic_transitions.setdefault(int(src_symbolic_node_id), {})
        transition = bucket.get(int(dst_symbolic_node_id))
        if transition is None:
            transition = SymbolicTransition(src=int(src_symbolic_node_id), dst=int(dst_symbolic_node_id))
            bucket[int(dst_symbolic_node_id)] = transition
        prev_last_seen = int(transition.last_seen_step)
        transition.last_seen_step = int(step_idx)
        transition.freshness = 1.0 if prev_last_seen < 0 else self._compute_freshness(step_idx, prev_last_seen)
        count = int(transition.context_histogram.get("__count__", 0)) + 1
        transition.context_histogram["__count__"] = int(count)
        transition.confidence = self._compute_confidence(count)
        if transition_success is not None:
            transition.success_fast = self._ema_update(transition.success_fast, transition_success, alpha_fast)
            transition.success_slow = self._ema_update(transition.success_slow, transition_success, alpha_slow)
        if transition_risk is not None:
            transition.risk_fast = self._ema_update(transition.risk_fast, transition_risk, alpha_fast)
            transition.risk_slow = self._ema_update(transition.risk_slow, transition_risk, alpha_slow)
        if transition_cost is not None:
            transition.cost_fast = self._ema_update(transition.cost_fast, transition_cost, alpha_fast)
            transition.cost_slow = self._ema_update(transition.cost_slow, transition_cost, alpha_slow)
        if context_label:
            transition.context_histogram[str(context_label)] = int(
                transition.context_histogram.get(str(context_label), 0)
            ) + 1

    def _apply_decay(self, step_idx: int):
        for node in self.symbolic_nodes.values():
            if int(node.last_seen_step) < 0:
                continue
            node.freshness = self._compute_freshness(int(step_idx), int(node.last_seen_step))
            stale_factor = 1.0 if node.freshness >= 0.05 else max(0.25, node.freshness / 0.05)
            node.utility_cached *= stale_factor
        for bucket in self.relation_records.values():
            for relation_record in bucket.values():
                if int(relation_record.last_seen_step) < 0:
                    continue
                relation_record.freshness = self._compute_freshness(int(step_idx), int(relation_record.last_seen_step))
        for bucket in self.symbolic_transitions.values():
            for transition in bucket.values():
                if int(transition.last_seen_step) < 0:
                    continue
                transition.freshness = self._compute_freshness(int(step_idx), int(transition.last_seen_step))

    def observe(self, scene_token=None, scene_state=None, agent_state=None, step_info: Optional[Dict[str, Any]] = None):
        step_info = dict(step_info or {})
        phase = str(step_info.get("phase", "pre_action"))
        step_idx = int(step_info.get("step_idx", 0))
        semantic_tags = dict(step_info.get("semantic_tags", {}))
        if scene_token is not None:
            semantic_tags = dict(getattr(scene_token, "semantic_tags", {}) or semantic_tags)
        token_label = step_info.get("token_label")
        if token_label is None and scene_token is not None:
            token_label = str(getattr(scene_token, "token_type", None))

        if phase == "pre_action":
            token_id = step_info.get("token_id")
            prev_graph_node_id = self.graph.current_phrase_id
            is_new_graph_node = False
            if token_id is not None:
                is_new_graph_node = bool(self.graph.update_token(int(token_id), step_idx))
            current_graph_node_id = self.graph.current_phrase_id
            self.graph.observe(
                step_idx,
                pose_xy=step_info.get("pose_xy"),
                goal_xy=step_info.get("goal_xy"),
                reward=float(step_info.get("reward", 0.0)),
                progress=step_info.get("progress"),
                risk=step_info.get("risk"),
                success=step_info.get("success"),
                comfort_cost=step_info.get("comfort_cost"),
                goal_alignment=step_info.get("goal_alignment"),
                phase_label=token_label,
            )
            if semantic_tags and current_graph_node_id is not None:
                self.graph.observe_semantics(current_graph_node_id, semantic_tags)
            symbolic_node_id = self._update_symbolic_node(
                graph_node_id=current_graph_node_id,
                token_label=None if token_label is None else str(token_label),
                semantic_tags=semantic_tags,
                pose_xy=step_info.get("pose_xy"),
                step_idx=step_idx,
            )
            self._update_relation_records(symbolic_node_id, semantic_tags=semantic_tags, step_idx=step_idx)
            self._apply_decay(step_idx=step_idx)
            self._pending_pre_step[int(step_idx)] = {
                "previous_graph_node_id": None if prev_graph_node_id is None else int(prev_graph_node_id),
                "current_graph_node_id": None if current_graph_node_id is None else int(current_graph_node_id),
                "symbolic_node_id": None if symbolic_node_id is None else int(symbolic_node_id),
                "semantic_tags": dict(semantic_tags),
                "token_label": None if token_label is None else str(token_label),
                "pose_xy": None if step_info.get("pose_xy") is None else np.asarray(step_info.get("pose_xy")).tolist(),
            }
            return {
                "is_new_graph_node": bool(is_new_graph_node),
                "previous_graph_node_id": None if prev_graph_node_id is None else int(prev_graph_node_id),
                "current_graph_node_id": None if current_graph_node_id is None else int(current_graph_node_id),
                "symbolic_node_id": None if symbolic_node_id is None else int(symbolic_node_id),
            }

        if phase == "post_action":
            pending = dict(self._pending_pre_step.pop(int(step_idx), {}))
            prev_graph_node_id = step_info.get("previous_graph_node_id", pending.get("previous_graph_node_id"))
            current_graph_node_id = step_info.get("current_graph_node_id", pending.get("current_graph_node_id"))
            symbolic_node_id = self.graph_node_to_symbolic_node.get(
                None if current_graph_node_id is None else int(current_graph_node_id)
            )
            self.graph.observe(
                step_idx,
                pose_xy=step_info.get("pose_xy"),
                goal_xy=step_info.get("goal_xy"),
                reward=float(step_info.get("reward", 0.0)),
                progress=step_info.get("progress"),
                risk=step_info.get("risk"),
                success=step_info.get("success"),
                comfort_cost=step_info.get("comfort_cost"),
                goal_alignment=step_info.get("goal_alignment"),
                phase_label=token_label,
            )
            self.graph.observe_transition(
                src=None if prev_graph_node_id is None else int(prev_graph_node_id),
                dst=None if current_graph_node_id is None else int(current_graph_node_id),
                step_idx=step_idx,
                transition_success=step_info.get("transition_success"),
                transition_risk=step_info.get("transition_risk"),
                transition_cost=step_info.get("transition_cost"),
            )
            if symbolic_node_id is not None:
                self._update_symbolic_node(
                    graph_node_id=None if current_graph_node_id is None else int(current_graph_node_id),
                    token_label=None if token_label is None else str(token_label),
                    semantic_tags=semantic_tags or dict(pending.get("semantic_tags", {})),
                    pose_xy=step_info.get("pose_xy"),
                    step_idx=step_idx,
                )
            prev_symbolic_node_id = self.graph_node_to_symbolic_node.get(
                None if prev_graph_node_id is None else int(prev_graph_node_id)
            )
            self._update_symbolic_transition(
                src_symbolic_node_id=prev_symbolic_node_id,
                dst_symbolic_node_id=symbolic_node_id,
                step_idx=step_idx,
                transition_success=step_info.get("transition_success"),
                transition_risk=step_info.get("transition_risk"),
                transition_cost=step_info.get("transition_cost"),
                context_label=step_info.get("context_label")
                or self._agent_state_snapshot(agent_state).get("task_phase")
                or token_label,
            )
            self._apply_decay(step_idx=step_idx)
            self.record_step(
                step_idx=int(step_idx),
                node_id=None if current_graph_node_id is None else int(current_graph_node_id),
                symbolic_node_id=None if symbolic_node_id is None else int(symbolic_node_id),
                token_type=None if token_label is None else str(token_label),
                active_intent=step_info.get("active_intent"),
                intent_reason=str(step_info.get("intent_reason", "")),
                agent_state=agent_state,
                semantic_tags=semantic_tags or dict(pending.get("semantic_tags", {})),
                observations=dict(step_info.get("observations", {})),
                outcome=dict(step_info.get("outcome", {})),
            )
            return {
                "previous_graph_node_id": None if prev_graph_node_id is None else int(prev_graph_node_id),
                "current_graph_node_id": None if current_graph_node_id is None else int(current_graph_node_id),
                "symbolic_node_id": None if symbolic_node_id is None else int(symbolic_node_id),
            }

        raise ValueError(f"Unsupported observe phase: {phase}")

    def record_step(
        self,
        step_idx: int,
        node_id: Optional[int],
        token_type: Optional[str],
        active_intent: Optional[str],
        symbolic_node_id: Optional[int] = None,
        intent_reason: str = "",
        agent_state=None,
        semantic_tags: Optional[Dict[str, float]] = None,
        observations: Optional[Dict[str, Any]] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ):
        if self.active_episode is None:
            return
        intent_value = None if active_intent is None else str(active_intent)
        if node_id is not None:
            self.active_episode.node_sequence.append(int(node_id))
        if intent_value is not None:
            self.active_episode.intent_sequence.append(intent_value)
        state_snapshot = self._agent_state_snapshot(agent_state)
        self.active_episode.state_trace.append(state_snapshot)
        self.active_episode.step_records.append(
            EpisodeStepRecord(
                step_idx=int(step_idx),
                node_id=None if node_id is None else int(node_id),
                token_type=None if token_type is None else str(token_type),
                active_intent=intent_value,
                symbolic_node_id=None if symbolic_node_id is None else int(symbolic_node_id),
                intent_reason=str(intent_reason),
                agent_state=state_snapshot,
                semantic_tags=dict(semantic_tags or {}),
                observations=dict(observations or {}),
                outcome=dict(outcome or {}),
            )
        )

    def _annotate_outcome_windows(self, episode: EpisodeRecord, window: int = 10):
        step_records = episode.step_records
        for idx, step_record in enumerate(step_records):
            tail = step_records[idx : min(len(step_records), idx + int(window))]
            success_within = 1 if any(int(row.outcome.get("success_signal", 0)) > 0 for row in tail) else 0
            progress_after = float(sum(float(row.outcome.get("delta_distance", 0.0)) for row in tail))
            entered_hazard_again = 1 if any(str(row.agent_state.get("task_phase", "")) == "hazard_navigation" for row in tail[1:]) else 0
            restored_safety = 1 if any(
                float(row.semantic_tags.get("adjacent_hazard", 0.0)) < 0.2
                and float(row.semantic_tags.get("hazard_edge", 0.0)) < 0.2
                for row in tail
            ) else 0
            resource_reached = 1 if any(
                int(row.outcome.get("water_task_success", 0)) > 0
                or int(row.outcome.get("rest_task_success", 0)) > 0
                or int(row.outcome.get("success_signal", 0)) > 0
                for row in tail
            ) else 0
            step_record.outcome_window = OutcomeWindow(
                success_within_10_steps=int(success_within),
                goal_progress_after_10_steps=float(progress_after),
                entered_hazard_again=int(entered_hazard_again),
                restored_safety=int(restored_safety),
                resource_reached_after_10_steps=int(resource_reached),
            )

    def _update_symbolic_outcome_summaries(self, episode: EpisodeRecord):
        for step_record in episode.step_records:
            symbolic_node_id = step_record.symbolic_node_id
            if symbolic_node_id is None or int(symbolic_node_id) not in self.symbolic_nodes:
                continue
            intent_key = str(step_record.active_intent or "none")
            node = self.symbolic_nodes[int(symbolic_node_id)]
            summary = node.intent_outcome_summary.setdefault(
                intent_key,
                {
                    "observations": 0.0,
                    "success_within_10_steps": 0.0,
                    "goal_progress_after_10_steps": 0.0,
                    "entered_hazard_again": 0.0,
                    "restored_safety": 0.0,
                    "resource_reached_after_10_steps": 0.0,
                },
            )
            count = float(summary["observations"]) + 1.0
            summary["observations"] = float(count)
            summary["success_within_10_steps"] = (
                (float(summary["success_within_10_steps"]) * (count - 1.0))
                + float(step_record.outcome_window.success_within_10_steps)
            ) / count
            summary["goal_progress_after_10_steps"] = (
                (float(summary["goal_progress_after_10_steps"]) * (count - 1.0))
                + float(step_record.outcome_window.goal_progress_after_10_steps)
            ) / count
            summary["entered_hazard_again"] = (
                (float(summary["entered_hazard_again"]) * (count - 1.0))
                + float(step_record.outcome_window.entered_hazard_again)
            ) / count
            summary["restored_safety"] = (
                (float(summary["restored_safety"]) * (count - 1.0))
                + float(step_record.outcome_window.restored_safety)
            ) / count
            summary["resource_reached_after_10_steps"] = (
                (float(summary["resource_reached_after_10_steps"]) * (count - 1.0))
                + float(step_record.outcome_window.resource_reached_after_10_steps)
            ) / count

    def finalize_episode(self, outcome: Optional[Dict[str, Any]] = None):
        if self.active_episode is None:
            return
        self._annotate_outcome_windows(self.active_episode)
        self._update_symbolic_outcome_summaries(self.active_episode)
        self.active_episode.outcome = dict(outcome or {})
        self.episode_records.append(self.active_episode)
        self.active_episode = None

    def _symbolic_node_for_graph_node(self, node_id: Optional[int]) -> Optional[SymbolicNode]:
        symbolic_node_id = self.graph_node_to_symbolic_node.get(None if node_id is None else int(node_id))
        if symbolic_node_id is None:
            return None
        return self.symbolic_nodes.get(int(symbolic_node_id))

    def _state_bonus(self, symbolic_node: SymbolicNode, planner_query: PlannerQuery, agent_state_snapshot: Dict[str, Any]) -> float:
        if not agent_state_snapshot:
            return 0.0
        intent_name = str(getattr(planner_query.intent_type, "value", planner_query.intent_type))
        thirst = float(agent_state_snapshot.get("thirst", 0.0))
        energy = float(agent_state_snapshot.get("energy", 1.0))
        risk_budget = float(agent_state_snapshot.get("risk_budget", 1.0))
        task_phase = str(agent_state_snapshot.get("task_phase", ""))
        confs = dict(symbolic_node.semantic_tag_confidence)
        if intent_name == "find_water_source":
            return float(
                (0.80 * thirst * confs.get("water_source", 0.0))
                + (0.45 * thirst * confs.get("near_water", 0.0))
                + (0.20 * max(0.0, 0.5 - risk_budget) * confs.get("water_candidate", 0.0))
            )
        if intent_name == "find_safe_rest_zone":
            energy_need = max(0.0, 1.0 - energy)
            return float(
                (0.95 * energy_need * confs.get("safe_rest_zone", 0.0))
                + (0.40 * energy_need * confs.get("open_safe_rest_zone", 0.0))
                - (0.15 * energy_need * confs.get("corridor", 0.0))
            )
        if intent_name == "hazard_recovery_exit":
            hazard_pressure = 1.0 if task_phase == "hazard_navigation" else max(0.0, 0.5 - risk_budget)
            return float(
                (0.85 * hazard_pressure * confs.get("hazard_recovery_route", 0.0))
                + (0.35 * hazard_pressure * confs.get("post_hazard_goal_rejoin", 0.0))
                - (0.30 * hazard_pressure * confs.get("hazard_edge", 0.0))
            )
        if intent_name == "find_goal_region":
            return float(
                (0.40 * confs.get("goal_region", 0.0))
                + (0.15 * confs.get("post_hazard_goal_rejoin", 0.0))
                + (0.10 * risk_budget * confs.get("safe_exit", 0.0))
            )
        return 0.0

    def _episodic_bonus(self, symbolic_node: SymbolicNode, planner_query: PlannerQuery) -> float:
        intent_name = str(getattr(planner_query.intent_type, "value", planner_query.intent_type))
        summary = symbolic_node.intent_outcome_summary.get(intent_name)
        if not summary:
            return 0.0
        return float(
            0.35 * float(summary.get("success_within_10_steps", 0.0))
            + 0.15 * float(summary.get("goal_progress_after_10_steps", 0.0))
            + 0.15 * float(summary.get("restored_safety", 0.0))
            + 0.10 * float(summary.get("resource_reached_after_10_steps", 0.0))
            - 0.30 * float(summary.get("entered_hazard_again", 0.0))
        )

    def _query_required_tags(self, planner_query: PlannerQuery) -> Dict[str, float]:
        planner_query = self._ensure_query(planner_query=planner_query)
        return dict(planner_query.required_tags)

    def _query_preferred_tags(self, planner_query: PlannerQuery) -> Dict[str, float]:
        planner_query = self._ensure_query(planner_query=planner_query)
        return dict(planner_query.preferred_tags)

    def _query_penalty_tags(self, planner_query: PlannerQuery) -> Dict[str, float]:
        planner_query = self._ensure_query(planner_query=planner_query)
        return dict(planner_query.penalty_tags)

    def _query_target_tag(self, planner_query: PlannerQuery) -> str:
        planner_query = self._ensure_query(planner_query=planner_query)
        return str(planner_query.target_predicate.tag_name)

    def _required_match_mode(self, planner_query: PlannerQuery) -> str:
        planner_query = self._ensure_query(planner_query=planner_query)
        mode = planner_query.metadata.get("required_match_mode")
        if mode is None and getattr(planner_query, "target_predicate", None) is not None:
            mode = planner_query.target_predicate.metadata.get("required_match_mode")
        return str(mode or "all")

    def _required_matches_profile(self, profile: Dict[str, float], planner_query: PlannerQuery) -> bool:
        required_tags = self._query_required_tags(planner_query)
        if not required_tags:
            return True
        satisfied = [
            float(profile.get(str(tag_name), 0.0)) >= float(min_value)
            for tag_name, min_value in required_tags.items()
        ]
        mode = self._required_match_mode(planner_query)
        if mode == "any":
            return any(satisfied)
        if mode == "at_least":
            min_matches = int(planner_query.metadata.get("min_required_matches", 1))
            return int(sum(1 for value in satisfied if value)) >= max(1, min_matches)
        return all(satisfied)

    def _query_alignment_bonus(
        self,
        profile: Dict[str, float],
        planner_query: PlannerQuery,
        dominant_tag: Optional[str] = None,
    ) -> float:
        planner_query = self._ensure_query(planner_query=planner_query)
        target_tag = self._query_target_tag(planner_query)
        target_conf = float(profile.get(target_tag, 0.0))
        bonus = 0.40 * target_conf
        if dominant_tag is not None and str(dominant_tag) == target_tag:
            bonus += 0.20
        if (
            dominant_tag is not None
            and str(dominant_tag) in {"safe_exit", "room_center", "corridor"}
            and target_conf < 0.20
        ):
            bonus -= 0.08
        return float(bonus)

    def node_matches_query(self, node_id: int, planner_query: PlannerQuery) -> bool:
        planner_query = self._ensure_query(planner_query=planner_query)
        symbolic_node = self._symbolic_node_for_graph_node(int(node_id))
        if symbolic_node is None:
            return False
        min_visits = int(planner_query.metadata.get("min_visits_override", getattr(self.graph, "min_goal_visits", 1)))
        if int(symbolic_node.visits) < min_visits:
            return False
        confs = self._combined_confidence(int(node_id), symbolic_node)
        if not self._required_matches_profile(confs, planner_query):
            return False
        min_freshness = float(planner_query.temporal_constraints.get("min_freshness", 0.0))
        min_confidence = float(planner_query.temporal_constraints.get("min_confidence", 0.0))
        if float(symbolic_node.freshness) < min_freshness:
            return False
        if float(symbolic_node.confidence) < min_confidence:
            return False
        return True

    def node_matches_intent(self, node_id: int, predicate) -> bool:
        planner_query = self._ensure_query(predicate=predicate)
        return self.node_matches_query(int(node_id), planner_query)

    def node_intent_score(
        self,
        node_id: int,
        predicate: Optional[SemanticTargetPredicate] = None,
        planner_query: Optional[PlannerQuery] = None,
        agent_state=None,
    ) -> float:
        planner_query = self._ensure_query(planner_query=planner_query, predicate=predicate)
        symbolic_node = self._symbolic_node_for_graph_node(int(node_id))
        if symbolic_node is None:
            return float(self.graph.node_utility(int(node_id)))
        confs = self._combined_confidence(int(node_id), symbolic_node)
        preferred_tags = self._query_preferred_tags(planner_query)
        penalty_tags = self._query_penalty_tags(planner_query)
        tag_bonus = 0.0
        tag_penalty = 0.0
        for tag_name, weight in preferred_tags.items():
            tag_bonus += float(weight) * float(confs.get(str(tag_name), 0.0))
        for tag_name, weight in penalty_tags.items():
            tag_penalty += float(weight) * float(confs.get(str(tag_name), 0.0))
        agent_state_snapshot = self._agent_state_snapshot(agent_state)
        if not agent_state_snapshot:
            agent_state_snapshot = dict(planner_query.metadata.get("agent_state_snapshot", {}))
        temporal_bonus = (
            0.20 * float(symbolic_node.freshness)
            + 0.15 * float(symbolic_node.confidence)
        )
        return float(
            0.45 * float(self.graph.node_utility(int(node_id)))
            + 0.55 * float(symbolic_node.utility_cached)
            + tag_bonus
            + temporal_bonus
            + self._query_alignment_bonus(confs, planner_query, dominant_tag=str(symbolic_node.dominant_tag))
            + self._state_bonus(symbolic_node, planner_query, agent_state_snapshot)
            + self._episodic_bonus(symbolic_node, planner_query)
            - tag_penalty
        )

    def explain_node_intent(
        self,
        node_id: int,
        planner_query: PlannerQuery,
        agent_state=None,
    ) -> Dict[str, Any]:
        planner_query = self._ensure_query(planner_query=planner_query)
        symbolic_node = self._symbolic_node_for_graph_node(int(node_id))
        confs = {} if symbolic_node is None else self._combined_confidence(int(node_id), symbolic_node)
        score_weights = self._query_preferred_tags(planner_query)
        penalty_weights = self._query_penalty_tags(planner_query)
        score_contributions = {
            str(tag_name): float(weight) * float(confs.get(tag_name, 0.0))
            for tag_name, weight in score_weights.items()
        }
        penalty_contributions = {
            str(tag_name): float(weight) * float(confs.get(tag_name, 0.0))
            for tag_name, weight in penalty_weights.items()
        }
        return {
            "node_id": int(node_id),
            "symbolic_node_id": None if symbolic_node is None else int(symbolic_node.symbolic_node_id),
            "intent_type": str(getattr(planner_query.intent_type, "value", planner_query.intent_type)),
            "query_tag_name": str(planner_query.target_predicate.tag_name),
            "matches": bool(self.node_matches_query(int(node_id), planner_query)),
            "base_utility": float(self.graph.node_utility(int(node_id))),
            "intent_score": float(self.node_intent_score(int(node_id), planner_query=planner_query, agent_state=agent_state)),
            "semantic_tag_confidence": confs,
            "required_tag_thresholds": self._query_required_tags(planner_query),
            "score_weights": score_weights,
            "penalty_weights": penalty_weights,
            "score_contributions": score_contributions,
            "penalty_contributions": penalty_contributions,
            "freshness": 0.0 if symbolic_node is None else float(symbolic_node.freshness),
            "confidence": 0.0 if symbolic_node is None else float(symbolic_node.confidence),
            "agent_state": self._agent_state_snapshot(agent_state),
            "episodic_summary": {} if symbolic_node is None else dict(
                symbolic_node.intent_outcome_summary.get(
                    str(getattr(planner_query.intent_type, "value", planner_query.intent_type)),
                    {},
                )
            ),
        }

    def _combined_confidence(self, node_id: int, symbolic_node: Optional[SymbolicNode]) -> Dict[str, float]:
        confs = {}
        if symbolic_node is not None:
            confs.update({str(key): float(value) for key, value in symbolic_node.semantic_tag_confidence.items()})
        graph_confs = self.graph.nodes.get(int(node_id), {}).get("semantic_tag_confidence", {})
        for tag_name, value in dict(graph_confs).items():
            confs[str(tag_name)] = max(float(confs.get(str(tag_name), 0.0)), float(value))
        return confs

    def _representative_graph_node(
        self,
        symbolic_node_id: int,
        current_node_id: Optional[int] = None,
        planner_query: Optional[PlannerQuery] = None,
    ) -> Optional[int]:
        symbolic_node = self.symbolic_nodes.get(int(symbolic_node_id))
        if symbolic_node is None:
            return None
        candidates = []
        for graph_node_id in symbolic_node.member_graph_node_ids:
            if int(graph_node_id) not in self.graph.nodes:
                continue
            if current_node_id is not None and int(graph_node_id) == int(current_node_id) and len(symbolic_node.member_graph_node_ids) > 1:
                continue
            if planner_query is not None:
                score = float(self.node_intent_score(int(graph_node_id), planner_query=planner_query))
            else:
                score = float(self.graph.node_utility(int(graph_node_id)))
            candidates.append((score, int(graph_node_id)))
        if not candidates:
            if symbolic_node.member_graph_node_ids:
                return int(symbolic_node.member_graph_node_ids[0])
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return int(candidates[0][1])

    def candidate_nodes_for_query(
        self,
        planner_query: PlannerQuery,
        top_k: int = 5,
        current_node_id: Optional[int] = None,
    ) -> List[int]:
        planner_query = self._ensure_query(planner_query=planner_query)
        use_concept_recall = bool(planner_query.metadata.get("use_concept_recall", True))
        if not use_concept_recall:
            ranked = []
            for symbolic_node_id in sorted(self.symbolic_nodes.keys()):
                representative_graph_node = self._representative_graph_node(
                    symbolic_node_id,
                    current_node_id=current_node_id,
                    planner_query=planner_query,
                )
                if representative_graph_node is None:
                    continue
                if not self.node_matches_query(representative_graph_node, planner_query):
                    continue
                score = self.node_intent_score(representative_graph_node, planner_query=planner_query)
                ranked.append((float(score), int(representative_graph_node), int(symbolic_node_id)))
            ranked.sort(key=lambda item: item[0], reverse=True)
            selected = [int(graph_node_id) for _, graph_node_id, _ in ranked[: max(1, int(top_k))]]
            self.last_concept_query_debug = {
                "mode": "node_only",
                "clusters": [],
                "cluster_membership": {str(graph_node_id): str(symbolic_node_id) for _, graph_node_id, symbolic_node_id in ranked},
                "selected_node_ids": selected,
            }
            return selected

        concept_candidates = self.concept_clusters_for_intent(planner_query, top_k=max(2, int(top_k)))
        if not concept_candidates:
            self.last_concept_query_debug = {"mode": "concept_recall_empty", "clusters": []}
            return self.candidate_nodes_for_query(
                planner_query=PlannerQuery(
                    intent_type=planner_query.intent_type,
                    target_predicate=planner_query.target_predicate,
                    fallback_goal_xy=planner_query.fallback_goal_xy,
                    metadata={**dict(planner_query.metadata), "use_concept_recall": False},
                    required_tags=dict(planner_query.required_tags),
                    preferred_tags=dict(planner_query.preferred_tags),
                    penalty_tags=dict(planner_query.penalty_tags),
                    state_constraints=dict(planner_query.state_constraints),
                    temporal_constraints=dict(planner_query.temporal_constraints),
                    fallback_mode=str(planner_query.fallback_mode),
                ),
                top_k=top_k,
                current_node_id=current_node_id,
            )

        ranked_nodes = []
        seen_graph_nodes = set()
        cluster_membership = {}
        for cluster in concept_candidates:
            cluster_score = float(cluster["cluster_score"])
            for symbolic_node_id in cluster["member_node_ids"]:
                representative_graph_node = self._representative_graph_node(
                    int(symbolic_node_id),
                    current_node_id=current_node_id,
                    planner_query=planner_query,
                )
                if representative_graph_node is None:
                    continue
                if representative_graph_node in seen_graph_nodes:
                    continue
                if not self.node_matches_query(representative_graph_node, planner_query):
                    continue
                node_score = float(self.node_intent_score(representative_graph_node, planner_query=planner_query))
                combined_score = float(node_score + (0.35 * cluster_score))
                ranked_nodes.append((combined_score, int(representative_graph_node), str(cluster["cluster_id"])))
                cluster_membership[str(representative_graph_node)] = str(cluster["cluster_id"])
                seen_graph_nodes.add(int(representative_graph_node))
        ranked_nodes.sort(key=lambda item: item[0], reverse=True)
        selected = [int(graph_node_id) for _, graph_node_id, _ in ranked_nodes[: max(1, int(top_k))]]
        self.last_concept_query_debug = {
            "mode": "concept_recall",
            "clusters": concept_candidates,
            "cluster_membership": cluster_membership,
            "selected_node_ids": selected,
        }
        return selected

    def candidate_nodes_for_intent(self, predicate, top_k: int = 5) -> List[int]:
        planner_query = self._ensure_query(predicate=predicate)
        return self.candidate_nodes_for_query(planner_query=planner_query, top_k=top_k)

    def query_nodes(
        self,
        planner_query: PlannerQuery,
        agent_state=None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        planner_query = self._ensure_query(planner_query=planner_query)
        if agent_state is not None:
            planner_query.metadata["agent_state_snapshot"] = self._agent_state_snapshot(agent_state)
        candidate_ids = self.candidate_nodes_for_query(
            planner_query=planner_query,
            top_k=top_k,
        )
        return [
            self.explain_node_intent(node_id=int(node_id), planner_query=planner_query, agent_state=agent_state)
            for node_id in candidate_ids
        ]

    def query_paths(
        self,
        planner_query: Optional[PlannerQuery],
        current_node_id: Optional[int],
        planner,
        horizon: int = 4,
        top_k: int = 5,
    ):
        if (
            planner_query is not None
            and current_node_id is not None
            and bool(planner_query.metadata.get("use_concept_planning", False))
        ):
            concept_plans = self._concept_rollout(
                planner_query=planner_query,
                current_node_id=int(current_node_id),
                planner=planner,
                horizon=horizon,
                top_k=top_k,
            )
            if concept_plans:
                return concept_plans
        return planner.rollout(
            graph=self,
            current_node_id=current_node_id,
            horizon=horizon,
            top_k=top_k,
            planner_query=planner_query,
        )

    def symbolic_node_snapshot(self, symbolic_node_id: int) -> Dict[str, Any]:
        node = self.symbolic_nodes[int(symbolic_node_id)]
        return {
            "symbolic_node_id": int(node.symbolic_node_id),
            "label": str(node.label),
            "dominant_tag": str(node.dominant_tag),
            "phrase_signature": list(node.phrase_signatures[0]) if node.phrase_signatures else [],
            "phrase_signatures": [list(signature) for signature in node.phrase_signatures],
            "member_graph_node_ids": [int(graph_node_id) for graph_node_id in node.member_graph_node_ids],
            "semantic_tag_confidence": dict(node.semantic_tag_confidence),
            "semantic_tag_counts": dict(node.semantic_tag_counts),
            "pose_mean": None if node.pose_mean is None else [float(v) for v in node.pose_mean],
            "pose_var": None if node.pose_var is None else [float(v) for v in node.pose_var],
            "visits": int(node.visits),
            "freshness": float(node.freshness),
            "confidence": float(node.confidence),
            "utility_fast": float(node.utility_fast),
            "utility_slow": float(node.utility_slow),
            "utility_cached": float(node.utility_cached),
            "last_seen_step": int(node.last_seen_step),
            "relation_confidence": dict(node.relation_confidence),
            "intent_outcome_summary": dict(node.intent_outcome_summary),
        }

    def symbolic_transition_snapshot(self, src: int, dst: int) -> Dict[str, Any]:
        transition = self.symbolic_transitions.get(int(src), {}).get(int(dst))
        if transition is None:
            return {
                "src": int(src),
                "dst": int(dst),
            }
        return {
            "src": int(transition.src),
            "dst": int(transition.dst),
            "success_fast": float(transition.success_fast),
            "success_slow": float(transition.success_slow),
            "risk_fast": float(transition.risk_fast),
            "risk_slow": float(transition.risk_slow),
            "cost_fast": float(transition.cost_fast),
            "cost_slow": float(transition.cost_slow),
            "freshness": float(transition.freshness),
            "confidence": float(transition.confidence),
            "last_seen_step": int(transition.last_seen_step),
            "context_histogram": dict(transition.context_histogram),
        }

    def retrieve_concept_instances(
        self,
        concept_tag: str,
        min_confidence: float = 0.5,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        scored = []
        for symbolic_node_id, node in self.symbolic_nodes.items():
            conf = float(node.semantic_tag_confidence.get(str(concept_tag), 0.0))
            if conf < float(min_confidence):
                continue
            scored.append((conf, int(symbolic_node_id)))
        scored.sort(reverse=True)
        return [
            {
                **self.symbolic_node_snapshot(symbolic_node_id),
                "concept_tag": str(concept_tag),
            }
            for _, symbolic_node_id in scored[: max(1, int(top_k))]
        ]

    def _cluster_key(self, node_snapshot: Dict[str, Any]) -> Optional[str]:
        confs = dict(node_snapshot.get("semantic_tag_confidence", {}))
        semantic_candidates = [
            (str(tag_name), float(value))
            for tag_name, value in confs.items()
            if str(tag_name) not in self.relation_tag_names and float(value) > 0.20
        ]
        if not semantic_candidates:
            return None
        semantic_candidates.sort(key=lambda item: item[1], reverse=True)
        concept_tag = semantic_candidates[0][0]
        if concept_tag in {"safe_exit", "room_center", "corridor"}:
            relation_candidates = [
                (
                    str(self.relation_target_concepts.get(str(tag_name), "")),
                    float(confs.get(str(tag_name), 0.0)),
                )
                for tag_name in self.relation_tag_names
                if str(tag_name) in self.relation_target_concepts
            ]
            relation_candidates = [
                (target_tag, value)
                for target_tag, value in relation_candidates
                if target_tag and value >= 0.35
            ]
            relation_candidates.sort(key=lambda item: item[1], reverse=True)
            if relation_candidates:
                concept_tag = relation_candidates[0][0]
        active_relations = sorted(
            str(tag_name)
            for tag_name in self.relation_tag_names
            if float(confs.get(tag_name, 0.0)) >= 0.35
        )
        suffix = "none" if not active_relations else "+".join(active_relations)
        return f"{concept_tag}|{suffix}"

    def _cluster_profile(self, cluster: ConceptCluster) -> Dict[str, float]:
        profile = dict(cluster.semantic_profile)
        profile.update(cluster.relation_profile)
        return profile

    def _profile_matches_query(self, profile: Dict[str, float], planner_query: PlannerQuery) -> bool:
        return self._required_matches_profile(profile, planner_query)

    def _profile_intent_score(self, profile: Dict[str, float], planner_query: PlannerQuery) -> float:
        preferred_tags = self._query_preferred_tags(planner_query)
        penalty_tags = self._query_penalty_tags(planner_query)
        bonus = 0.0
        penalty = 0.0
        for tag_name, weight in preferred_tags.items():
            bonus += float(weight) * float(profile.get(str(tag_name), 0.0))
        for tag_name, weight in penalty_tags.items():
            penalty += float(weight) * float(profile.get(str(tag_name), 0.0))
        return float(bonus - penalty)

    def _cluster_base_label(self, cluster: ConceptCluster) -> str:
        active_relations = sorted(
            tag_name
            for tag_name, value in dict(cluster.relation_profile).items()
            if float(value) >= 0.35
        )
        suffix = "none" if not active_relations else "+".join(active_relations)
        return f"{cluster.concept_tag}|{suffix}"

    def _cluster_query_role(self, cluster: ConceptCluster, planner_query: PlannerQuery) -> str:
        profile = self._cluster_profile(cluster)
        query_tag = str(planner_query.target_predicate.tag_name)
        if float(profile.get(query_tag, 0.0)) >= float(self._query_required_tags(planner_query).get(query_tag, 0.0)):
            return f"{query_tag}_candidate"
        weighted_tags = []
        for tag_name, weight in self._query_preferred_tags(planner_query).items():
            weighted_tags.append((float(weight) * float(profile.get(str(tag_name), 0.0)), str(tag_name)))
        weighted_tags.sort(reverse=True)
        if weighted_tags and weighted_tags[0][0] > 0.0:
            return f"{weighted_tags[0][1]}_support"
        return f"{query_tag}_support"

    def build_concept_clusters(self) -> List[ConceptCluster]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for symbolic_node_id in sorted(self.symbolic_nodes.keys()):
            snapshot = self.symbolic_node_snapshot(symbolic_node_id)
            cluster_key = self._cluster_key(snapshot)
            if cluster_key is None:
                continue
            buckets.setdefault(cluster_key, []).append(snapshot)

        clusters: List[ConceptCluster] = []
        for cluster_key, members in sorted(buckets.items()):
            concept_tag = str(cluster_key.split("|", 1)[0])
            member_node_ids = [int(member["symbolic_node_id"]) for member in members]
            semantic_profile: Dict[str, float] = {}
            relation_profile: Dict[str, float] = {}
            pose_means = []
            pose_vars = []
            freshness_vals = []
            confidence_vals = []
            for member in members:
                confs = dict(member.get("semantic_tag_confidence", {}))
                for tag_name, value in confs.items():
                    if str(tag_name) in self.relation_tag_names:
                        relation_profile[str(tag_name)] = relation_profile.get(str(tag_name), 0.0) + float(value)
                    else:
                        semantic_profile[str(tag_name)] = semantic_profile.get(str(tag_name), 0.0) + float(value)
                if member.get("pose_mean") is not None:
                    pose_means.append(np.asarray(member["pose_mean"], dtype=np.float64))
                if member.get("pose_var") is not None:
                    pose_vars.append(np.asarray(member["pose_var"], dtype=np.float64))
                freshness_vals.append(float(member.get("freshness", 0.0)))
                confidence_vals.append(float(member.get("confidence", 0.0)))
            denom = float(max(1, len(members)))
            semantic_profile = {key: float(value / denom) for key, value in semantic_profile.items()}
            relation_profile = {key: float(value / denom) for key, value in relation_profile.items()}
            clusters.append(
                ConceptCluster(
                    cluster_id=str(cluster_key),
                    concept_tag=str(concept_tag),
                    member_node_ids=[int(node_id) for node_id in member_node_ids],
                    relation_profile=relation_profile,
                    semantic_profile=semantic_profile,
                    pose_mean=None if not pose_means else tuple(float(v) for v in np.mean(pose_means, axis=0)),
                    pose_var=None if not pose_vars else tuple(float(v) for v in np.mean(pose_vars, axis=0)),
                    freshness_mean=float(np.mean(freshness_vals)) if freshness_vals else 0.0,
                    confidence_mean=float(np.mean(confidence_vals)) if confidence_vals else 0.0,
                )
            )
        return clusters

    def concept_clusters_for_intent(self, planner_query, top_k: int = 5) -> List[Dict[str, Any]]:
        planner_query = self._ensure_query(planner_query=planner_query) if isinstance(planner_query, PlannerQuery) else self._ensure_query(predicate=planner_query)
        clusters = self.build_concept_clusters()
        scored = []
        for cluster in clusters:
            profile = self._cluster_profile(cluster)
            if not self._profile_matches_query(profile, planner_query):
                continue
            cluster_score = self._profile_intent_score(profile, planner_query)
            cluster_score += self._query_alignment_bonus(
                profile,
                planner_query,
                dominant_tag=str(cluster.concept_tag),
            )
            cluster_score += 0.10 * float(cluster.freshness_mean)
            cluster_score += 0.05 * float(cluster.confidence_mean)
            scored.append(
                {
                    "cluster_id": str(cluster.cluster_id),
                    "concept_tag": str(cluster.concept_tag),
                    "cluster_base_label": self._cluster_base_label(cluster),
                    "cluster_query_role": self._cluster_query_role(cluster, planner_query),
                    "member_node_ids": [int(node_id) for node_id in cluster.member_node_ids],
                    "cluster_score": float(cluster_score),
                    "semantic_profile": dict(cluster.semantic_profile),
                    "relation_profile": dict(cluster.relation_profile),
                    "freshness_mean": float(cluster.freshness_mean),
                    "confidence_mean": float(cluster.confidence_mean),
                }
            )
        scored.sort(key=lambda item: float(item["cluster_score"]), reverse=True)
        return scored[: max(1, int(top_k))]

    def concept_debug_summary(self) -> Dict[str, Any]:
        clusters = self.build_concept_clusters()
        by_concept: Dict[str, int] = {}
        by_base_label: Dict[str, int] = {}
        for cluster in clusters:
            by_concept[str(cluster.concept_tag)] = int(by_concept.get(str(cluster.concept_tag), 0)) + 1
            base_label = self._cluster_base_label(cluster)
            by_base_label[base_label] = int(by_base_label.get(base_label, 0)) + 1
        return {
            "num_clusters": int(len(clusters)),
            "num_symbolic_nodes": int(len(self.symbolic_nodes)),
            "clusters_by_concept": by_concept,
            "clusters_by_base_label": by_base_label,
            "relation_tag_names": sorted(self.relation_tag_names),
        }

    def _cluster_membership(self, clusters: List[ConceptCluster]) -> Dict[int, str]:
        membership: Dict[int, str] = {}
        for cluster in clusters:
            for symbolic_node_id in cluster.member_node_ids:
                membership[int(symbolic_node_id)] = str(cluster.cluster_id)
        return membership

    def _cluster_graph(self, clusters: List[ConceptCluster], membership: Dict[int, str]) -> Dict[str, Set[str]]:
        cluster_graph: Dict[str, Set[str]] = {str(cluster.cluster_id): set() for cluster in clusters}
        for src_graph_node, dsts in self.graph.edges.items():
            src_symbolic_node_id = self.graph_node_to_symbolic_node.get(int(src_graph_node))
            src_cluster_id = membership.get(int(src_symbolic_node_id)) if src_symbolic_node_id is not None else None
            if src_cluster_id is None:
                continue
            for dst_graph_node in dsts.keys():
                dst_symbolic_node_id = self.graph_node_to_symbolic_node.get(int(dst_graph_node))
                dst_cluster_id = membership.get(int(dst_symbolic_node_id)) if dst_symbolic_node_id is not None else None
                if dst_cluster_id is None or dst_cluster_id == src_cluster_id:
                    continue
                cluster_graph.setdefault(str(src_cluster_id), set()).add(str(dst_cluster_id))
        return cluster_graph

    def _shortest_cluster_path(
        self,
        cluster_graph: Dict[str, Set[str]],
        src_cluster_id: str,
        dst_cluster_id: str,
    ) -> Optional[List[str]]:
        if str(src_cluster_id) == str(dst_cluster_id):
            return [str(src_cluster_id)]
        frontier = [str(src_cluster_id)]
        parents = {str(src_cluster_id): None}
        while frontier:
            current = frontier.pop(0)
            for nxt in sorted(cluster_graph.get(str(current), set())):
                if nxt in parents:
                    continue
                parents[nxt] = current
                if nxt == str(dst_cluster_id):
                    path = [str(dst_cluster_id)]
                    while parents[path[-1]] is not None:
                        path.append(parents[path[-1]])
                    path.reverse()
                    return path
                frontier.append(nxt)
        return None

    def _node_path_utility(self, planner, node_path: List[int], planner_query: Optional[PlannerQuery] = None) -> float:
        if not node_path:
            return 0.0
        path_utility = 0.0
        for idx, node_id in enumerate(node_path):
            path_utility += float(
                self.node_intent_score(int(node_id), planner_query=planner_query)
                if planner_query is not None
                else planner.node_utility(self, int(node_id))
            )
            if idx > 0:
                prev_node_id = int(node_path[idx - 1])
                path_utility += float(planner.edge_bonus) * float(planner.edge_utility(self, prev_node_id, int(node_id)))
        path_utility -= float(planner.path_penalty) * max(0, len(node_path) - 1)
        return float(path_utility)

    def _concept_rollout(
        self,
        planner_query: PlannerQuery,
        current_node_id: int,
        planner,
        horizon: int = 4,
        top_k: int = 5,
    ):
        planner_query = self._ensure_query(planner_query=planner_query)
        current_symbolic_node = self.graph_node_to_symbolic_node.get(int(current_node_id))
        if current_symbolic_node is None:
            self.last_concept_query_debug = {
                "mode": "concept_plan_missing_current_symbolic_node",
                "clusters": [],
                "cluster_membership": {},
            }
            return []
        clusters = self.build_concept_clusters()
        membership = self._cluster_membership(clusters)
        current_cluster_id = membership.get(int(current_symbolic_node))
        if current_cluster_id is None:
            self.last_concept_query_debug = {
                "mode": "concept_plan_missing_current_cluster",
                "clusters": [],
                "cluster_membership": {},
            }
            return []
        concept_candidates = self.concept_clusters_for_intent(planner_query, top_k=max(2, int(top_k)))
        if not concept_candidates:
            self.last_concept_query_debug = {
                "mode": "concept_plan_empty",
                "clusters": [],
                "cluster_membership": {},
            }
            return []
        cluster_graph = self._cluster_graph(clusters, membership)
        cluster_membership = {}
        for graph_node_id, symbolic_node_id in self.graph_node_to_symbolic_node.items():
            cluster_id = membership.get(int(symbolic_node_id))
            if cluster_id is not None:
                cluster_membership[str(graph_node_id)] = str(cluster_id)
        plans = []
        selected_cluster_paths: Dict[str, List[str]] = {}
        for cluster in concept_candidates:
            target_cluster_id = str(cluster["cluster_id"])
            cluster_path = self._shortest_cluster_path(cluster_graph, str(current_cluster_id), target_cluster_id)
            if cluster_path is None:
                continue
            selected_cluster_paths[target_cluster_id] = list(cluster_path)
            next_cluster_id = str(cluster_path[1]) if len(cluster_path) > 1 else str(target_cluster_id)
            entry_candidates = []
            for symbolic_node_id, cluster_id in membership.items():
                if str(cluster_id) != str(next_cluster_id):
                    continue
                representative_graph_node = self._representative_graph_node(
                    int(symbolic_node_id),
                    current_node_id=current_node_id,
                    planner_query=planner_query,
                )
                if representative_graph_node is not None:
                    entry_candidates.append(int(representative_graph_node))
            ranked_entries = []
            for entry_graph_node_id in entry_candidates:
                node_path = self.graph.shortest_path(int(current_node_id), int(entry_graph_node_id))
                if node_path is None:
                    continue
                if next_cluster_id == target_cluster_id and not self.node_matches_query(int(entry_graph_node_id), planner_query):
                    continue
                grounded_score = float(
                    self.node_intent_score(int(entry_graph_node_id), planner_query=planner_query)
                    + 0.35 * float(cluster["cluster_score"])
                    - 0.05 * max(0, len(node_path) - 1)
                )
                ranked_entries.append((grounded_score, int(entry_graph_node_id), list(node_path)))
            if not ranked_entries:
                continue
            ranked_entries.sort(key=lambda item: item[0], reverse=True)
            grounded_score, selected_node_id, full_path = ranked_entries[0]
            truncated_path = list(full_path[: max(1, int(horizon) + 1)])
            next_node_id = int(truncated_path[1]) if len(truncated_path) > 1 else int(truncated_path[0])
            waypoint_xy = None
            pose_xy = self.get_mean_xy(int(next_node_id), "pose")
            if pose_xy is not None:
                rounded = np.rint(pose_xy).astype(np.int32)
                waypoint_xy = (int(rounded[0]), int(rounded[1]))
            plans.append(
                ManeuverPlan(
                    token_sequence=[str(self.graph.nodes[node_id].get("token_type", "phrase")) for node_id in truncated_path],
                    node_path=[int(node_id) for node_id in truncated_path],
                    utility=float(self._node_path_utility(planner, truncated_path, planner_query=planner_query) + 0.45 * float(cluster["cluster_score"])),
                    graph_path_length=int(max(0, len(full_path) - 1)),
                    target_node_id=int(selected_node_id),
                    waypoint_xy=waypoint_xy,
                    metadata={
                        "next_node_id": int(next_node_id),
                        "intent_type": str(getattr(planner_query.intent_type, "value", planner_query.intent_type)),
                        "query_tag_name": str(planner_query.target_predicate.tag_name),
                        "used_intent_query": True,
                        "candidate_node_ids": [int(row[1]) for row in ranked_entries[: max(1, int(top_k))]],
                        "candidate_base_utilities": {
                            str(node_id): float(self.graph.node_utility(int(node_id)))
                            for _, node_id, _ in ranked_entries[: max(1, int(top_k))]
                        },
                        "candidate_tag_confidences": {
                            str(node_id): float(
                                self._symbolic_node_for_graph_node(int(node_id)).semantic_tag_confidence.get(
                                    str(planner_query.target_predicate.tag_name),
                                    0.0,
                                )
                            )
                            for _, node_id, _ in ranked_entries[: max(1, int(top_k))]
                            if self._symbolic_node_for_graph_node(int(node_id)) is not None
                        },
                        "candidate_intent_scores": {
                            str(node_id): float(self.node_intent_score(int(node_id), planner_query=planner_query))
                            for _, node_id, _ in ranked_entries[: max(1, int(top_k))]
                        },
                        "candidate_concept_membership": dict(cluster_membership),
                        "concept_query_debug": {
                            "mode": "concept_plan",
                            "current_cluster_id": str(current_cluster_id),
                            "clusters": concept_candidates,
                            "cluster_membership": dict(cluster_membership),
                            "selected_cluster_paths": selected_cluster_paths,
                            "target_cluster_id": str(target_cluster_id),
                            "next_cluster_id": str(next_cluster_id),
                            "selected_node_ids": [int(row[1]) for row in ranked_entries[: max(1, int(top_k))]],
                        },
                        "selected_tag_confidence": float(
                            0.0
                            if self._symbolic_node_for_graph_node(int(selected_node_id)) is None
                            else self._symbolic_node_for_graph_node(int(selected_node_id)).semantic_tag_confidence.get(
                                str(planner_query.target_predicate.tag_name),
                                0.0,
                            )
                        ),
                        "selected_intent_score": float(self.node_intent_score(int(selected_node_id), planner_query=planner_query)),
                        "selected_plan_utility": float(grounded_score),
                        "selected_node_semantic_tag_confidence": {}
                        if self._symbolic_node_for_graph_node(int(selected_node_id)) is None
                        else dict(self._symbolic_node_for_graph_node(int(selected_node_id)).semantic_tag_confidence),
                        "plan_source": "concept_plan",
                    },
                )
            )
        if not plans:
            self.last_concept_query_debug = {
                "mode": "concept_plan_no_grounding",
                "clusters": concept_candidates,
                "cluster_membership": dict(cluster_membership),
            }
            return []
        plans.sort(key=lambda plan: float(plan.utility), reverse=True)
        self.last_concept_query_debug = dict(plans[0].metadata.get("concept_query_debug", {}) or {})
        return plans[: max(1, int(top_k))]

    def record_query(
        self,
        step_idx: int,
        current_node_id: Optional[int],
        planner_query: Optional[PlannerQuery],
        planner_debug: Optional[Dict[str, Any]],
        selected_node_id: Optional[int] = None,
        selected_source: str = "none",
    ):
        if planner_query is None and planner_debug is None:
            return
        planner_debug = dict(planner_debug or {})
        candidate_node_ids = [int(value) for value in planner_debug.get("candidate_node_ids", [])]
        candidate_query_matches = {}
        retrieval_precision = 0.0
        if planner_query is not None and candidate_node_ids:
            candidate_query_matches = {
                str(node_id): bool(self.node_matches_query(int(node_id), planner_query))
                for node_id in candidate_node_ids
            }
            retrieval_precision = float(
                sum(1.0 for matched in candidate_query_matches.values() if matched)
                / max(1, len(candidate_query_matches))
            )
        selected_query_satisfied = False
        if planner_query is not None and selected_node_id is not None:
            selected_query_satisfied = bool(self.node_matches_query(int(selected_node_id), planner_query))
        semantic_target_materialized = bool(
            selected_source not in {"", "none"}
            or selected_node_id is not None
            or planner_debug.get("goal_rejoin_target_materialized", False)
        )
        planner_debug["candidate_query_matches"] = dict(candidate_query_matches)
        planner_debug["retrieval_precision_at_k"] = float(retrieval_precision)
        planner_debug["selected_query_satisfied"] = int(selected_query_satisfied)
        planner_debug["semantic_target_materialized"] = int(semantic_target_materialized)
        candidate_scores = dict(planner_debug.get("candidate_intent_scores", {}))
        if not candidate_scores:
            candidate_scores = dict(planner_debug.get("candidate_base_utilities", {}))
        record = QueryDebugRecord(
            episode_id=-1 if self.active_episode is None else int(self.active_episode.episode_id),
            step_idx=int(step_idx),
            current_node_id=None if current_node_id is None else int(current_node_id),
            intent_type=None if planner_query is None else str(getattr(planner_query.intent_type, "value", planner_query.intent_type)),
            query_tag_name=None if planner_query is None else str(planner_query.target_predicate.tag_name),
            candidate_node_ids=candidate_node_ids,
            candidate_scores={str(key): float(value) for key, value in candidate_scores.items()},
            selected_node_id=None if selected_node_id is None else int(selected_node_id),
            selected_source=str(selected_source),
            metadata=planner_debug,
        )
        self.query_debug_records.append(record)

    def query_debug_summary(self) -> Dict[str, Any]:
        by_intent: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        query_nonempty = 0.0
        query_satisfied = 0.0
        target_materialized = 0.0
        retrieval_precision_vals = []
        for record in self.query_debug_records:
            intent_type = str(record.intent_type or "none")
            selected_source = str(record.selected_source or "none")
            by_intent[intent_type] = int(by_intent.get(intent_type, 0)) + 1
            by_source[selected_source] = int(by_source.get(selected_source, 0)) + 1
            if record.candidate_node_ids:
                query_nonempty += 1.0
            if int(record.metadata.get("selected_query_satisfied", 0)) > 0:
                query_satisfied += 1.0
            if int(record.metadata.get("semantic_target_materialized", 0)) > 0:
                target_materialized += 1.0
            retrieval_precision_vals.append(float(record.metadata.get("retrieval_precision_at_k", 0.0)))
        denom = float(max(1, len(self.query_debug_records)))
        return {
            "num_episode_records": int(len(self.episode_records)),
            "num_query_records": int(len(self.query_debug_records)),
            "num_symbolic_nodes": int(len(self.symbolic_nodes)),
            "queries_by_intent": by_intent,
            "queries_by_source": by_source,
            "query_nonempty_rate": float(query_nonempty / denom),
            "query_satisfaction_rate": float(query_satisfied / denom),
            "semantic_target_materialization_rate": float(target_materialized / denom),
            "retrieval_precision_at_k_mean": float(np.mean(retrieval_precision_vals)) if retrieval_precision_vals else 0.0,
        }

    def export_debug(self, out_dir: str, env_idx: int = 0):
        self.export(out_dir=out_dir, env_idx=env_idx)

    def export(self, out_dir: str, env_idx: int):
        self.graph.export(out_dir, env_idx=env_idx)
        env_dir = os.path.join(out_dir, f"env_{int(env_idx)}")
        os.makedirs(env_dir, exist_ok=True)

        symbolic_nodes = [
            self.symbolic_node_snapshot(symbolic_node_id)
            for symbolic_node_id in sorted(self.symbolic_nodes.keys())
        ]
        with open(os.path.join(env_dir, "symbolic_nodes.json"), "w") as file_obj:
            json.dump(symbolic_nodes, file_obj, indent=2)

        symbolic_transitions = []
        for src, dsts in self.symbolic_transitions.items():
            for dst in dsts.keys():
                symbolic_transitions.append(self.symbolic_transition_snapshot(int(src), int(dst)))
        with open(os.path.join(env_dir, "symbolic_transitions.json"), "w") as file_obj:
            json.dump(symbolic_transitions, file_obj, indent=2)

        relation_rows = []
        for relation_name, bucket in sorted(self.relation_records.items()):
            for symbolic_node_id, relation_record in sorted(bucket.items()):
                row = asdict(relation_record)
                row["relation_name"] = str(relation_name)
                row["symbolic_node_id"] = int(symbolic_node_id)
                relation_rows.append(row)
        with open(os.path.join(out_dir, "relation_records.json"), "w") as file_obj:
            json.dump(relation_rows, file_obj, indent=2)

        concept_rows = [asdict(record) for _, record in sorted(self.concept_records.items())]
        with open(os.path.join(out_dir, "concept_records.json"), "w") as file_obj:
            json.dump(concept_rows, file_obj, indent=2)

        with open(os.path.join(out_dir, "episode_records.json"), "w") as file_obj:
            json.dump([asdict(record) for record in self.episode_records], file_obj, indent=2)

        with open(os.path.join(out_dir, "query_debug.json"), "w") as file_obj:
            json.dump([asdict(record) for record in self.query_debug_records], file_obj, indent=2)

        with open(os.path.join(out_dir, "query_debug_summary.json"), "w") as file_obj:
            json.dump(self.query_debug_summary(), file_obj, indent=2)

        concept_clusters = []
        for cluster in self.build_concept_clusters():
            row = asdict(cluster)
            row["cluster_base_label"] = self._cluster_base_label(cluster)
            concept_clusters.append(row)
        with open(os.path.join(out_dir, "concept_clusters.json"), "w") as file_obj:
            json.dump(concept_clusters, file_obj, indent=2)

        with open(os.path.join(out_dir, "concept_debug_summary.json"), "w") as file_obj:
            json.dump(self.concept_debug_summary(), file_obj, indent=2)
