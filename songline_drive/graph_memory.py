import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from songline_drive.types import GraphEdge, GraphNode, SemanticTargetPredicate


class DynamicSonglineGraph:
    def __init__(
        self,
        min_goal_visits: int = 3,
        freshness_tau: float = 50.0,
        graph_update_mode: str = "static",
        alpha_fast: float = 0.15,
        alpha_slow: float = 0.02,
        confidence_kappa: float = 5.0,
    ):
        self.min_goal_visits = min_goal_visits
        self.freshness_tau = max(1.0, float(freshness_tau))
        self.graph_update_mode = str(graph_update_mode)
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.confidence_kappa = max(1.0, float(confidence_kappa))
        self.dictionary: Dict[Tuple[int, ...], int] = {}
        self.nodes: Dict[int, Dict[str, object]] = {}
        self.edges: Dict[int, Dict[int, int]] = {}
        self.edge_stats: Dict[int, Dict[int, GraphEdge]] = {}
        self.current_phrase: List[int] = []
        self.previous_phrase_id: Optional[int] = None
        self.current_phrase_id: Optional[int] = None

        self.token_sequence: List[int] = []
        self.completed_phrases: List[int] = []
        self.node_growth: List[int] = []
        self.edge_growth: List[int] = []
        self.interventions = 0
        self.intervention_attempts = 0
        self.plan_hits = 0
        self.plan_total = 0
        self.last_goal_distance: Optional[float] = None
        self.previous_pose_xy: Optional[np.ndarray] = None

    def _new_node(self, phrase: Tuple[int, ...]) -> int:
        node = GraphNode(
            node_id=len(self.dictionary),
            token_type="phrase",
            context_signature=phrase,
            phrase=phrase,
        )
        node_id = node.node_id
        self.dictionary[phrase] = node_id
        self.nodes[node_id] = {
            "phrase": phrase,
            "token_type": node.token_type,
            "context_signature": node.context_signature,
            "visits": node.visits,
            "success_count": node.success_count,
            "progress_sum": node.progress_sum,
            "progress_count": node.progress_count,
            "risk_sum": node.risk_sum,
            "risk_count": node.risk_count,
            "comfort_cost_sum": node.comfort_cost_sum,
            "comfort_cost_count": node.comfort_cost_count,
            "goal_alignment_sum": node.goal_alignment_sum,
            "goal_alignment_count": node.goal_alignment_count,
            "reward_sum": node.reward_sum,
            "reward_count": node.reward_count,
            "speed_sum": node.speed_sum,
            "speed_count": node.speed_count,
            "freshness": node.freshness,
            "confidence": node.confidence,
            "progress_slow": node.progress_slow,
            "risk_slow": node.risk_slow,
            "success_slow": node.success_slow,
            "comfort_slow": node.comfort_slow,
            "goal_alignment_slow": node.goal_alignment_slow,
            "progress_fast": node.progress_fast,
            "risk_fast": node.risk_fast,
            "success_fast": node.success_fast,
            "comfort_fast": node.comfort_fast,
            "goal_alignment_fast": node.goal_alignment_fast,
            "progress_var": node.progress_var,
            "risk_var": node.risk_var,
            "success_var": node.success_var,
            "reuse_score": node.reuse_score,
            "utility_cached": node.utility_cached,
            "uncertainty_sum": node.uncertainty_sum,
            "uncertainty_count": node.uncertainty_count,
            "last_seen_step": node.last_seen_step,
            "pose_sum": np.zeros(2, dtype=np.float64),
            "pose_count": node.pose_count,
            "goal_sum": np.zeros(2, dtype=np.float64),
            "goal_count": node.goal_count,
            "phase_histogram": {},
            "semantic_tag_counts": {},
            "semantic_tag_confidence": {},
        }
        self.edges[node_id] = {}
        self.edge_stats[node_id] = {}
        return node_id

    def update_token(self, token: int, step_idx: int) -> bool:
        self.token_sequence.append(int(token))
        self.current_phrase.append(int(token))
        phrase_key = tuple(self.current_phrase)

        new_phrase = False
        if phrase_key not in self.dictionary:
            new_phrase = True
            phrase_id = self._new_node(phrase_key)
            self.completed_phrases.append(phrase_id)
            self.current_phrase_id = phrase_id
            if self.previous_phrase_id is not None:
                self.edges[self.previous_phrase_id].setdefault(phrase_id, 0)
                edge = self.edge_stats[self.previous_phrase_id].get(phrase_id)
                if edge is None:
                    self.edge_stats[self.previous_phrase_id][phrase_id] = GraphEdge(
                        src=self.previous_phrase_id,
                        dst=phrase_id,
                        weight=self.edges[self.previous_phrase_id][phrase_id],
                    )
            self.previous_phrase_id = phrase_id
            self.current_phrase = []

        self.node_growth.append(len(self.nodes))
        edge_count = sum(len(dsts) for dsts in self.edges.values())
        self.edge_growth.append(edge_count)
        return new_phrase

    def observe(
        self,
        step_idx: int,
        pose_xy=None,
        goal_xy=None,
        reward=None,
        progress=None,
        risk=None,
        success=None,
        comfort_cost=None,
        goal_alignment=None,
        phase_label: Optional[str] = None,
    ):
        if self.current_phrase_id is None:
            return

        node = self.nodes[self.current_phrase_id]
        node["visits"] += 1
        prev_last_seen = int(node["last_seen_step"])
        if prev_last_seen < 0:
            node["freshness"] = 1.0
        else:
            node["freshness"] = self._compute_freshness(step_idx, prev_last_seen)
        node["last_seen_step"] = int(step_idx)
        node["confidence"] = self._compute_confidence(int(node["visits"]))
        if int(node["visits"]) > 1:
            node["reuse_score"] = float(node["visits"] - 1) / float(node["visits"])
        if phase_label:
            phase_histogram = node.get("phase_histogram", {})
            phase_histogram[str(phase_label)] = int(phase_histogram.get(str(phase_label), 0)) + 1
            node["phase_histogram"] = phase_histogram

        pose_arr = None if pose_xy is None else np.asarray(pose_xy, dtype=np.float64)
        goal_arr = None if goal_xy is None else np.asarray(goal_xy, dtype=np.float64)

        if pose_arr is not None:
            node["pose_sum"] += pose_arr
            node["pose_count"] += 1

        if goal_arr is not None:
            node["goal_sum"] += goal_arr
            node["goal_count"] += 1

        if reward is not None:
            reward_val = float(reward)
            node["reward_sum"] += reward_val
            node["reward_count"] += 1
            if reward_val > 0:
                node["success_count"] += 1
            if success is None:
                success = 1.0 if reward_val > 0 else 0.0

        derived_progress = None
        derived_goal_alignment = None
        if pose_arr is not None and goal_arr is not None:
            goal_distance = float(np.abs(pose_arr - goal_arr).sum())
            derived_goal_alignment = -goal_distance
            if self.last_goal_distance is not None:
                derived_progress = float(self.last_goal_distance - goal_distance)
            self.last_goal_distance = goal_distance

        progress_value = derived_progress if progress is None else float(progress)
        goal_alignment_value = derived_goal_alignment if goal_alignment is None else float(goal_alignment)
        risk_value = None if risk is None else float(risk)
        success_value = None if success is None else float(success)
        comfort_value = None if comfort_cost is None else float(comfort_cost)

        if progress_value is not None:
            node["progress_sum"] += progress_value
            node["progress_count"] += 1

        if goal_alignment_value is not None:
            node["goal_alignment_sum"] += goal_alignment_value
            node["goal_alignment_count"] += 1

        if risk_value is not None:
            node["risk_sum"] += risk_value
            node["risk_count"] += 1

        if comfort_value is not None:
            node["comfort_cost_sum"] += comfort_value
            node["comfort_cost_count"] += 1

        if success_value is not None:
            node["uncertainty_sum"] += 0.0
            node["uncertainty_count"] += 1

        if pose_arr is not None and self.previous_pose_xy is not None:
            step_distance = float(np.abs(pose_arr - self.previous_pose_xy).sum())
            node["speed_sum"] += step_distance
            node["speed_count"] += 1
        if pose_arr is not None:
            self.previous_pose_xy = pose_arr

        if self.graph_update_mode == "adaptive":
            self._update_adaptive_node(
                node,
                progress_value=progress_value,
                risk_value=risk_value,
                success_value=success_value,
                comfort_value=comfort_value,
                goal_alignment_value=goal_alignment_value,
            )
        node["utility_cached"] = self.node_utility(self.current_phrase_id)

    def observe_semantics(self, node_id: Optional[int], semantic_tags: Dict[str, float]):
        if node_id is None:
            return
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        counts = node.setdefault("semantic_tag_counts", {})
        confs = node.setdefault("semantic_tag_confidence", {})

        for tag_name, value in semantic_tags.items():
            tag_value = float(value)
            if tag_value <= 0.0:
                continue
            counts[tag_name] = int(counts.get(tag_name, 0)) + 1
            prev = float(confs.get(tag_name, 0.0))
            new_count = int(counts[tag_name])
            confs[tag_name] = ((prev * float(new_count - 1)) + tag_value) / float(new_count)

    def _compute_freshness(self, now_step: int, last_seen_step: int) -> float:
        age = max(0.0, float(now_step - last_seen_step))
        return float(np.exp(-age / self.freshness_tau))

    def _compute_confidence(self, visits: int) -> float:
        return float(1.0 - np.exp(-max(0, int(visits)) / self.confidence_kappa))

    def _ema_update(self, old: float, value: float, alpha: float) -> float:
        return ((1.0 - alpha) * float(old)) + (alpha * float(value))

    def _ema_mean_var_update(self, mean: float, var: float, value: float, alpha: float) -> Tuple[float, float]:
        new_mean = self._ema_update(mean, value, alpha)
        new_var = ((1.0 - alpha) * float(var)) + (alpha * float(value - new_mean) ** 2)
        return new_mean, new_var

    def _update_adaptive_node(
        self,
        node: Dict[str, object],
        progress_value: Optional[float],
        risk_value: Optional[float],
        success_value: Optional[float],
        comfort_value: Optional[float],
        goal_alignment_value: Optional[float],
    ):
        if progress_value is not None:
            node["progress_fast"], node["progress_var"] = self._ema_mean_var_update(
                float(node.get("progress_fast", 0.0)),
                float(node.get("progress_var", 0.0)),
                progress_value,
                self.alpha_fast,
            )
            node["progress_slow"] = self._ema_update(float(node.get("progress_slow", 0.0)), progress_value, self.alpha_slow)
        if risk_value is not None:
            node["risk_fast"], node["risk_var"] = self._ema_mean_var_update(
                float(node.get("risk_fast", 0.0)),
                float(node.get("risk_var", 0.0)),
                risk_value,
                self.alpha_fast,
            )
            node["risk_slow"] = self._ema_update(float(node.get("risk_slow", 0.0)), risk_value, self.alpha_slow)
        if success_value is not None:
            node["success_fast"], node["success_var"] = self._ema_mean_var_update(
                float(node.get("success_fast", 0.0)),
                float(node.get("success_var", 0.0)),
                success_value,
                self.alpha_fast,
            )
            node["success_slow"] = self._ema_update(float(node.get("success_slow", 0.0)), success_value, self.alpha_slow)
        if comfort_value is not None:
            node["comfort_fast"] = self._ema_update(float(node.get("comfort_fast", 0.0)), comfort_value, self.alpha_fast)
            node["comfort_slow"] = self._ema_update(float(node.get("comfort_slow", 0.0)), comfort_value, self.alpha_slow)
        if goal_alignment_value is not None:
            node["goal_alignment_fast"] = self._ema_update(
                float(node.get("goal_alignment_fast", 0.0)),
                goal_alignment_value,
                self.alpha_fast,
            )
            node["goal_alignment_slow"] = self._ema_update(
                float(node.get("goal_alignment_slow", 0.0)),
                goal_alignment_value,
                self.alpha_slow,
            )

    def _observe_edge(
        self,
        src: int,
        dst: int,
        step_idx: int,
        success_value: Optional[float] = None,
        risk_value: Optional[float] = None,
        cost_value: Optional[float] = None,
    ):
        edge = self.edge_stats[src][dst]
        edge.weight = int(edge.weight) + 1
        prev_last_seen = int(edge.last_seen_step)
        if prev_last_seen < 0:
            edge.freshness = 1.0
        else:
            edge.freshness = self._compute_freshness(step_idx, prev_last_seen)
        edge.last_seen_step = int(step_idx)
        edge.confidence = self._compute_confidence(edge.weight)
        if success_value is not None:
            edge.transition_success_fast = self._ema_update(edge.transition_success_fast, success_value, self.alpha_fast)
            edge.transition_success_slow = self._ema_update(edge.transition_success_slow, success_value, self.alpha_slow)
            edge.success_weight = edge.transition_success_slow
        if risk_value is not None:
            edge.transition_risk_fast = self._ema_update(edge.transition_risk_fast, risk_value, self.alpha_fast)
            edge.transition_risk_slow = self._ema_update(edge.transition_risk_slow, risk_value, self.alpha_slow)
            edge.risk_weight = edge.transition_risk_slow
        if cost_value is not None:
            edge.transition_cost_fast = self._ema_update(edge.transition_cost_fast, cost_value, self.alpha_fast)
            edge.transition_cost_slow = self._ema_update(edge.transition_cost_slow, cost_value, self.alpha_slow)

    def observe_transition(
        self,
        src: Optional[int],
        dst: Optional[int],
        step_idx: int,
        transition_success: Optional[float] = None,
        transition_risk: Optional[float] = None,
        transition_cost: Optional[float] = None,
    ):
        if src is None or dst is None or src == dst:
            return
        if src not in self.edges:
            self.edges[src] = {}
        if src not in self.edge_stats:
            self.edge_stats[src] = {}
        self.edges[src].setdefault(dst, 0)
        edge = self.edge_stats[src].get(dst)
        if edge is None:
            edge = GraphEdge(src=src, dst=dst, weight=0)
            self.edge_stats[src][dst] = edge
        self._observe_edge(
            src,
            dst,
            step_idx=step_idx,
            success_value=transition_success,
            risk_value=transition_risk,
            cost_value=transition_cost,
        )
        self.edges[src][dst] = int(self.edge_stats[src][dst].weight)

    def _mean(self, node: Dict[str, object], key: str) -> float:
        count = int(node.get(f"{key}_count", 0))
        if count <= 0:
            return 0.0
        return float(node[f"{key}_sum"]) / float(count)

    def _blended_stat(self, node: Dict[str, object], key: str) -> float:
        fast = float(node.get(f"{key}_fast", 0.0))
        slow = float(node.get(f"{key}_slow", 0.0))
        freshness = float(node.get("freshness", 0.0))
        beta = min(1.0, max(0.2, freshness))
        return (beta * fast) + ((1.0 - beta) * slow)

    def _node_uncertainty(self, node: Dict[str, object]) -> float:
        return float(
            np.sqrt(
                max(0.0, float(node.get("progress_var", 0.0)))
                + max(0.0, float(node.get("risk_var", 0.0)))
                + max(0.0, float(node.get("success_var", 0.0)))
            )
        )

    def current_node_id(self) -> Optional[int]:
        return self.current_phrase_id

    def candidate_nodes(self, top_k: int = 5) -> List[int]:
        scored = []
        for node_id, node in self.nodes.items():
            if int(node["visits"]) < self.min_goal_visits:
                continue
            if int(node["goal_count"]) <= 0 or int(node["pose_count"]) <= 0:
                continue
            scored.append((self.node_utility(node_id), node_id))
        scored.sort(reverse=True)
        return [node_id for _, node_id in scored[:top_k]]

    def node_matches_intent(self, node_id: int, predicate: SemanticTargetPredicate) -> bool:
        node = self.nodes[node_id]
        confs = node.get("semantic_tag_confidence", {})
        tag_value = float(confs.get(predicate.tag_name, 0.0))
        if tag_value < float(predicate.min_confidence):
            return False
        for tag_name, min_value in predicate.required_tag_thresholds.items():
            if float(confs.get(tag_name, 0.0)) < float(min_value):
                return False
        return True

    def node_intent_score(self, node_id: int, predicate: SemanticTargetPredicate) -> float:
        node = self.nodes[node_id]
        confs = node.get("semantic_tag_confidence", {})
        base_utility = self.node_utility(node_id)
        score_weights = dict(predicate.score_weights or {predicate.tag_name: 0.5})
        bonus = 0.0
        for tag_name, weight in score_weights.items():
            bonus += float(weight) * float(confs.get(tag_name, 0.0))
        penalty = 0.0
        for tag_name, weight in predicate.penalty_weights.items():
            penalty += float(weight) * float(confs.get(tag_name, 0.0))
        return float(base_utility + bonus - penalty)

    def candidate_nodes_for_intent(self, predicate: SemanticTargetPredicate, top_k: int = 5) -> List[int]:
        scored = []
        min_visits = int(predicate.metadata.get("min_visits_override", self.min_goal_visits))
        for node_id, node in self.nodes.items():
            if int(node["visits"]) < min_visits:
                continue
            if not self.node_matches_intent(node_id, predicate):
                continue
            score = self.node_intent_score(node_id, predicate)
            scored.append((score, node_id))

        scored.sort(reverse=True)
        return [node_id for _, node_id in scored[:top_k]]

    def shortest_path(self, src: int, dst: int) -> Optional[List[int]]:
        if src == dst:
            return [src]

        parents = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nxt in self.edges.get(cur, {}):
                if nxt in parents:
                    continue
                parents[nxt] = cur
                if nxt == dst:
                    path = [dst]
                    while parents[path[-1]] is not None:
                        path.append(parents[path[-1]])
                    path.reverse()
                    return path
                q.append(nxt)
        return None

    def node_utility(self, node_id: int) -> float:
        node = self.nodes[node_id]
        if self.graph_update_mode == "adaptive":
            progress = self._blended_stat(node, "progress")
            goal_alignment = self._blended_stat(node, "goal_alignment")
            success = self._blended_stat(node, "success")
            risk = self._blended_stat(node, "risk")
            comfort = self._blended_stat(node, "comfort")
            uncertainty = self._node_uncertainty(node)
            reuse = float(node.get("reuse_score", 0.0))
            freshness = float(node.get("freshness", 0.0))
            confidence = float(node.get("confidence", 0.0))
            return (
                1.5 * progress
                + 1.0 * goal_alignment
                + 1.25 * success
                + 0.35 * reuse
                + 0.25 * freshness
                + 0.20 * confidence
                - 1.25 * risk
                - 0.25 * comfort
                - 0.35 * uncertainty
            )
        progress = self._mean(node, "progress")
        goal_alignment = self._mean(node, "goal_alignment")
        reward = self._mean(node, "reward")
        risk = self._mean(node, "risk")
        comfort = self._mean(node, "comfort_cost")
        uncertainty = self._mean(node, "uncertainty")
        reuse = float(node.get("reuse_score", 0.0))
        freshness = float(node.get("freshness", 0.0))
        return (
            1.0 * progress
            + 1.0 * reward
            + 0.25 * goal_alignment
            + 0.35 * reuse
            + 0.2 * freshness
            - 1.0 * risk
            - 0.25 * comfort
            - 0.1 * uncertainty
        )

    def edge_utility(self, src: int, dst: int) -> float:
        edge = self.edge_stats.get(src, {}).get(dst)
        if edge is None:
            return 0.0
        if self.graph_update_mode != "adaptive":
            return 0.0
        beta = min(1.0, max(0.2, float(edge.freshness)))
        success = (beta * float(edge.transition_success_fast)) + ((1.0 - beta) * float(edge.transition_success_slow))
        risk = (beta * float(edge.transition_risk_fast)) + ((1.0 - beta) * float(edge.transition_risk_slow))
        cost = (beta * float(edge.transition_cost_fast)) + ((1.0 - beta) * float(edge.transition_cost_slow))
        return (
            1.0 * success
            - 1.0 * risk
            - 0.5 * cost
            + 0.25 * float(edge.freshness)
            + 0.20 * float(edge.confidence)
        )

    def get_mean_xy(self, node_id: int, key_prefix: str) -> Optional[np.ndarray]:
        node = self.nodes[node_id]
        count = int(node.get(f"{key_prefix}_count", 0))
        if count <= 0:
            return None
        return node[f"{key_prefix}_sum"] / float(count)

    def suggest_subgoal(self, top_k: int = 5):
        self.intervention_attempts += 1
        current = self.current_phrase_id
        if current is None:
            return None

        candidates = self.candidate_nodes(top_k=top_k)
        if not candidates:
            return None

        selected = None
        selected_path = None
        selected_utility = None
        for node_id in candidates:
            path = self.shortest_path(current, node_id)
            if path is None:
                continue
            utility = self.node_utility(node_id)
            if selected is None or utility > selected_utility:
                selected = node_id
                selected_path = path
                selected_utility = utility

        if selected is None or selected_path is None:
            return None

        goal_xy = self.get_mean_xy(selected, "goal")
        if goal_xy is None:
            return None

        self.interventions += 1
        return {
            "goal_xy": goal_xy,
            "node_id": int(selected),
            "path_len": int(max(0, len(selected_path) - 1)),
            "mean_reward": float(self._mean(self.nodes[selected], "reward")),
            "utility": float(selected_utility),
        }

    def record_plan_outcome(self, improved: bool):
        self.plan_total += 1
        if improved:
            self.plan_hits += 1

    def export(self, out_dir: str, env_idx: int):
        env_dir = os.path.join(out_dir, f"env_{env_idx}")
        os.makedirs(env_dir, exist_ok=True)

        with open(os.path.join(env_dir, "token_sequence.json"), "w") as f:
            json.dump(self.token_sequence, f)

        phrases = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            phrases.append(
                {
                    "node_id": node_id,
                    "phrase": list(node["phrase"]),
                    "length": len(node["phrase"]),
                    "visits": int(node["visits"]),
                    "mean_reward": self._mean(node, "reward") if int(node["reward_count"]) > 0 else None,
                    "mean_progress": self._mean(node, "progress") if int(node["progress_count"]) > 0 else None,
                    "mean_goal_alignment": self._mean(node, "goal_alignment") if int(node["goal_alignment_count"]) > 0 else None,
                    "reuse_score": float(node["reuse_score"]),
                    "freshness": float(node["freshness"]),
                    "confidence": float(node.get("confidence", 0.0)),
                    "utility_cached": float(node.get("utility_cached", 0.0)),
                    "progress_fast": float(node.get("progress_fast", 0.0)),
                    "progress_slow": float(node.get("progress_slow", 0.0)),
                    "risk_fast": float(node.get("risk_fast", 0.0)),
                    "risk_slow": float(node.get("risk_slow", 0.0)),
                    "semantic_tag_counts": node.get("semantic_tag_counts", {}),
                    "semantic_tag_confidence": node.get("semantic_tag_confidence", {}),
                    "last_seen_step": int(node["last_seen_step"]),
                }
            )
        with open(os.path.join(env_dir, "phrases.json"), "w") as f:
            json.dump(phrases, f)

        edge_list = []
        for src, dsts in self.edges.items():
            for dst, weight in dsts.items():
                edge = self.edge_stats.get(src, {}).get(dst)
                edge_list.append(
                    {
                        "src": src,
                        "dst": dst,
                        "weight": int(weight),
                        "freshness": 0.0 if edge is None else float(edge.freshness),
                        "confidence": 0.0 if edge is None else float(edge.confidence),
                        "transition_success_fast": 0.0 if edge is None else float(edge.transition_success_fast),
                        "transition_success_slow": 0.0 if edge is None else float(edge.transition_success_slow),
                        "transition_risk_fast": 0.0 if edge is None else float(edge.transition_risk_fast),
                        "transition_risk_slow": 0.0 if edge is None else float(edge.transition_risk_slow),
                        "transition_cost_fast": 0.0 if edge is None else float(edge.transition_cost_fast),
                        "transition_cost_slow": 0.0 if edge is None else float(edge.transition_cost_slow),
                    }
                )
        with open(os.path.join(env_dir, "graph_edges.json"), "w") as f:
            json.dump(edge_list, f)

        metrics = {
            "num_tokens": len(self.token_sequence),
            "num_nodes": len(self.nodes),
            "num_edges": len(edge_list),
            "intervention_rate": 0.0 if self.intervention_attempts == 0 else self.interventions / self.intervention_attempts,
            "plan_hit_rate": 0.0 if self.plan_total == 0 else self.plan_hits / self.plan_total,
        }
        with open(os.path.join(env_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
