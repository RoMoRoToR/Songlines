import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from songline_drive.types import GraphEdge, GraphNode


class DynamicSonglineGraph:
    def __init__(self, min_goal_visits: int = 3, freshness_tau: float = 50.0):
        self.min_goal_visits = min_goal_visits
        self.freshness_tau = max(1.0, float(freshness_tau))
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
            "reuse_score": node.reuse_score,
            "uncertainty_sum": node.uncertainty_sum,
            "uncertainty_count": node.uncertainty_count,
            "last_seen_step": node.last_seen_step,
            "pose_sum": np.zeros(2, dtype=np.float64),
            "pose_count": node.pose_count,
            "goal_sum": np.zeros(2, dtype=np.float64),
            "goal_count": node.goal_count,
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
                self.edges[self.previous_phrase_id][phrase_id] = self.edges[self.previous_phrase_id].get(phrase_id, 0) + 1
                edge = self.edge_stats[self.previous_phrase_id].get(phrase_id)
                if edge is None:
                    self.edge_stats[self.previous_phrase_id][phrase_id] = GraphEdge(
                        src=self.previous_phrase_id,
                        dst=phrase_id,
                        weight=self.edges[self.previous_phrase_id][phrase_id],
                    )
                else:
                    edge.weight = self.edges[self.previous_phrase_id][phrase_id]
            self.previous_phrase_id = phrase_id
            self.current_phrase = []

        self.node_growth.append(len(self.nodes))
        edge_count = sum(len(dsts) for dsts in self.edges.values())
        self.edge_growth.append(edge_count)
        return new_phrase

    def observe(self, step_idx: int, pose_xy=None, goal_xy=None, reward=None):
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
        if int(node["visits"]) > 1:
            node["reuse_score"] = float(node["visits"] - 1) / float(node["visits"])

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

        progress = None
        goal_alignment = None
        if pose_arr is not None and goal_arr is not None:
            goal_distance = float(np.abs(pose_arr - goal_arr).sum())
            goal_alignment = -goal_distance
            if self.last_goal_distance is not None:
                progress = float(self.last_goal_distance - goal_distance)
            self.last_goal_distance = goal_distance

        if progress is not None:
            node["progress_sum"] += progress
            node["progress_count"] += 1

        if goal_alignment is not None:
            node["goal_alignment_sum"] += goal_alignment
            node["goal_alignment_count"] += 1

        if pose_arr is not None and self.previous_pose_xy is not None:
            step_distance = float(np.abs(pose_arr - self.previous_pose_xy).sum())
            node["speed_sum"] += step_distance
            node["speed_count"] += 1
        if pose_arr is not None:
            self.previous_pose_xy = pose_arr

    def _compute_freshness(self, now_step: int, last_seen_step: int) -> float:
        age = max(0.0, float(now_step - last_seen_step))
        return float(np.exp(-age / self.freshness_tau))

    def _mean(self, node: Dict[str, object], key: str) -> float:
        count = int(node.get(f"{key}_count", 0))
        if count <= 0:
            return 0.0
        return float(node[f"{key}_sum"]) / float(count)

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
                    "last_seen_step": int(node["last_seen_step"]),
                }
            )
        with open(os.path.join(env_dir, "phrases.json"), "w") as f:
            json.dump(phrases, f)

        edge_list = []
        for src, dsts in self.edges.items():
            for dst, weight in dsts.items():
                edge_list.append({"src": src, "dst": dst, "weight": int(weight)})
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
