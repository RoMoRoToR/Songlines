from typing import List, Optional

import numpy as np

from songline_drive.types import ManeuverPlan


class GraphRolloutPlanner:
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        eta: float = 0.5,
        gamma: float = 1.0,
        delta: float = 0.5,
        zeta: float = 0.5,
        path_penalty: float = 0.1,
    ):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.path_penalty = path_penalty
        self.edge_bonus = 1.0

    def utility(self, node_stats) -> float:
        return (
            self.alpha * float(node_stats["progress"])
            + self.beta * float(node_stats["goal_alignment"])
            + self.eta * float(node_stats["reuse"])
            - self.gamma * float(node_stats["risk"])
            - self.delta * float(node_stats["comfort"])
            - self.zeta * float(node_stats["uncertainty"])
        )

    def _node_stats(self, graph, node_id: int):
        node = graph.nodes[node_id]
        return {
            "progress": graph._mean(node, "progress"),
            "goal_alignment": graph._mean(node, "goal_alignment"),
            "reuse": float(node.get("reuse_score", 0.0)),
            "risk": graph._mean(node, "risk"),
            "comfort": graph._mean(node, "comfort_cost"),
            "uncertainty": graph._mean(node, "uncertainty"),
        }

    def node_utility(self, graph, node_id: int) -> float:
        if hasattr(graph, "node_utility"):
            return float(graph.node_utility(node_id))
        return float(self.utility(self._node_stats(graph, node_id)))

    def edge_utility(self, graph, src: int, dst: int) -> float:
        if hasattr(graph, "edge_utility"):
            return float(graph.edge_utility(src, dst))
        return 0.0

    def rollout(
        self,
        graph,
        current_node_id: Optional[int],
        horizon: int = 4,
        top_k: int = 5,
        planner_query=None,
    ) -> List[ManeuverPlan]:
        if current_node_id is None:
            return []

        if planner_query is not None and hasattr(graph, "candidate_nodes_for_intent"):
            candidate_ids = graph.candidate_nodes_for_intent(
                planner_query.target_predicate,
                top_k=top_k,
            )
        else:
            candidate_ids = graph.candidate_nodes(top_k=top_k)
        query_tag_name = None if planner_query is None else str(planner_query.target_predicate.tag_name)
        candidate_base_utilities = {}
        candidate_tag_confidences = {}
        for node_id in candidate_ids:
            candidate_base_utilities[str(node_id)] = float(self.node_utility(graph, node_id))
            if query_tag_name is not None:
                candidate_tag_confidences[str(node_id)] = float(
                    graph.nodes[node_id].get("semantic_tag_confidence", {}).get(query_tag_name, 0.0)
                )
        plans: List[ManeuverPlan] = []
        for node_id in candidate_ids:
            path = graph.shortest_path(current_node_id, node_id)
            if path is None:
                continue
            truncated_path = path[: max(1, horizon + 1)]
            path_utility = 0.0
            token_sequence = []
            for idx, path_node_id in enumerate(truncated_path):
                node = graph.nodes[path_node_id]
                token_sequence.append(str(node.get("token_type", "phrase")))
                path_utility += self.node_utility(graph, path_node_id)
                if idx > 0:
                    prev_node_id = truncated_path[idx - 1]
                    path_utility += self.edge_bonus * self.edge_utility(graph, prev_node_id, path_node_id)
            path_utility -= self.path_penalty * max(0, len(truncated_path) - 1)

            waypoint_xy = None
            next_node_id = truncated_path[1] if len(truncated_path) > 1 else truncated_path[0]
            pose_xy = graph.get_mean_xy(next_node_id, "pose")
            if pose_xy is not None:
                rounded = np.rint(pose_xy).astype(np.int32)
                waypoint_xy = (int(rounded[0]), int(rounded[1]))

            plans.append(
                ManeuverPlan(
                    token_sequence=token_sequence,
                    node_path=[int(nid) for nid in truncated_path],
                    utility=float(path_utility),
                    graph_path_length=int(max(0, len(path) - 1)),
                    target_node_id=int(node_id),
                    waypoint_xy=waypoint_xy,
                    metadata={
                        "next_node_id": int(next_node_id),
                        "intent_type": None if planner_query is None else str(planner_query.intent_type.value),
                        "query_tag_name": query_tag_name,
                        "used_intent_query": bool(planner_query is not None),
                        "candidate_node_ids": [int(cid) for cid in candidate_ids],
                        "candidate_base_utilities": candidate_base_utilities,
                        "candidate_tag_confidences": candidate_tag_confidences,
                        "selected_tag_confidence": float(candidate_tag_confidences.get(str(node_id), 0.0)),
                        "selected_plan_utility": float(path_utility),
                    },
                )
            )

        plans.sort(key=lambda plan: plan.utility, reverse=True)
        return plans
