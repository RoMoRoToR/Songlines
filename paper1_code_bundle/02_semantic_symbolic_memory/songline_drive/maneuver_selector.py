from songline_drive.types import ManeuverPlan


class ManeuverSelector:
    def select(self, plan: ManeuverPlan, current_token: str = None):
        if not plan.node_path:
            return {
                "command_type": "go_to_waypoint",
                "waypoint_xy": plan.waypoint_xy,
            }
        if current_token in {"hazard_front", "gap_search"}:
            return {
                "command_type": "align_and_cross_gap",
                "waypoint_xy": plan.waypoint_xy,
                "target_node_id": plan.target_node_id,
                "graph_path_length": plan.graph_path_length,
            }
        return {
            "command_type": "follow_graph_path",
            "waypoint_xy": plan.waypoint_xy,
            "target_node_id": plan.target_node_id,
            "graph_path_length": plan.graph_path_length,
        }
