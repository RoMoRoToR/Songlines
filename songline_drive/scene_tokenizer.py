import hashlib

from songline_drive.types import SceneState, SceneToken


class SceneTokenizer:
    def __init__(self, mode: str = "semantic"):
        self.mode = mode
        self.previous_topology = None
        self.previous_route_token = None
        self.previous_remaining_distance = None

    def reset(self):
        self.previous_topology = None
        self.previous_route_token = None
        self.previous_remaining_distance = None

    def tokenize(self, scene: SceneState) -> SceneToken:
        if self.mode == "semantic":
            return self._semantic_token(scene)
        if self.mode == "patch_hash":
            return self._patch_hash_token(scene)
        raise ValueError(f"Unknown scene tokenizer mode: {self.mode}")

    def _semantic_token(self, scene: SceneState) -> SceneToken:
        distance_progress = self._distance_progress(scene)
        goal_context = self._goal_context(scene)
        topology_context = self._topology_context(scene)
        hazard_context = self._hazard_context(scene)
        transition_context = self._transition_context(scene, topology_context, hazard_context)

        route_token = self._route_semantic_token(
            scene,
            goal_context=goal_context,
            topology_context=topology_context,
            hazard_context=hazard_context,
            transition_context=transition_context,
            distance_progress=distance_progress,
        )
        semantic_tags = self._semantic_tags(scene)
        self.previous_topology = topology_context
        self.previous_route_token = route_token
        self.previous_remaining_distance = float(scene.route_context.remaining_distance)
        return SceneToken(
            token_type=route_token,
            confidence=0.8,
            attributes={
                "goal_context": goal_context,
                "topology_context": topology_context,
                "hazard_context": hazard_context,
                "transition_context": transition_context,
                "remaining_distance": scene.route_context.remaining_distance,
                "distance_progress": distance_progress,
                "free_space_score": scene.free_space_score,
            },
            semantic_tags=semantic_tags,
        )

    def _semantic_tags(self, scene: SceneState):
        rf = scene.risk_features
        return {
            "safe_exit": float(rf.get("place_is_safe_zone", 0.0)),
            "hazard_recovery_route": float(rf.get("place_is_hazard_recovery_route", 0.0)),
            "goal_region": float(rf.get("place_is_goal_region", 0.0)),
            "hazard_edge": float(rf.get("place_is_hazard_edge", 0.0)),
            "room_center": float(rf.get("place_is_room_center", 0.0)),
            "corridor": float(rf.get("place_is_corridor", 0.0)),
        }

    def _distance_progress(self, scene: SceneState) -> float:
        current_distance = float(scene.route_context.remaining_distance)
        if self.previous_remaining_distance is None:
            return 0.0
        return float(self.previous_remaining_distance - current_distance)

    def _goal_context(self, scene: SceneState) -> str:
        goal_visible = bool(scene.route_context.goal_visible)
        goal_dist = float(scene.route_context.remaining_distance)
        if goal_visible and goal_dist <= 2.0:
            return "goal_visible_near"
        if goal_visible:
            return "goal_visible_far"
        return "goal_hidden"

    def _topology_context(self, scene: SceneState) -> str:
        doorway_like = float(scene.risk_features.get("doorway_like", 0.0)) > 0.0
        corridor_like = float(scene.risk_features.get("corridor_like", 0.0)) > 0.0
        open_space_like = float(scene.risk_features.get("open_space_like", 0.0)) > 0.0
        free_neighbor_count = int(scene.risk_features.get("free_neighbor_count", 0.0))
        if doorway_like:
            return "doorway"
        if corridor_like:
            return "corridor"
        if open_space_like and free_neighbor_count >= 3:
            return "room_center"
        if scene.free_space_score > 0.55:
            return "open_space"
        return "wall_follow"

    def _hazard_context(self, scene: SceneState) -> str:
        hazard_front = float(scene.risk_features.get("hazard_front", 0.0)) > 0.0
        hazard_near = float(scene.risk_features.get("hazard_near", 0.0)) > 0.0
        front_safe = float(scene.risk_features.get("front_safe", 0.0)) > 0.0
        lateral_hazard = float(scene.risk_features.get("lateral_hazard", 0.0)) > 0.0
        hazard_asymmetric_gap_channel = float(
            scene.risk_features.get("hazard_asymmetric_gap_channel", 0.0)
        ) > 0.0
        narrow_safe_channel = float(scene.risk_features.get("narrow_safe_channel", 0.0)) > 0.0
        goal_heading_alignment = float(scene.risk_features.get("goal_heading_alignment", 0.0))
        if (
            front_safe
            and goal_heading_alignment > 0.4
            and (hazard_front or hazard_near or lateral_hazard)
            and (narrow_safe_channel or hazard_asymmetric_gap_channel)
        ):
            return "gap_aligned"
        if hazard_front and goal_heading_alignment > 0.2:
            return "hazard_front"
        if hazard_front:
            return "gap_search"
        if front_safe and lateral_hazard:
            return "gap_search"
        if hazard_near:
            return "hazard_near"
        return "safe"

    def _transition_context(self, scene: SceneState, topology_context: str, hazard_context: str) -> str:
        goal_heading_alignment = float(scene.risk_features.get("goal_heading_alignment", 0.0))
        front_safe = float(scene.risk_features.get("front_safe", 0.0)) > 0.0
        hazard_asymmetric_gap_channel = float(
            scene.risk_features.get("hazard_asymmetric_gap_channel", 0.0)
        ) > 0.0
        narrow_safe_channel = float(scene.risk_features.get("narrow_safe_channel", 0.0)) > 0.0
        if hazard_context == "hazard_front":
            return "search_gap"
        if hazard_context == "gap_search" and front_safe and goal_heading_alignment > 0.55:
            return "align_gap"
        if hazard_context == "gap_aligned" and (narrow_safe_channel or hazard_asymmetric_gap_channel):
            return "cross"
        if self.previous_topology is not None and topology_context != self.previous_topology:
            return "cross"
        if topology_context == "doorway":
            return "transition"
        return "stay"

    def _route_semantic_token(
        self,
        scene: SceneState,
        goal_context: str,
        topology_context: str,
        hazard_context: str,
        transition_context: str,
        distance_progress: float,
    ) -> str:
        goal_dist = float(scene.route_context.remaining_distance)
        goal_alignment = float(scene.risk_features.get("goal_heading_alignment", 0.0))
        front_safe = float(scene.risk_features.get("front_safe", 0.0)) > 0.0
        hazard_front = float(scene.risk_features.get("hazard_front", 0.0)) > 0.0
        hazard_near = float(scene.risk_features.get("hazard_near", 0.0)) > 0.0
        lateral_hazard = float(scene.risk_features.get("lateral_hazard", 0.0)) > 0.0
        hazard_asymmetric_gap_channel = float(
            scene.risk_features.get("hazard_asymmetric_gap_channel", 0.0)
        ) > 0.0
        narrow_safe_channel = float(scene.risk_features.get("narrow_safe_channel", 0.0)) > 0.0
        hazard_channel = narrow_safe_channel or hazard_asymmetric_gap_channel or lateral_hazard
        crossing_progress = (
            self.previous_route_token in {"gap_aligned", "safe_crossing"}
            and distance_progress > 0.0
            and hazard_near
            and (front_safe or hazard_channel)
        )
        post_crossing_exit = (
            self.previous_route_token in {"safe_crossing", "post_hazard"}
            and distance_progress > 0.0
            and not hazard_front
            and front_safe
            and not lateral_hazard
        )

        if post_crossing_exit:
            return "post_hazard"
        if crossing_progress:
            return "safe_crossing"
        if hazard_context == "gap_aligned":
            return "gap_aligned"
        if hazard_context == "hazard_front":
            return "hazard_front"
        if hazard_context == "gap_search":
            return "gap_search"
        if (
            front_safe
            and goal_alignment > 0.4
            and (hazard_front or hazard_near or lateral_hazard)
            and hazard_channel
        ):
            return "safe_crossing"
        if hazard_context == "hazard_near":
            return "hazard_near"
        if topology_context == "doorway" and transition_context == "cross":
            return "doorway_cross"
        if topology_context == "doorway":
            return "doorway_approach"
        if topology_context == "corridor":
            return "corridor_follow"
        if topology_context == "wall_follow":
            return "wall_follow"
        if topology_context == "room_center" and goal_context == "goal_hidden":
            return "room_center"
        if goal_context == "goal_visible_near":
            return "goal_visible_near"
        if goal_context == "goal_visible_far":
            return "goal_visible_far"
        if goal_dist <= 3.0 and topology_context in {"doorway", "room_center", "open_space"}:
            return "goal_room_entry"
        if topology_context == "open_space":
            return "open_space_explore"
        return f"{goal_context}__{topology_context}__{hazard_context}__{transition_context}"

    def _patch_hash_token(self, scene: SceneState) -> SceneToken:
        digest = hashlib.blake2b(
            bytes(int(v) & 0xFF for v in scene.local_patch),
            digest_size=8,
        ).digest()
        return SceneToken(
            token_type=f"patch_{int.from_bytes(digest, 'little', signed=False)}",
            confidence=0.5,
            attributes={"local_patch_size": len(scene.local_patch)},
        )
