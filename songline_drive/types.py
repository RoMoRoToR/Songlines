from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    speed: float
    yaw_rate: float
    acceleration: float


@dataclass
class RouteContext:
    route_id: Optional[str]
    target_lane_id: Optional[int]
    remaining_distance: float
    goal_alignment: float
    goal_xy: Optional[Tuple[int, int]] = None
    goal_visible: bool = False


@dataclass
class LaneContext:
    current_lane_id: Optional[int]
    left_lane_id: Optional[int]
    right_lane_id: Optional[int]
    lane_curvature: float
    drivable_width: float


@dataclass
class SignalContext:
    traffic_light_state: Optional[str]
    stop_line_distance: Optional[float]
    crosswalk_distance: Optional[float]


@dataclass
class SceneState:
    ego_state: EgoState
    route_context: RouteContext
    lane_context: LaneContext
    agents: List[Any]
    signals: SignalContext
    free_space_score: float
    risk_features: Dict[str, float] = field(default_factory=dict)
    local_patch: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class SceneToken:
    token_type: str
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    semantic_tags: Dict[str, float] = field(default_factory=dict)


class IntentType(str, Enum):
    REACH_SAFE_EXIT = "reach_safe_exit"
    HAZARD_RECOVERY_EXIT = "hazard_recovery_exit"
    FIND_GOAL_REGION = "find_goal_region"
    FIND_WATER_SOURCE = "find_water_source"


@dataclass
class NodeSemanticTag:
    name: str
    confidence: float = 0.0
    count: int = 0


@dataclass
class SemanticTargetPredicate:
    tag_name: str
    min_confidence: float = 0.5


@dataclass
class PlannerQuery:
    intent_type: IntentType
    target_predicate: SemanticTargetPredicate
    fallback_goal_xy: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    node_id: int
    token_type: str
    context_signature: Tuple[int, ...]
    phrase: Tuple[int, ...]
    visits: int = 0
    success_count: int = 0
    last_seen_step: int = -1
    progress_sum: float = 0.0
    progress_count: int = 0
    risk_sum: float = 0.0
    risk_count: int = 0
    comfort_cost_sum: float = 0.0
    comfort_cost_count: int = 0
    goal_alignment_sum: float = 0.0
    goal_alignment_count: int = 0
    reward_sum: float = 0.0
    reward_count: int = 0
    speed_sum: float = 0.0
    speed_count: int = 0
    freshness: float = 1.0
    confidence: float = 0.0
    progress_slow: float = 0.0
    risk_slow: float = 0.0
    success_slow: float = 0.0
    comfort_slow: float = 0.0
    goal_alignment_slow: float = 0.0
    progress_fast: float = 0.0
    risk_fast: float = 0.0
    success_fast: float = 0.0
    comfort_fast: float = 0.0
    goal_alignment_fast: float = 0.0
    progress_var: float = 0.0
    risk_var: float = 0.0
    success_var: float = 0.0
    reuse_score: float = 0.0
    utility_cached: float = 0.0
    uncertainty_sum: float = 0.0
    uncertainty_count: int = 0
    pose_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    pose_count: int = 0
    goal_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    goal_count: int = 0
    phase_histogram: Dict[str, int] = field(default_factory=dict)
    semantic_tag_counts: Dict[str, int] = field(default_factory=dict)
    semantic_tag_confidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: int
    dst: int
    weight: int = 1
    success_weight: float = 0.0
    risk_weight: float = 0.0
    freshness: float = 1.0
    confidence: float = 0.0
    last_seen_step: int = -1
    transition_success_slow: float = 0.0
    transition_success_fast: float = 0.0
    transition_risk_slow: float = 0.0
    transition_risk_fast: float = 0.0
    transition_cost_slow: float = 0.0
    transition_cost_fast: float = 0.0


@dataclass
class ManeuverPlan:
    token_sequence: List[str]
    node_path: List[int]
    utility: float
    graph_path_length: int
    target_node_id: Optional[int] = None
    waypoint_xy: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
