from dataclasses import dataclass, field
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


@dataclass
class GraphNode:
    node_id: int
    token_type: str
    context_signature: Tuple[int, ...]
    phrase: Tuple[int, ...]
    visits: int = 0
    success_count: int = 0
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
    reuse_score: float = 0.0
    uncertainty_sum: float = 0.0
    uncertainty_count: int = 0
    last_seen_step: int = -1
    pose_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    pose_count: int = 0
    goal_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    goal_count: int = 0


@dataclass
class GraphEdge:
    src: int
    dst: int
    weight: int = 1
    success_weight: float = 0.0
    risk_weight: float = 0.0
    freshness: float = 1.0


@dataclass
class ManeuverPlan:
    token_sequence: List[str]
    node_path: List[int]
    utility: float
    graph_path_length: int
    target_node_id: Optional[int] = None
    waypoint_xy: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
