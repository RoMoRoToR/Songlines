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
    FIND_SAFE_REST_ZONE = "find_safe_rest_zone"


@dataclass
class NodeSemanticTag:
    name: str
    confidence: float = 0.0
    count: int = 0


@dataclass
class SemanticTargetPredicate:
    tag_name: str
    min_confidence: float = 0.5
    required_tag_thresholds: Dict[str, float] = field(default_factory=dict)
    score_weights: Dict[str, float] = field(default_factory=dict)
    penalty_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerQuery:
    intent_type: IntentType
    target_predicate: SemanticTargetPredicate
    fallback_goal_xy: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    required_tags: Dict[str, float] = field(default_factory=dict)
    preferred_tags: Dict[str, float] = field(default_factory=dict)
    penalty_tags: Dict[str, float] = field(default_factory=dict)
    state_constraints: Dict[str, float] = field(default_factory=dict)
    temporal_constraints: Dict[str, float] = field(default_factory=dict)
    fallback_mode: str = "goal_xy"


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
    pose_sq_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    pose_count: int = 0
    goal_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    goal_sq_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
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


@dataclass
class OutcomeWindow:
    success_within_10_steps: int = 0
    goal_progress_after_10_steps: float = 0.0
    entered_hazard_again: int = 0
    restored_safety: int = 0
    resource_reached_after_10_steps: int = 0


@dataclass
class EpisodeStepRecord:
    step_idx: int
    node_id: Optional[int]
    token_type: Optional[str]
    active_intent: Optional[str]
    symbolic_node_id: Optional[int] = None
    intent_reason: str = ""
    agent_state: Dict[str, Any] = field(default_factory=dict)
    semantic_tags: Dict[str, float] = field(default_factory=dict)
    observations: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    outcome_window: OutcomeWindow = field(default_factory=OutcomeWindow)


@dataclass
class EpisodeRecord:
    episode_id: int
    env_id: str = ""
    task_mode: str = "default"
    node_sequence: List[int] = field(default_factory=list)
    intent_sequence: List[str] = field(default_factory=list)
    state_trace: List[Dict[str, Any]] = field(default_factory=list)
    step_records: List[EpisodeStepRecord] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryDebugRecord:
    episode_id: int
    step_idx: int
    current_node_id: Optional[int]
    intent_type: Optional[str]
    query_tag_name: Optional[str]
    candidate_node_ids: List[int] = field(default_factory=list)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    selected_node_id: Optional[int] = None
    selected_source: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptCluster:
    cluster_id: str
    concept_tag: str
    member_node_ids: List[int] = field(default_factory=list)
    relation_profile: Dict[str, float] = field(default_factory=dict)
    semantic_profile: Dict[str, float] = field(default_factory=dict)
    pose_mean: Optional[Tuple[float, float]] = None
    pose_var: Optional[Tuple[float, float]] = None
    freshness_mean: float = 0.0
    confidence_mean: float = 0.0


@dataclass
class SymbolicNode:
    symbolic_node_id: int
    label: str
    dominant_tag: str
    phrase_signatures: List[List[str]] = field(default_factory=list)
    member_graph_node_ids: List[int] = field(default_factory=list)
    semantic_tag_confidence: Dict[str, float] = field(default_factory=dict)
    semantic_tag_counts: Dict[str, int] = field(default_factory=dict)
    pose_mean: Optional[Tuple[float, float]] = None
    pose_var: Optional[Tuple[float, float]] = None
    visits: int = 0
    freshness: float = 1.0
    confidence: float = 0.0
    utility_fast: float = 0.0
    utility_slow: float = 0.0
    utility_cached: float = 0.0
    last_seen_step: int = -1
    relation_confidence: Dict[str, float] = field(default_factory=dict)
    intent_outcome_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SymbolicTransition:
    src: int
    dst: int
    success_fast: float = 0.0
    success_slow: float = 0.0
    risk_fast: float = 0.0
    risk_slow: float = 0.0
    cost_fast: float = 0.0
    cost_slow: float = 0.0
    freshness: float = 1.0
    confidence: float = 0.0
    last_seen_step: int = -1
    context_histogram: Dict[str, int] = field(default_factory=dict)


@dataclass
class RelationRecord:
    relation_name: str
    symbolic_node_id: int
    target_concept_tag: Optional[str] = None
    confidence: float = 0.0
    support_count: int = 0
    freshness: float = 1.0
    last_seen_step: int = -1


@dataclass
class ConceptRecord:
    concept_tag: str
    member_symbolic_node_ids: List[int] = field(default_factory=list)
    semantic_profile: Dict[str, float] = field(default_factory=dict)
    relation_profile: Dict[str, float] = field(default_factory=dict)
    freshness_mean: float = 0.0
    confidence_mean: float = 0.0
    support_count: int = 0
    last_seen_step: int = -1
