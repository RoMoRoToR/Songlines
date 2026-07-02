from dataclasses import dataclass

from songline_drive.types import IntentType


@dataclass
class AgentState:
    thirst: float = 0.0
    energy: float = 1.0
    risk_budget: float = 1.0
    task_phase: str = "explore"
    previous_task_phase: str = "explore"
    recent_hazard_exit_steps: int = 0
    carrying_object: bool = False
    steps_elapsed: int = 0
    active_intent: IntentType = IntentType.FIND_GOAL_REGION
    active_intent_reason: str = "init"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def infer_task_phase(scene=None, token_label: str = None) -> str:
    if token_label in {"hazard_front", "gap_search", "gap_aligned", "safe_crossing", "post_hazard"}:
        return "hazard_navigation"
    if scene is None:
        return "explore"
    risk = scene.risk_features
    if float(risk.get("place_is_goal_region", 0.0)) > 0.0 or bool(scene.route_context.goal_visible):
        return "goal_approach"
    if float(risk.get("place_is_corridor", 0.0)) > 0.0:
        return "corridor_traverse"
    return "explore"


def update_agent_state(
    agent_state: AgentState,
    scene=None,
    token_label: str = None,
):
    risk = 0.0
    if scene is not None:
        rf = scene.risk_features
        risk = max(
            float(rf.get("hazard_front", 0.0)),
            0.75 * float(rf.get("hazard_near", 0.0)),
            0.5 * float(rf.get("lateral_hazard", 0.0)),
        )
    next_task_phase = infer_task_phase(scene=scene, token_label=token_label)
    previous_task_phase = str(agent_state.task_phase)

    agent_state.steps_elapsed += 1
    agent_state.thirst = _clamp(agent_state.thirst + 0.005)
    agent_state.energy = _clamp(agent_state.energy - 0.002 - (0.006 * risk))
    risk_budget_delta = (0.03 * (1.0 - risk)) - (0.08 * risk)
    agent_state.risk_budget = _clamp(agent_state.risk_budget + risk_budget_delta)
    agent_state.previous_task_phase = previous_task_phase
    agent_state.task_phase = next_task_phase

    if previous_task_phase == "hazard_navigation" and next_task_phase != "hazard_navigation":
        agent_state.recent_hazard_exit_steps = 3
    elif next_task_phase == "hazard_navigation":
        agent_state.recent_hazard_exit_steps = 0
    elif agent_state.recent_hazard_exit_steps > 0:
        agent_state.recent_hazard_exit_steps -= 1
    return agent_state


class IntentPolicy:
    def __init__(
        self,
        low_risk_budget_threshold: float = 0.45,
        low_energy_threshold: float = 0.18,
        hazard_intent: IntentType = IntentType.REACH_SAFE_EXIT,
        water_intent: IntentType = None,
        rest_intent: IntentType = None,
        thirst_on_threshold: float = 0.10,
        thirst_off_threshold: float = 0.04,
        water_local_activation_threshold: float = 0.0,
        water_local_hold_threshold: float = 0.0,
        rest_energy_on_threshold: float = 0.95,
        rest_energy_off_threshold: float = 0.98,
        rest_local_activation_threshold: float = 0.0,
        rest_local_hold_threshold: float = 0.0,
    ):
        self.low_risk_budget_threshold = float(low_risk_budget_threshold)
        self.low_energy_threshold = float(low_energy_threshold)
        self.hazard_intent = hazard_intent
        self.water_intent = water_intent
        self.rest_intent = rest_intent
        self.thirst_on_threshold = float(thirst_on_threshold)
        self.thirst_off_threshold = float(thirst_off_threshold)
        self.water_local_activation_threshold = float(water_local_activation_threshold)
        self.water_local_hold_threshold = float(water_local_hold_threshold)
        self.rest_energy_on_threshold = float(rest_energy_on_threshold)
        self.rest_energy_off_threshold = float(rest_energy_off_threshold)
        self.rest_local_activation_threshold = float(rest_local_activation_threshold)
        self.rest_local_hold_threshold = float(rest_local_hold_threshold)

    def select_intent(self, agent_state: AgentState, scene=None) -> IntentType:
        intent, _ = self.select_intent_with_reason(agent_state, scene=scene)
        return intent

    def select_intent_with_reason(self, agent_state: AgentState, scene=None):
        if scene is None:
            return IntentType.FIND_GOAL_REGION, "no_scene_default_goal"

        risk = scene.risk_features
        if agent_state.task_phase == "hazard_navigation":
            return self.hazard_intent, "hazard_navigation"
        # Once a hazard-recovery maneuver has cleared the active hazard band, hand off
        # immediately to goal-directed navigation instead of lingering on recovery nodes.
        if (
            self.hazard_intent == IntentType.HAZARD_RECOVERY_EXIT
            and agent_state.recent_hazard_exit_steps > 0
            and float(risk.get("hazard_front", 0.0)) <= 0.0
        ):
            return IntentType.FIND_GOAL_REGION, "post_hazard_goal_handoff"
        if self.water_intent is not None:
            water_visible = float(risk.get("water_visible", 0.0))
            water_accessible = float(risk.get("water_accessible", 0.0))
            water_confidence_local = float(risk.get("water_confidence_local", 0.0))
            if agent_state.active_intent == self.water_intent:
                if (
                    water_visible > 0.0
                    or water_accessible > 0.0
                    or water_confidence_local >= self.water_local_hold_threshold > 0.0
                ):
                    return self.water_intent, "water_local_evidence_hold"
                if agent_state.thirst >= self.thirst_off_threshold:
                    return self.water_intent, "thirst_hysteresis_hold"
                return IntentType.FIND_GOAL_REGION, "thirst_recovered_goal_default"
            if (
                water_visible > 0.0
                or water_accessible > 0.0
                or (
                    self.water_local_activation_threshold > 0.0
                    and water_confidence_local >= self.water_local_activation_threshold
                )
            ):
                return self.water_intent, "water_local_evidence"
            if agent_state.thirst >= self.thirst_on_threshold:
                return self.water_intent, "thirst_above_threshold"
            return IntentType.FIND_GOAL_REGION, "thirst_below_threshold"
        if self.rest_intent is not None:
            rest_visible = float(risk.get("rest_visible", 0.0))
            rest_accessible = float(risk.get("rest_accessible", 0.0))
            rest_confidence_local = float(risk.get("rest_confidence_local", 0.0))
            if agent_state.active_intent == self.rest_intent:
                if (
                    rest_visible > 0.0
                    or rest_accessible > 0.0
                    or rest_confidence_local >= self.rest_local_hold_threshold > 0.0
                ):
                    return self.rest_intent, "rest_local_evidence_hold"
                if agent_state.energy <= self.rest_energy_off_threshold:
                    return self.rest_intent, "low_energy_hysteresis_hold"
                return IntentType.FIND_GOAL_REGION, "energy_recovered_goal_default"
            if (
                rest_visible > 0.0
                or rest_accessible > 0.0
                or (
                    self.rest_local_activation_threshold > 0.0
                    and rest_confidence_local >= self.rest_local_activation_threshold
                )
            ):
                return self.rest_intent, "rest_local_evidence"
            if agent_state.energy <= self.rest_energy_on_threshold:
                return self.rest_intent, "low_energy_below_threshold"
            return IntentType.FIND_GOAL_REGION, "energy_above_threshold"
        if float(risk.get("hazard_front", 0.0)) > 0.0 or float(risk.get("hazard_near", 0.0)) > 0.0:
            return self.hazard_intent, "hazard_proximity"
        if agent_state.risk_budget <= self.low_risk_budget_threshold:
            return IntentType.REACH_SAFE_EXIT, "low_risk_budget"
        if agent_state.energy <= self.low_energy_threshold:
            return IntentType.REACH_SAFE_EXIT, "low_energy"
        return IntentType.FIND_GOAL_REGION, "default_goal"
