from songline_drive.types import IntentType, PlannerQuery, SemanticTargetPredicate


def build_planner_query(intent_type: IntentType, goal_xy=None) -> PlannerQuery:
    if intent_type == IntentType.REACH_SAFE_EXIT:
        return PlannerQuery(
            intent_type=intent_type,
            target_predicate=SemanticTargetPredicate(
                tag_name="safe_exit",
                min_confidence=0.5,
                score_weights={"safe_exit": 0.5},
            ),
            fallback_goal_xy=None if goal_xy is None else tuple(goal_xy),
            required_tags={"safe_exit": 0.5},
            preferred_tags={"safe_exit": 0.5},
            temporal_constraints={"min_freshness": 0.0},
            fallback_mode="goal_xy" if goal_xy is not None else "semantic_only",
        )

    if intent_type == IntentType.HAZARD_RECOVERY_EXIT:
        return PlannerQuery(
            intent_type=intent_type,
            target_predicate=SemanticTargetPredicate(
                tag_name="hazard_recovery_route",
                min_confidence=0.5,
                score_weights={"hazard_recovery_route": 0.6, "goal_region": 0.2},
                penalty_weights={"hazard_edge": 0.1},
            ),
            fallback_goal_xy=None if goal_xy is None else tuple(goal_xy),
            required_tags={"hazard_recovery_route": 0.5},
            preferred_tags={
                "hazard_recovery_route": 0.6,
                "goal_region": 0.2,
                "post_hazard_goal_rejoin": 0.25,
            },
            penalty_tags={"hazard_edge": 0.1, "adjacent_hazard": 0.15},
            temporal_constraints={"min_freshness": 0.05},
            fallback_mode="goal_xy" if goal_xy is not None else "semantic_only",
        )

    if intent_type == IntentType.FIND_GOAL_REGION:
        return PlannerQuery(
            intent_type=intent_type,
            target_predicate=SemanticTargetPredicate(
                tag_name="goal_region",
                min_confidence=0.5,
                score_weights={"goal_region": 0.5},
            ),
            fallback_goal_xy=None if goal_xy is None else tuple(goal_xy),
            required_tags={"goal_region": 0.5},
            preferred_tags={"goal_region": 0.5, "post_hazard_goal_rejoin": 0.15},
            temporal_constraints={"min_freshness": 0.0},
            fallback_mode="goal_xy" if goal_xy is not None else "semantic_only",
        )

    if intent_type == IntentType.FIND_WATER_SOURCE:
        return PlannerQuery(
            intent_type=intent_type,
            target_predicate=SemanticTargetPredicate(
                tag_name="water_source",
                min_confidence=0.25,
                required_tag_thresholds={
                    "water_candidate": 0.2,
                    "near_water": 0.2,
                },
                score_weights={
                    "water_source": 0.65,
                    "water_candidate": 0.35,
                    "water_nearby": 0.20,
                    "near_water": 0.25,
                },
                penalty_weights={
                    "hazard_edge": 0.15,
                    "adjacent_hazard": 0.20,
                },
                metadata={
                    "semantic_task": "water_search",
                    "requires_known_coordinates": False,
                    "min_visits_override": 1,
                    "use_concept_recall": True,
                    "required_match_mode": "any",
                },
            ),
            fallback_goal_xy=None if goal_xy is None else tuple(goal_xy),
            required_tags={
                "water_source": 0.25,
                "water_candidate": 0.2,
                "near_water": 0.2,
            },
            preferred_tags={
                "water_source": 0.65,
                "water_candidate": 0.35,
                "water_nearby": 0.20,
                "near_water": 0.25,
            },
            penalty_tags={
                "hazard_edge": 0.15,
                "adjacent_hazard": 0.20,
            },
            state_constraints={"thirst_min": 0.0},
            temporal_constraints={"min_freshness": 0.05},
            fallback_mode="semantic_only",
            metadata={"required_match_mode": "any"},
        )

    if intent_type == IntentType.FIND_SAFE_REST_ZONE:
        return PlannerQuery(
            intent_type=intent_type,
            target_predicate=SemanticTargetPredicate(
                tag_name="safe_rest_zone",
                min_confidence=0.25,
                required_tag_thresholds={
                    "rest_candidate": 0.2,
                    "open_safe_rest_zone": 0.2,
                },
                score_weights={
                    "safe_rest_zone": 0.60,
                    "rest_candidate": 0.35,
                    "rest_nearby": 0.20,
                    "safe_exit": 0.15,
                    "room_center": 0.10,
                    "open_safe_rest_zone": 0.25,
                },
                penalty_weights={
                    "hazard_edge": 0.20,
                    "corridor": 0.05,
                    "adjacent_hazard": 0.20,
                },
                metadata={
                    "semantic_task": "rest_search",
                    "requires_known_coordinates": False,
                    "min_visits_override": 1,
                    "use_concept_recall": True,
                    "required_match_mode": "any",
                },
            ),
            fallback_goal_xy=None if goal_xy is None else tuple(goal_xy),
            required_tags={
                "safe_rest_zone": 0.25,
                "rest_candidate": 0.2,
                "open_safe_rest_zone": 0.2,
            },
            preferred_tags={
                "safe_rest_zone": 0.60,
                "rest_candidate": 0.35,
                "rest_nearby": 0.20,
                "safe_exit": 0.15,
                "room_center": 0.10,
                "open_safe_rest_zone": 0.25,
            },
            penalty_tags={
                "hazard_edge": 0.20,
                "corridor": 0.05,
                "adjacent_hazard": 0.20,
            },
            state_constraints={"energy_max": 1.0},
            temporal_constraints={"min_freshness": 0.05},
            fallback_mode="semantic_only",
            metadata={"required_match_mode": "any"},
        )

    raise ValueError(f"Unsupported intent_type: {intent_type}")


def default_intent_for_task(task_name: str = "safe_exit") -> IntentType:
    if task_name == "safe_exit":
        return IntentType.REACH_SAFE_EXIT
    if task_name == "hazard_recovery":
        return IntentType.HAZARD_RECOVERY_EXIT
    if task_name == "goal_region":
        return IntentType.FIND_GOAL_REGION
    if task_name == "water":
        return IntentType.FIND_WATER_SOURCE
    if task_name == "rest":
        return IntentType.FIND_SAFE_REST_ZONE
    return IntentType.REACH_SAFE_EXIT
