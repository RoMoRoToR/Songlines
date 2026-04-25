import argparse
import csv
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from scripts.songline_minigrid import (
    current_phrase_node,
    current_planner_query,
    ensure_dir,
    export_debug_trace,
    export_demo_artifacts,
    export_run_summary,
    get_phrase_length_mean,
    resolve_active_intent_type,
    safe_rate,
    scene_comfort_proxy,
    scene_risk_proxy,
    scene_token_to_int,
)
from songline_drive.agent_state import AgentState, IntentPolicy, update_agent_state
from songline_drive.graph_memory import DynamicSonglineGraph
from songline_drive.graph_rollout import GraphRolloutPlanner
from songline_drive.maneuver_selector import ManeuverSelector
from songline_drive.miniworld_support import (
    MINIWORLD_ENV_ALIASES,
    MiniWorldSceneEncoder,
    MiniWorldTrajectoryPlanner,
    build_miniworld_env,
    canonical_miniworld_env_id,
    check_miniworld_runtime,
    ensure_miniworld_available,
    get_agent_position_xz,
    get_goal_position_xz,
    miniworld_symbolic_observation,
)
from songline_drive.scene_tokenizer import SceneTokenizer
from songline_drive.symbolic_memory import SymbolicMemory
from utils.lz_memory import SymbolicTokenizer


def make_method_name(agent_mode, songline_policy):
    if agent_mode == "songline":
        return f"songline_{songline_policy}"
    return agent_mode


def build_memory(args):
    tokenizer = None
    encoder = None
    if args.token_source == "symbolic_hash":
        tokenizer = SymbolicTokenizer(
            mode=args.tokenizer_mode,
            proj_dim=args.tokenizer_proj_dim,
            seed=args.seed,
        )
    elif args.token_source == "scene_semantic":
        encoder = MiniWorldSceneEncoder()
        tokenizer = SceneTokenizer(mode="semantic")
    elif args.token_source == "scene_patch_hash":
        encoder = MiniWorldSceneEncoder()
        tokenizer = SceneTokenizer(mode="patch_hash")
    else:
        raise ValueError(f"Unknown token_source: {args.token_source}")
    memory = DynamicSonglineGraph(
        min_goal_visits=args.min_goal_visits,
        graph_update_mode=args.graph_update_mode,
    )
    return tokenizer, encoder, SymbolicMemory(memory), GraphRolloutPlanner()


def build_local_planner(args):
    return ManeuverSelector(), MiniWorldTrajectoryPlanner()


def select_graph_waypoint_miniworld(
    memory,
    planner,
    selector,
    top_k,
    rollout_horizon,
    current_token_label=None,
    planner_query=None,
):
    plans = memory.query_paths(
        planner_query=planner_query,
        current_node_id=current_phrase_node(memory),
        planner=planner,
        horizon=rollout_horizon,
        top_k=top_k,
    )
    if not plans:
        return None

    plan = plans[0]
    command = selector.select(plan, current_token=current_token_label)
    waypoint_xy = command.get("waypoint_xy")
    if waypoint_xy is None:
        return None
    planner_debug = dict(plan.metadata)
    if plan.target_node_id is not None and plan.target_node_id in memory.nodes:
        planner_debug["selected_node_semantic_tag_confidence"] = dict(
            memory.nodes[plan.target_node_id].get("semantic_tag_confidence", {})
        )
    else:
        planner_debug["selected_node_semantic_tag_confidence"] = {}
    return {
        "waypoint_xy": np.asarray(waypoint_xy, dtype=np.float32).copy(),
        "target_node_id": command.get("target_node_id"),
        "next_node_id": int(plan.metadata["next_node_id"]),
        "graph_path_length": int(command.get("graph_path_length", plan.graph_path_length)),
        "utility": float(plan.utility),
        "token_sequence": list(plan.token_sequence),
        "maneuver_command": command,
        "planner_debug": planner_debug,
    }


def select_semantic_waypoint_fallback_continuous(memory, planner_query, current_node_id=None, source_xy=None, top_k=5):
    if planner_query is None or memory is None or not hasattr(memory, "candidate_nodes_for_query"):
        return None
    candidate_ids = memory.candidate_nodes_for_query(
        planner_query,
        top_k=top_k,
        current_node_id=current_node_id,
    )
    if not candidate_ids:
        return None
    source_arr = None if source_xy is None else np.asarray(source_xy, dtype=np.float32)
    selected_node_id = None
    waypoint_xy = None
    for node_id in candidate_ids:
        node_id = int(node_id)
        if current_node_id is not None and node_id == int(current_node_id):
            continue
        pose_xy = memory.get_mean_xy(node_id, "pose")
        if pose_xy is None:
            continue
        pose_xy = np.asarray(pose_xy, dtype=np.float32)
        if source_arr is not None and float(np.linalg.norm(pose_xy - source_arr)) < 1e-3:
            continue
        selected_node_id = node_id
        waypoint_xy = pose_xy
        break
    if selected_node_id is None or waypoint_xy is None:
        return None
    query_tag_name = str(planner_query.target_predicate.tag_name)
    candidate_base_utilities = {str(node_id): float(memory.node_utility(node_id)) for node_id in candidate_ids}
    candidate_tag_confidences = {
        str(node_id): float(memory.nodes[node_id].get("semantic_tag_confidence", {}).get(query_tag_name, 0.0))
        for node_id in candidate_ids
    }
    candidate_intent_scores = {
        str(node_id): float(memory.node_intent_score(node_id, planner_query=planner_query))
        for node_id in candidate_ids
    }
    return {
        "waypoint_xy": np.asarray(waypoint_xy, dtype=np.float32).copy(),
        "target_node_id": int(selected_node_id),
        "next_node_id": int(selected_node_id),
        "graph_path_length": 0,
        "utility": float(candidate_intent_scores[str(selected_node_id)]),
        "token_sequence": [],
        "maneuver_command": {
            "command_type": "go_to_waypoint",
            "waypoint_xy": tuple(float(v) for v in waypoint_xy),
        },
        "planner_debug": {
            "next_node_id": int(selected_node_id),
            "intent_type": str(planner_query.intent_type.value),
            "query_tag_name": query_tag_name,
            "used_intent_query": True,
            "candidate_node_ids": [int(cid) for cid in candidate_ids],
            "candidate_base_utilities": candidate_base_utilities,
            "candidate_tag_confidences": candidate_tag_confidences,
            "candidate_intent_scores": candidate_intent_scores,
            "candidate_concept_membership": dict(
                getattr(memory, "last_concept_query_debug", {}).get("cluster_membership", {})
            ),
            "concept_query_debug": dict(getattr(memory, "last_concept_query_debug", {}) or {}),
            "selected_tag_confidence": float(candidate_tag_confidences.get(str(selected_node_id), 0.0)),
            "selected_intent_score": float(candidate_intent_scores.get(str(selected_node_id), 0.0)),
            "selected_plan_utility": float(candidate_intent_scores.get(str(selected_node_id), 0.0)),
            "selected_node_semantic_tag_confidence": dict(
                memory.nodes[selected_node_id].get("semantic_tag_confidence", {})
            ),
            "plan_source": "semantic_pose_fallback",
        },
    }


def build_env(args):
    return build_miniworld_env(args.env_id, render_mode="rgb_array")


def summarise_run(run_summary, memory):
    final_nodes = 0 if memory is None else int(len(memory.nodes))
    final_edges = 0 if memory is None else int(sum(len(v) for v in memory.edges.values()))
    return {
        "success_rate": float(np.mean(run_summary["successes"])) if run_summary["successes"] else 0.0,
        "avg_return": float(np.mean(run_summary["episode_returns"])) if run_summary["episode_returns"] else 0.0,
        "avg_steps_to_goal": float(np.mean(run_summary["episode_lengths"])) if run_summary["episode_lengths"] else 0.0,
        "avg_steps": float(np.mean(run_summary["episode_lengths"])) if run_summary["episode_lengths"] else 0.0,
        "intervention_rate": 0.0 if memory is None else safe_rate(memory.interventions, memory.intervention_attempts),
        "plan_hit_rate": 0.0 if memory is None else safe_rate(memory.plan_hits, memory.plan_total),
        "graph_nodes": final_nodes,
        "graph_edges": final_edges,
        "phrase_length_mean": get_phrase_length_mean(memory),
        "subgoal_reach_rate": safe_rate(float(np.sum(run_summary["subgoal_reached"])), len(run_summary["subgoal_reached"])),
        "node_reuse_rate": safe_rate(float(np.sum(run_summary["node_reuse_rate"])), len(run_summary["node_reuse_rate"])),
        "new_nodes_per_episode": float(np.mean(run_summary["new_nodes_per_episode"])) if run_summary["new_nodes_per_episode"] else 0.0,
        "graph_path_length": float(np.mean(run_summary["graph_path_length"])) if run_summary["graph_path_length"] else 0.0,
        "final_nodes": final_nodes,
        "final_edges": final_edges,
    }


def run_songline_miniworld_experiment(args, export_outputs=True, verbose=True):
    env = build_env(args)
    method_name = make_method_name(args.agent_mode, args.songline_policy)

    tokenizer = None
    scene_encoder = None
    memory = None
    rollout_planner = None
    maneuver_selector = None
    trajectory_planner = None
    if args.agent_mode == "songline":
        tokenizer, scene_encoder, memory, rollout_planner = build_memory(args)
        maneuver_selector, trajectory_planner = build_local_planner(args)
    else:
        _, trajectory_planner = build_local_planner(args)

    run_summary = {
        "env_id": canonical_miniworld_env_id(args.env_id),
        "agent_mode": args.agent_mode,
        "songline_policy": args.songline_policy,
        "method": method_name,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "intent_mode": args.intent_mode,
        "intent_selection_mode": args.intent_selection_mode,
        "intent_type": args.intent_type,
        "semantic_retrieval_mode": args.semantic_retrieval_mode,
        "episode_returns": [],
        "episode_lengths": [],
        "successes": [],
        "graph_nodes": [],
        "graph_edges": [],
        "intervention_rate": [],
        "plan_hit_rate": [],
        "phrase_length_mean": [],
        "subgoal_reached": [],
        "node_reuse_rate": [],
        "new_nodes_per_episode": [],
        "graph_path_length": [],
        "episode_metrics": [],
    }
    debug_trace_rows = []
    demo_frames = []
    demo_frame_meta = []

    intent_policy = IntentPolicy()
    if args.intent_selection_mode == "state_v1":
        intent_policy = IntentPolicy(
            water_intent=None,
            rest_intent=None,
            hazard_intent=parse_intent_type(args.intent_type),
        )

    total_step_idx = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if tokenizer is not None and hasattr(tokenizer, "reset"):
            tokenizer.reset()
        if trajectory_planner is not None and hasattr(trajectory_planner, "reset"):
            trajectory_planner.reset()
        if memory is not None and hasattr(memory, "start_episode"):
            memory.start_episode(episode_id=ep + 1, env_id=canonical_miniworld_env_id(args.env_id), task_mode="default")

        episode_return = 0.0
        success = 0
        active_maneuver_command = None
        active_target_node_id = None
        active_next_node_id = None
        active_planner_debug = None
        active_graph_path_length = 0
        subgoal_hits = 0
        graph_path_lengths = []
        agent_state = AgentState()
        goal_xy = get_goal_position_xz(env)
        prev_graph_node_id = None
        current_graph_node_id = None
        last_goal_distance = None
        new_nodes_start = 0 if memory is None else len(memory.nodes)

        record_demo_episode = bool(getattr(args, "record_demo", False) and int(getattr(args, "demo_episode", 1)) == ep + 1)

        for step in range(args.max_steps):
            total_step_idx += 1
            agent_pos = get_agent_position_xz(env)
            goal_xy = get_goal_position_xz(env)
            distance_before = None if goal_xy is None else float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(agent_pos)))

            if args.agent_mode == "songline":
                if args.token_source == "symbolic_hash":
                    obs_vec = miniworld_symbolic_observation(obs, env)
                    token_id = tokenizer.encode(obs_vec)
                    scene = None
                    scene_token = None
                    token_label = f"hash_{int(token_id)}"
                else:
                    scene = scene_encoder.encode(env, obs=obs)
                    scene_token = tokenizer.tokenize(scene)
                    token_id = scene_token_to_int(scene_token)
                    token_label = str(scene_token.token_type)
                observe_result = memory.observe(
                    scene_token=scene_token,
                    scene_state=scene,
                    agent_state=agent_state,
                    step_info={
                        "phase": "pre_action",
                        "step_idx": int(total_step_idx),
                        "token_id": int(token_id),
                        "token_label": str(token_label),
                        "pose_xy": agent_pos,
                        "goal_xy": goal_xy,
                        "reward": 0.0,
                        "risk": scene_risk_proxy(scene) if scene is not None else None,
                        "comfort_cost": scene_comfort_proxy(scene) if scene is not None else None,
                        "goal_alignment": None if scene is None else scene.route_context.goal_alignment,
                    },
                )
                prev_graph_node_id = observe_result.get("previous_graph_node_id")
                current_graph_node_id = observe_result.get("current_graph_node_id")
                if scene is None:
                    scene = scene_encoder.encode(env, obs=obs)
                agent_state = update_agent_state(agent_state, scene=scene, token_label=token_label)
                active_intent_type, active_intent_reason = resolve_active_intent_type(
                    args,
                    agent_state=agent_state,
                    intent_policy=intent_policy,
                    scene=scene,
                )
                agent_state.active_intent = active_intent_type if active_intent_type is not None else agent_state.active_intent
                agent_state.active_intent_reason = str(active_intent_reason)
                planner_query = current_planner_query(
                    args,
                    goal_xy=goal_xy,
                    active_intent_type=active_intent_type,
                    agent_state=agent_state,
                )

                should_plan = active_maneuver_command is None or (step % max(1, int(args.suggest_every)) == 0)
                if should_plan:
                    suggestion = select_graph_waypoint_miniworld(
                        memory,
                        rollout_planner,
                        maneuver_selector,
                        top_k=args.top_k_goals,
                        rollout_horizon=args.graph_rollout_horizon,
                        current_token_label=token_label,
                        planner_query=planner_query,
                    )
                    if suggestion is None and planner_query is not None:
                        suggestion = select_semantic_waypoint_fallback_continuous(
                            memory,
                            planner_query=planner_query,
                            current_node_id=current_graph_node_id,
                            source_xy=agent_pos,
                            top_k=args.top_k_goals,
                        )
                    if suggestion is not None:
                        active_maneuver_command = suggestion["maneuver_command"]
                        active_target_node_id = suggestion["target_node_id"]
                        active_next_node_id = suggestion["next_node_id"]
                        active_planner_debug = suggestion["planner_debug"]
                        active_graph_path_length = int(suggestion["graph_path_length"])
                        graph_path_lengths.append(float(active_graph_path_length))
                    if memory is not None and hasattr(memory, "record_query"):
                        memory.record_query(
                            step_idx=int(total_step_idx),
                            current_node_id=current_graph_node_id,
                            planner_query=planner_query,
                            planner_debug={} if active_planner_debug is None else dict(active_planner_debug),
                            selected_node_id=active_target_node_id,
                            selected_source="graph_plan" if active_planner_debug is not None else "none",
                        )
            else:
                scene = None
                scene_token = None
                token_label = "policy_free"
                active_intent_reason = "not_applicable"
                active_intent_type = None
                active_planner_debug = None

            if args.agent_mode == "random":
                action = int(env.action_space.sample())
            elif args.agent_mode == "greedy":
                if goal_xy is None:
                    action = int(env.action_space.sample())
                else:
                    action = int(
                        trajectory_planner.next_action(
                            env,
                            {
                                "command_type": "go_to_waypoint",
                                "waypoint_xy": tuple(float(v) for v in np.asarray(goal_xy, dtype=np.float32)),
                            },
                        )
                    )
            else:
                if active_maneuver_command is None and goal_xy is not None:
                    action = int(
                        trajectory_planner.next_action(
                            env,
                            {
                                "command_type": "go_to_waypoint",
                                "waypoint_xy": tuple(float(v) for v in np.asarray(goal_xy, dtype=np.float32)),
                            },
                        )
                    )
                else:
                    action = int(trajectory_planner.next_action(env, active_maneuver_command or {}))

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            agent_pos_new = get_agent_position_xz(env)
            goal_after = get_goal_position_xz(env)
            distance_after = None if goal_after is None else float(np.linalg.norm(np.asarray(goal_after) - np.asarray(agent_pos_new)))
            delta_distance = None
            if distance_before is not None and distance_after is not None:
                delta_distance = float(distance_before - distance_after)

            if args.agent_mode == "songline":
                post_scene = scene_encoder.encode(env, obs=obs)
                if last_goal_distance is not None and distance_after is not None:
                    memory.record_plan_outcome(distance_after < last_goal_distance)
                memory.observe(
                    scene_token=scene_token,
                    scene_state=post_scene,
                    agent_state=agent_state,
                    step_info={
                        "phase": "post_action",
                        "step_idx": int(total_step_idx),
                        "previous_graph_node_id": None if prev_graph_node_id is None else int(prev_graph_node_id),
                        "current_graph_node_id": None if current_graph_node_id is None else int(current_graph_node_id),
                        "token_label": str(token_label),
                        "pose_xy": agent_pos_new,
                        "goal_xy": goal_after,
                        "reward": float(reward),
                        "progress": delta_distance,
                        "risk": scene_risk_proxy(post_scene),
                        "success": 1.0 if float(reward) > 0.0 else 0.0,
                        "comfort_cost": scene_comfort_proxy(post_scene),
                        "goal_alignment": post_scene.route_context.goal_alignment,
                        "transition_success": None if delta_distance is None else (1.0 if delta_distance > 0.0 else 0.0),
                        "transition_risk": scene_risk_proxy(post_scene),
                        "transition_cost": 1.0 if delta_distance is not None else None,
                        "context_label": str(getattr(agent_state, "task_phase", token_label)),
                        "active_intent": None if active_intent_type is None else str(active_intent_type.value),
                        "intent_reason": str(active_intent_reason),
                        "semantic_tags": {} if scene_token is None else dict(getattr(scene_token, "semantic_tags", {})),
                        "observations": {
                            "pos_before": [float(agent_pos[0]), float(agent_pos[1])],
                            "pos_after": [float(agent_pos_new[0]), float(agent_pos_new[1])],
                            "distance_before": None if distance_before is None else float(distance_before),
                            "distance_after": None if distance_after is None else float(distance_after),
                            "planner_query_intent": None if active_planner_debug is None else active_planner_debug.get("intent_type"),
                            "planner_query_tag_name": None if active_planner_debug is None else active_planner_debug.get("query_tag_name"),
                            "target_node_id": None if active_target_node_id is None else int(active_target_node_id),
                        },
                        "outcome": {
                            "reward": float(reward),
                            "success_signal": int(float(reward) > 0.0),
                            "delta_distance": 0.0 if delta_distance is None else float(delta_distance),
                        },
                    },
                )

            if active_maneuver_command is not None:
                waypoint = active_maneuver_command.get("waypoint_xy")
                if waypoint is not None:
                    waypoint = np.asarray(waypoint, dtype=np.float32)
                    if float(np.linalg.norm(agent_pos_new - waypoint)) <= 0.75:
                        active_maneuver_command = None
                        subgoal_hits += 1

            last_goal_distance = distance_after
            success = max(success, int(float(reward) > 0.0 or terminated))

            if args.debug_trace:
                debug_trace_rows.append(
                    {
                        "episode": int(ep + 1),
                        "step": int(step + 1),
                        "token": token_label,
                        "active_intent": None if active_intent_type is None else str(active_intent_type.value),
                        "intent_switch_reason": str(active_intent_reason),
                        "pos_x": float(agent_pos_new[0]),
                        "pos_z": float(agent_pos_new[1]),
                        "distance_before": None if distance_before is None else float(distance_before),
                        "distance_after": None if distance_after is None else float(distance_after),
                        "reward": float(reward),
                        "success": int(float(reward) > 0.0),
                        "target_node_id": None if active_target_node_id is None else int(active_target_node_id),
                        "planner_query_intent": None if active_planner_debug is None else active_planner_debug.get("intent_type"),
                        "planner_query_tag_name": None if active_planner_debug is None else active_planner_debug.get("query_tag_name"),
                        "planner_candidate_concept_membership": "{}" if active_planner_debug is None else json.dumps(active_planner_debug.get("candidate_concept_membership", {}), sort_keys=True),
                        "planner_concept_query_debug": "{}" if active_planner_debug is None else json.dumps(active_planner_debug.get("concept_query_debug", {}), sort_keys=True),
                    }
                )

            if record_demo_episode:
                frame = env.render()
                if frame is not None:
                    demo_frames.append(np.asarray(frame, dtype=np.uint8))
                    demo_frame_meta.append(
                        {
                            "episode": int(ep + 1),
                            "step": int(step + 1),
                            "token": token_label,
                            "action": int(action),
                            "intent": None if active_intent_type is None else str(active_intent_type.value),
                            "pos": tuple(float(v) for v in agent_pos_new),
                            "subgoal": None if active_maneuver_command is None or active_maneuver_command.get("waypoint_xy") is None else tuple(float(v) for v in active_maneuver_command.get("waypoint_xy")),
                            "distance_after": None if distance_after is None else float(distance_after),
                            "reward": float(reward),
                            "event": "goal_reached" if float(reward) > 0.0 else "",
                        }
                    )

            if terminated or truncated:
                break

        edge_count = 0 if memory is None else sum(len(v) for v in memory.edges.values())
        intervention_rate = 0.0 if memory is None else safe_rate(memory.interventions, memory.intervention_attempts)
        plan_hit_rate = 0.0 if memory is None else safe_rate(memory.plan_hits, memory.plan_total)
        node_reuse_rate = 0.0
        if args.agent_mode == "songline" and memory is not None:
            reused = sum(1 for node in memory.nodes.values() if node["visits"] > 1)
            node_reuse_rate = safe_rate(reused, len(memory.nodes))
        new_nodes = 0 if memory is None else len(memory.nodes) - new_nodes_start
        graph_path_length_mean = float(np.mean(graph_path_lengths)) if graph_path_lengths else 0.0
        episode_metrics = {
            "episode": ep + 1,
            "env_id": canonical_miniworld_env_id(args.env_id),
            "agent_mode": args.agent_mode,
            "songline_policy": args.songline_policy,
            "method": method_name,
            "seed": args.seed,
            "token_source": args.token_source,
            "intent_mode": args.intent_mode,
            "intent_selection_mode": args.intent_selection_mode,
            "intent_type": args.intent_type,
            "semantic_retrieval_mode": args.semantic_retrieval_mode,
            "return": float(episode_return),
            "steps_to_goal": int(step + 1),
            "steps": int(step + 1),
            "success": int(success),
            "intervention_rate": float(intervention_rate),
            "plan_hit_rate": float(plan_hit_rate),
            "graph_nodes": 0 if memory is None else int(len(memory.nodes)),
            "graph_edges": int(edge_count),
            "phrase_length_mean": float(get_phrase_length_mean(memory)),
            "subgoal_reach_rate": float(safe_rate(subgoal_hits, max(1, len(graph_path_lengths)))),
            "node_reuse_rate": float(node_reuse_rate),
            "new_nodes_per_episode": int(new_nodes),
            "graph_path_length": float(graph_path_length_mean),
        }
        run_summary["episode_returns"].append(float(episode_return))
        run_summary["episode_lengths"].append(int(step + 1))
        run_summary["successes"].append(int(success))
        run_summary["graph_nodes"].append(episode_metrics["graph_nodes"])
        run_summary["graph_edges"].append(episode_metrics["graph_edges"])
        run_summary["intervention_rate"].append(float(intervention_rate))
        run_summary["plan_hit_rate"].append(float(plan_hit_rate))
        run_summary["phrase_length_mean"].append(float(episode_metrics["phrase_length_mean"]))
        run_summary["subgoal_reached"].append(float(episode_metrics["subgoal_reach_rate"]))
        run_summary["node_reuse_rate"].append(float(node_reuse_rate))
        run_summary["new_nodes_per_episode"].append(int(new_nodes))
        run_summary["graph_path_length"].append(float(graph_path_length_mean))
        run_summary["episode_metrics"].append(episode_metrics)
        if memory is not None and hasattr(memory, "finalize_episode"):
            memory.finalize_episode(
                {
                    "success": int(success),
                    "return": float(episode_return),
                    "steps": int(step + 1),
                    "active_intent_type_final": None if active_intent_type is None else str(active_intent_type.value),
                }
            )
        if verbose:
            print(
                f"Episode {ep + 1:03d} | method={method_name} | return={episode_return:.3f} | "
                f"steps={step + 1:03d} | success={success} | nodes={episode_metrics['graph_nodes']} | "
                f"edges={episode_metrics['graph_edges']} | subgoal_hit={episode_metrics['subgoal_reach_rate']:.3f}"
            )

    summary = summarise_run(run_summary, memory)
    summary.update(
        {
            "env_id": canonical_miniworld_env_id(args.env_id),
            "agent_mode": args.agent_mode,
            "songline_policy": args.songline_policy,
            "method": method_name,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "intent_mode": args.intent_mode,
            "intent_selection_mode": args.intent_selection_mode,
            "intent_type": args.intent_type,
            "semantic_retrieval_mode": args.semantic_retrieval_mode,
        }
    )

    if export_outputs:
        with open(os.path.join(args.out_dir, "episodes.json"), "w") as f:
            json.dump(run_summary["episode_metrics"], f, indent=2)
        if memory is not None:
            memory.export(args.out_dir, env_idx=0)
        export_run_summary(args.out_dir, run_summary)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if args.debug_trace:
            export_debug_trace(args.out_dir, debug_trace_rows)
        if getattr(args, "record_demo", False):
            export_demo_artifacts(
                args.out_dir,
                demo_frames,
                demo_frame_meta,
                fps=int(getattr(args, "demo_fps", 3)),
            )
    env.close()
    return run_summary, summary


def parse_intent_type(name: str):
    from songline_drive.types import IntentType
    return IntentType(name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniWorld-Hallway-v0")
    parser.add_argument("--agent_mode", type=str, default="songline", choices=["random", "greedy", "songline"])
    parser.add_argument("--songline_policy", type=str, default="graph_path", choices=["subgoal_controller", "graph_path", "no_override"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--suggest_every", type=int, default=8)
    parser.add_argument("--min_goal_visits", type=int, default=2)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--graph_rollout_horizon", type=int, default=4)
    parser.add_argument("--token_source", type=str, default="scene_semantic", choices=["symbolic_hash", "scene_semantic", "scene_patch_hash"])
    parser.add_argument("--graph_update_mode", type=str, default="static", choices=["static", "adaptive"])
    parser.add_argument("--intent_mode", type=str, default="goal_region_v1", choices=["none", "goal_region_v1"])
    parser.add_argument("--intent_selection_mode", type=str, default="fixed", choices=["fixed", "state_v1"])
    parser.add_argument("--intent_type", type=str, default="find_goal_region", choices=["find_goal_region"])
    parser.add_argument("--semantic_retrieval_mode", type=str, default="concept_recall_v1", choices=["node_only", "concept_recall_v1", "concept_plan_v1"])
    parser.add_argument("--debug_trace", action="store_true")
    parser.add_argument("--record_demo", action="store_true")
    parser.add_argument("--demo_episode", type=int, default=1)
    parser.add_argument("--demo_fps", type=int, default=3)
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign", choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="tmp/songline_miniworld")
    parser.add_argument("--check_dependencies", action="store_true")
    parser.add_argument("--list_envs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_envs:
        print(json.dumps(MINIWORLD_ENV_ALIASES, indent=2, sort_keys=True))
        return
    if args.check_dependencies:
        runtime_ok, runtime_error = check_miniworld_runtime()
        print(
            json.dumps(
                {
                    "miniworld_available": True if _safe_dependency_check() else False,
                    "miniworld_runtime_usable": bool(runtime_ok),
                    "runtime_error": str(runtime_error),
                },
                indent=2,
            )
        )
        return
    _, summary = run_songline_miniworld_experiment(args, export_outputs=True, verbose=True)
    print("\nFinal summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


def _safe_dependency_check():
    try:
        ensure_miniworld_available()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
