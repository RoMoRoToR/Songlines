import argparse
import csv
import json
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid
import numpy as np
from minigrid.core.world_object import Goal

from songline_drive.graph_memory import DynamicSonglineGraph
from songline_drive.graph_rollout import GraphRolloutPlanner
from songline_drive.maneuver_selector import ManeuverSelector
from songline_drive.scene_encoder import MiniGridSceneEncoder
from songline_drive.scene_tokenizer import SceneTokenizer
from songline_drive.trajectory_planner import TrajectoryPlanner
from utils.lz_memory import SymbolicTokenizer


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def local_symbolic_observation(env, radius=1):
    """
    Строит компактное символическое наблюдение:
    - направление агента
    - локальное окно (2r+1)x(2r+1) вокруг агента
    - тип клетки кодируется коротким числом
    """
    unwrapped = env.unwrapped
    ax, ay = unwrapped.agent_pos
    direction = unwrapped.agent_dir
    grid = unwrapped.grid

    patch = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            x = ax + dx
            y = ay + dy
            if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
                row.append(99)
                continue
            cell = grid.get(x, y)
            if cell is None:
                row.append(0)
            else:
                cell_name = cell.type
                if cell_name == "wall":
                    row.append(1)
                elif cell_name == "goal":
                    row.append(2)
                elif cell_name == "lava":
                    row.append(3)
                elif cell_name == "door":
                    row.append(4)
                elif cell_name == "key":
                    row.append(5)
                elif cell_name == "ball":
                    row.append(6)
                elif cell_name == "box":
                    row.append(7)
                else:
                    row.append(8)
        patch.append(row)

    patch = np.array(patch, dtype=np.int32).reshape(-1)
    obs_vec = np.concatenate([np.array([direction], dtype=np.int32), patch])
    return obs_vec.astype(np.float32)


def scene_token_to_int(scene_token):
    token_name = str(scene_token.token_type)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(token_name))


def manhattan(a, b):
    return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))


def random_safe_action(rng):
    return int(rng.choice([0, 1, 2]))


def get_goal_position(env):
    grid = env.unwrapped.grid
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return np.array([x, y], dtype=np.int32)
    return None


def apply_goal_shift_v1(env, change_phase: int) -> bool:
    unwrapped = env.unwrapped
    grid = unwrapped.grid
    agent_xy = tuple(int(v) for v in unwrapped.agent_pos)
    current_goal = None
    candidates = []

    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                current_goal = (x, y)
            if (x, y) == agent_xy:
                continue
            if cell is None or (cell is not None and cell.type == "goal"):
                dist = abs(x - agent_xy[0]) + abs(y - agent_xy[1])
                candidates.append((dist, y, x))

    if len(candidates) < 2:
        return False

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    target_rank = min(len(candidates) - 1, int(change_phase % 2))
    tx = int(candidates[target_rank][2])
    ty = int(candidates[target_rank][1])
    target = (tx, ty)
    if current_goal == target:
        alt_rank = min(len(candidates) - 1, target_rank + 1)
        tx = int(candidates[alt_rank][2])
        ty = int(candidates[alt_rank][1])
        target = (tx, ty)

    if current_goal is not None and current_goal != target:
        grid.set(int(current_goal[0]), int(current_goal[1]), None)
    grid.set(tx, ty, Goal())
    return True


def apply_env_change(env, episode_idx: int, env_change_mode: str, change_after_episode: int) -> bool:
    if env_change_mode == "none":
        return False
    if int(change_after_episode) < 0 or int(episode_idx) < int(change_after_episode):
        return False
    if env_change_mode == "goal_shift_v1":
        return apply_goal_shift_v1(env, change_phase=int(episode_idx - change_after_episode))
    raise ValueError(f"Unknown env_change_mode: {env_change_mode}")


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def scene_risk_proxy(scene):
    if scene is None:
        return None
    risk = scene.risk_features
    return float(
        max(
            float(risk.get("collision_risk", 0.0)),
            float(risk.get("hazard_front", 0.0)),
            0.5 * float(risk.get("hazard_near", 0.0)),
            float(risk.get("lateral_hazard", 0.0)),
        )
    )


def scene_comfort_proxy(scene):
    if scene is None:
        return None
    alignment = float(scene.route_context.goal_alignment)
    return float(max(0.0, 1.0 - alignment))


def get_phrase_length_mean(memory):
    if memory is None or not memory.nodes:
        return 0.0
    lengths = [len(node["phrase"]) for node in memory.nodes.values()]
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def get_mean_xy(node, key_prefix):
    count = node.get(f"{key_prefix}_count", 0)
    if count <= 0:
        return None
    return node[f"{key_prefix}_sum"] / float(count)


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
        encoder = MiniGridSceneEncoder(radius=args.scene_radius)
        tokenizer = SceneTokenizer(mode="semantic")
    elif args.token_source == "scene_patch_hash":
        encoder = MiniGridSceneEncoder(radius=args.scene_radius)
        tokenizer = SceneTokenizer(mode="patch_hash")
    else:
        raise ValueError(f"Unknown token_source: {args.token_source}")
    memory = DynamicSonglineGraph(
        min_goal_visits=args.min_goal_visits,
        graph_update_mode=args.graph_update_mode,
    )
    planner = GraphRolloutPlanner()
    return tokenizer, encoder, memory, planner


def build_local_planner(args):
    return ManeuverSelector(), TrajectoryPlanner(commit_to_corridor=args.commit_to_corridor)


def build_episode_memory():
    return {
        "waypoints": {},
        "active_waypoint": None,
        "last_delta": None,
    }


def episodic_observe(ep_memory, pose_xy, distance_before, distance_after):
    if distance_before is None or distance_after is None:
        return
    improvement = float(distance_before - distance_after)
    key = (int(pose_xy[0]), int(pose_xy[1]))
    entry = ep_memory["waypoints"].get(key)
    if entry is None:
        ep_memory["waypoints"][key] = {
            "pos": np.asarray(pose_xy, dtype=np.int32),
            "score_sum": improvement,
            "score_count": 1,
            "last_improvement": improvement,
        }
    else:
        entry["score_sum"] += improvement
        entry["score_count"] += 1
        entry["last_improvement"] = improvement

    active = ep_memory["active_waypoint"]
    if active is not None and key == active["key"] and improvement <= 0:
        ep_memory["active_waypoint"] = None


def choose_episodic_waypoint(ep_memory):
    if not ep_memory["waypoints"]:
        return None

    scored = []
    for key, entry in ep_memory["waypoints"].items():
        mean_score = entry["score_sum"] / max(1, entry["score_count"])
        scored.append((mean_score, entry["score_count"], key))
    scored.sort(reverse=True)

    best_score, _, best_key = scored[0]
    if best_score <= 0:
        return None

    best_entry = ep_memory["waypoints"][best_key]
    ep_memory["active_waypoint"] = {
        "key": best_key,
        "pos": best_entry["pos"].copy(),
        "score": float(best_score),
    }
    return best_entry["pos"].copy()


def current_phrase_node(memory):
    if memory is None:
        return None
    return memory.current_phrase_id


def _cell_code(env, x, y):
    grid = env.unwrapped.grid
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return 99
    cell = grid.get(x, y)
    if cell is None:
        return 0
    if cell.type == "wall":
        return 1
    if cell.type == "goal":
        return 2
    if cell.type == "lava":
        return 3
    if cell.type == "door":
        return 4
    return 8


def _dir_to_delta(agent_dir):
    return {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1),
    }[int(agent_dir)]


def _hazard_aware_waypoint(env, current_token_label, plan_waypoint_xy, goal_xy):
    if current_token_label not in {"hazard_front", "gap_search"}:
        return None

    unwrapped = env.unwrapped
    ax, ay = [int(v) for v in unwrapped.agent_pos]
    agent_dir = int(unwrapped.agent_dir)
    target_xy = goal_xy if goal_xy is not None else plan_waypoint_xy
    if target_xy is None:
        return None

    candidates = []
    for dir_idx in range(4):
        dx, dy = _dir_to_delta(dir_idx)
        nx, ny = ax + dx, ay + dy
        code = _cell_code(env, nx, ny)
        if code not in (0, 2):
            continue

        left_dir = (dir_idx - 1) % 4
        right_dir = (dir_idx + 1) % 4
        ldx, ldy = _dir_to_delta(left_dir)
        rdx, rdy = _dir_to_delta(right_dir)
        left_code = _cell_code(env, nx + ldx, ny + ldy)
        right_code = _cell_code(env, nx + rdx, ny + rdy)
        lateral_hazard = float(left_code == 3 or right_code == 3)
        lateral_blocked = float(left_code in (1, 3, 99) or right_code in (1, 3, 99))
        goal_dx = int(target_xy[0]) - nx
        goal_dy = int(target_xy[1]) - ny
        norm = float(max(1e-6, (goal_dx * goal_dx + goal_dy * goal_dy) ** 0.5))
        heading_alignment = ((dx * goal_dx) + (dy * goal_dy)) / norm
        goal_progress = -manhattan(np.array([nx, ny]), np.asarray(target_xy))
        score = (2.5 * heading_alignment) + (1.5 * lateral_hazard) + (0.75 * lateral_blocked) + (0.1 * goal_progress)
        candidates.append((score, np.array([nx, ny], dtype=np.int32)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def select_graph_waypoint(memory, planner, selector, top_k, rollout_horizon, current_token_label=None, env=None, goal_xy=None):
    plans = planner.rollout(
        graph=memory,
        current_node_id=current_phrase_node(memory),
        horizon=rollout_horizon,
        top_k=top_k,
    )
    if not plans:
        return None

    plan = plans[0]
    command = selector.select(plan, current_token=current_token_label)
    if command.get("waypoint_xy") is None:
        return None

    waypoint_xy = np.asarray(command["waypoint_xy"], dtype=np.int32)
    hazard_waypoint = None
    if env is not None:
        hazard_waypoint = _hazard_aware_waypoint(
            env,
            current_token_label=current_token_label,
            plan_waypoint_xy=waypoint_xy,
            goal_xy=goal_xy,
        )
    if hazard_waypoint is not None:
        waypoint_xy = hazard_waypoint
        command["waypoint_xy"] = tuple(int(v) for v in waypoint_xy)

    return {
        "waypoint_xy": waypoint_xy,
        "target_node_id": command.get("target_node_id"),
        "next_node_id": int(plan.metadata["next_node_id"]),
        "graph_path_length": int(command.get("graph_path_length", plan.graph_path_length)),
        "utility": float(plan.utility),
        "token_sequence": list(plan.token_sequence),
        "maneuver_command": command,
    }


def make_method_name(agent_mode, songline_policy):
    if agent_mode == "songline":
        return f"songline_{songline_policy}"
    return agent_mode


def apply_milestone_mode(args):
    if getattr(args, "milestone_mode", "none") == "semantic_handoff_v1":
        args.agent_mode = "songline"
        args.songline_policy = "graph_path"
        args.token_source = "scene_semantic"
        args.early_hazard_intervention = True
    return args


def export_run_summary(out_dir, run_summary):
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    episodes = np.arange(1, len(run_summary["episode_returns"]) + 1)
    if len(episodes) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(episodes, run_summary["episode_returns"])
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "episode_returns.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(episodes, run_summary["episode_lengths"])
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Lengths")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "episode_lengths.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(episodes, run_summary["graph_nodes"])
    plt.plot(episodes, run_summary["graph_edges"])
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.title("Graph Growth by Episode")
    plt.legend(["nodes", "edges"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_growth_by_episode.png"))
    plt.close()


def export_debug_trace(out_dir, trace_rows):
    if not trace_rows:
        return
    trace_dir = os.path.join(out_dir, "traces")
    ensure_dir(trace_dir)

    json_path = os.path.join(trace_dir, "step_trace.json")
    with open(json_path, "w") as f:
        json.dump(trace_rows, f, indent=2)

    csv_path = os.path.join(trace_dir, "step_trace.csv")
    fieldnames = list(trace_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace_rows:
            writer.writerow(row)


def is_hazard_phase_token(token_label):
    return token_label in {"hazard_front", "gap_search", "gap_aligned", "safe_crossing"}


def should_run_graph_intervention(args, step, token_label, active_subgoal_key, hazard_phase_intervened):
    scheduled = (step % args.suggest_every == 0)
    if args.songline_policy != "graph_path":
        return scheduled
    if not args.early_hazard_intervention:
        return scheduled
    hazard_trigger = is_hazard_phase_token(token_label) and active_subgoal_key is None and not hazard_phase_intervened
    return scheduled or hazard_trigger


def should_activate_hazard_commit(scene, hazard_phase_active, subgoal_xy):
    if not hazard_phase_active or scene is None or subgoal_xy is None:
        return False
    risk = scene.risk_features
    front_safe = int(risk.get("front_safe", 0.0)) == 1
    goal_alignment = float(risk.get("goal_heading_alignment", 0.0))
    return bool(front_safe and goal_alignment >= 0.35)


def summarise_run(run_summary, memory):
    final_nodes = 0 if memory is None else int(len(memory.nodes))
    final_edges = 0 if memory is None else int(sum(len(v) for v in memory.edges.values()))
    intervention_rate = 0.0
    plan_hit_rate = 0.0
    phrase_length_mean = 0.0

    if memory is not None:
        intervention_rate = safe_rate(memory.interventions, memory.intervention_attempts)
        plan_hit_rate = safe_rate(memory.plan_hits, memory.plan_total)
        phrase_length_mean = get_phrase_length_mean(memory)

    summary = {
        "success_rate": float(np.mean(run_summary["successes"])) if run_summary["successes"] else 0.0,
        "success_rate_pre_change": float(np.mean(run_summary["successes_pre_change"])) if run_summary["successes_pre_change"] else 0.0,
        "success_rate_post_change": float(np.mean(run_summary["successes_post_change"])) if run_summary["successes_post_change"] else 0.0,
        "avg_return": float(np.mean(run_summary["episode_returns"])) if run_summary["episode_returns"] else 0.0,
        "avg_steps_to_goal": float(np.mean(run_summary["episode_lengths"])) if run_summary["episode_lengths"] else 0.0,
        "avg_steps": float(np.mean(run_summary["episode_lengths"])) if run_summary["episode_lengths"] else 0.0,
        "intervention_rate": intervention_rate,
        "plan_hit_rate": plan_hit_rate,
        "graph_nodes": final_nodes,
        "graph_edges": final_edges,
        "phrase_length_mean": phrase_length_mean,
        "subgoal_reach_rate": safe_rate(
            float(np.sum(run_summary["subgoal_reached"])),
            len(run_summary["subgoal_reached"]),
        ),
        "goal_distance_delta_per_intervention": safe_rate(
            float(np.sum(run_summary["goal_distance_delta_per_intervention"])),
            max(1, len(run_summary["goal_distance_delta_per_intervention"])),
        ),
        "node_reuse_rate": safe_rate(
            float(np.sum(run_summary["node_reuse_rate"])),
            len(run_summary["node_reuse_rate"]),
        ),
        "new_nodes_per_episode": float(np.mean(run_summary["new_nodes_per_episode"])) if run_summary["new_nodes_per_episode"] else 0.0,
        "graph_path_length": float(np.mean(run_summary["graph_path_length"])) if run_summary["graph_path_length"] else 0.0,
        "fraction_gap_aligned": float(np.mean(run_summary["has_gap_aligned"])) if run_summary["has_gap_aligned"] else 0.0,
        "fraction_safe_crossing": float(np.mean(run_summary["has_safe_crossing"])) if run_summary["has_safe_crossing"] else 0.0,
        "fraction_post_hazard": float(np.mean(run_summary["has_post_hazard"])) if run_summary["has_post_hazard"] else 0.0,
        "fraction_final_exit_maneuver": float(np.mean(run_summary["has_final_exit_maneuver"])) if run_summary["has_final_exit_maneuver"] else 0.0,
        "fraction_resume_to_goal": float(np.mean(run_summary["has_resume_to_goal"])) if run_summary["has_resume_to_goal"] else 0.0,
        "fraction_post_hazard_progress": float(np.mean(run_summary["has_post_hazard_progress"])) if run_summary["has_post_hazard_progress"] else 0.0,
        "fraction_resume_to_goal_progress": float(np.mean(run_summary["has_resume_to_goal_progress"])) if run_summary["has_resume_to_goal_progress"] else 0.0,
        "fraction_post_hazard_to_success": float(np.mean(run_summary["post_hazard_to_success"])) if run_summary["post_hazard_to_success"] else 0.0,
        "fraction_resume_to_goal_to_success": float(np.mean(run_summary["resume_to_goal_to_success"])) if run_summary["resume_to_goal_to_success"] else 0.0,
        "conditional_post_hazard_success": safe_rate(
            float(np.sum(run_summary["post_hazard_to_success"])),
            float(np.sum(run_summary["has_post_hazard"])),
        ),
        "conditional_resume_to_goal_success": safe_rate(
            float(np.sum(run_summary["resume_to_goal_to_success"])),
            float(np.sum(run_summary["has_resume_to_goal"])),
        ),
        "mean_max_phase_depth": float(np.mean(run_summary["max_phase_depth"])) if run_summary["max_phase_depth"] else 0.0,
        "final_nodes": final_nodes,
        "final_edges": final_edges,
    }
    return summary


def run_songline_experiment(args, export_outputs=True, verbose=True):
    args = apply_milestone_mode(args)
    ensure_dir(args.out_dir)

    env = gym.make(args.env_id, render_mode="rgb_array")
    rng = np.random.RandomState(args.seed)
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
        "env_id": args.env_id,
        "agent_mode": args.agent_mode,
        "songline_policy": args.songline_policy,
        "method": method_name,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "env_change_mode": args.env_change_mode,
        "change_after_episode": args.change_after_episode,
        "episode_returns": [],
        "episode_lengths": [],
        "successes": [],
        "successes_pre_change": [],
        "successes_post_change": [],
        "graph_nodes": [],
        "graph_edges": [],
        "intervention_rate": [],
        "plan_hit_rate": [],
        "phrase_length_mean": [],
        "subgoal_reached": [],
        "goal_distance_delta_per_intervention": [],
        "node_reuse_rate": [],
        "new_nodes_per_episode": [],
        "graph_path_length": [],
        "has_gap_aligned": [],
        "has_safe_crossing": [],
        "has_post_hazard": [],
        "has_final_exit_maneuver": [],
        "has_resume_to_goal": [],
        "has_post_hazard_progress": [],
        "has_resume_to_goal_progress": [],
        "post_hazard_to_success": [],
        "resume_to_goal_to_success": [],
        "max_phase_depth": [],
        "episode_metrics": [],
    }

    total_step_idx = 0
    seen_songline_nodes = set()
    debug_trace_rows = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        del obs, info
        change_active = apply_env_change(
            env,
            episode_idx=ep,
            env_change_mode=args.env_change_mode,
            change_after_episode=args.change_after_episode,
        )
        if tokenizer is not None and hasattr(tokenizer, "reset"):
            tokenizer.reset()
        if trajectory_planner is not None and hasattr(trajectory_planner, "reset"):
            trajectory_planner.reset()

        goal_xy = get_goal_position(env)
        episode_return = 0.0
        success = 0
        previous_goal_distance = None
        subgoal_xy = None
        step_count = 0
        interventions_this_episode = 0
        subgoal_hits_this_episode = 0
        distance_delta_sum = 0.0
        active_intervention_distance = None
        active_subgoal_key = None
        active_graph_path_length = 0
        graph_path_lengths = []
        unique_songline_nodes_this_episode = set()
        new_nodes_start = 0 if memory is None else len(memory.nodes)
        ep_memory = build_episode_memory() if args.agent_mode == "greedy_episodic" else None
        no_override_suggestions = 0
        previous_token_label = None
        active_target_node_id = None
        active_next_node_id = None
        hazard_phase_active = False
        hazard_phase_intervened = False
        active_maneuver_command = None
        active_maneuver_command_type = None
        hazard_commit_active = False
        hazard_commit_dir = None
        hazard_commit_steps_left = 0
        exit_hazard_commit_active = False
        exit_hazard_commit_dir = None
        exit_hazard_commit_steps_left = 0
        final_exit_active = False
        final_exit_dir = None
        final_exit_steps_left = 0
        resume_to_goal_active = False
        resume_to_goal_steps_left = 0
        phase_flags = {
            "gap_aligned": 0,
            "safe_crossing": 0,
            "post_hazard": 0,
            "final_exit_maneuver": 0,
            "resume_to_goal": 0,
        }
        post_hazard_armed = 0
        resume_to_goal_armed = 0
        has_post_hazard_progress = 0
        has_resume_to_goal_progress = 0

        for step in range(args.max_steps):
            step_count = step + 1
            unwrapped = env.unwrapped
            agent_xy = np.array(unwrapped.agent_pos, dtype=np.int32)
            distance_before = None if goal_xy is None else manhattan(agent_xy, goal_xy)
            scene = None
            scene_token = None
            token_label = None

            if args.agent_mode == "songline":
                prev_graph_node_id = memory.current_phrase_id
                if args.token_source == "symbolic_hash":
                    obs_vec = local_symbolic_observation(env, radius=args.scene_radius)
                    token = tokenizer.encode(obs_vec)
                    token_label = str(token)
                else:
                    scene = scene_encoder.encode(env)
                    scene_token = tokenizer.tokenize(scene)
                    token = scene_token_to_int(scene_token)
                    token_label = str(scene_token.token_type)
                is_new_node = memory.update_token(token, total_step_idx)
                current_graph_node_id = memory.current_phrase_id
                if is_new_node and memory.current_phrase_id is not None:
                    unique_songline_nodes_this_episode.add(memory.current_phrase_id)
                    if memory.current_phrase_id in seen_songline_nodes:
                        pass
                    else:
                        seen_songline_nodes.add(memory.current_phrase_id)
                memory.observe(
                    total_step_idx,
                    pose_xy=agent_xy,
                    goal_xy=goal_xy,
                    reward=0.0,
                    risk=scene_risk_proxy(scene) if args.token_source != "symbolic_hash" else None,
                    comfort_cost=scene_comfort_proxy(scene) if args.token_source != "symbolic_hash" else None,
                    goal_alignment=scene.route_context.goal_alignment if args.token_source != "symbolic_hash" else None,
                    phase_label=token_label,
                )

                hazard_phase_active = is_hazard_phase_token(token_label)
                post_exit_pending = bool(
                    (final_exit_active and final_exit_steps_left > 0)
                    or (resume_to_goal_active and resume_to_goal_steps_left > 0)
                )
                if not hazard_phase_active and not post_exit_pending:
                    hazard_phase_intervened = False
                    hazard_commit_active = False
                    hazard_commit_dir = None
                    hazard_commit_steps_left = 0
                    exit_hazard_commit_active = False
                    exit_hazard_commit_dir = None
                    exit_hazard_commit_steps_left = 0
                    final_exit_active = False
                    final_exit_dir = None
                    final_exit_steps_left = 0

                if token_label == "safe_crossing":
                    exit_hazard_commit_active = True
                    exit_hazard_commit_dir = int(env.unwrapped.agent_dir)
                    exit_hazard_commit_steps_left = max(exit_hazard_commit_steps_left, 2)
                elif token_label == "post_hazard":
                    final_exit_active = True
                    final_exit_dir = int(env.unwrapped.agent_dir)
                    final_exit_steps_left = max(final_exit_steps_left, 2)

                if should_run_graph_intervention(
                    args,
                    step=step,
                    token_label=token_label,
                    active_subgoal_key=active_subgoal_key,
                    hazard_phase_intervened=hazard_phase_intervened,
                ):
                    proposal = memory.suggest_subgoal(top_k=args.top_k_goals)
                    if proposal is not None:
                        no_override_suggestions += 1

                    if args.songline_policy == "subgoal_controller" and proposal is not None:
                        gx = int(round(proposal["goal_xy"][0]))
                        gy = int(round(proposal["goal_xy"][1]))
                        gx = max(0, min(gx, env.unwrapped.grid.width - 1))
                        gy = max(0, min(gy, env.unwrapped.grid.height - 1))
                        subgoal_xy = np.array([gx, gy], dtype=np.int32)
                        interventions_this_episode += 1
                        active_intervention_distance = distance_before
                        active_subgoal_key = (int(subgoal_xy[0]), int(subgoal_xy[1]))
                        active_graph_path_length = 0
                        active_target_node_id = int(proposal["node_id"])
                        active_next_node_id = int(proposal["node_id"])
                        if hazard_phase_active:
                            hazard_phase_intervened = True

                    if args.songline_policy == "graph_path":
                        preserve_commit = bool(
                            (hazard_commit_active and hazard_commit_steps_left > 0 and hazard_phase_active)
                            or (exit_hazard_commit_active and exit_hazard_commit_steps_left > 0)
                            or (final_exit_active and final_exit_steps_left > 0)
                            or (resume_to_goal_active and resume_to_goal_steps_left > 0)
                        )
                        if not preserve_commit:
                            path_plan = select_graph_waypoint(
                                memory,
                                rollout_planner,
                                maneuver_selector,
                                top_k=args.top_k_goals,
                                rollout_horizon=args.graph_rollout_horizon,
                                current_token_label=token_label,
                                env=env,
                                goal_xy=goal_xy,
                            )
                            if path_plan is not None:
                                subgoal_xy = path_plan["waypoint_xy"].copy()
                                interventions_this_episode += 1
                                active_intervention_distance = distance_before
                                active_subgoal_key = (int(subgoal_xy[0]), int(subgoal_xy[1]))
                                active_graph_path_length = int(path_plan["graph_path_length"])
                                graph_path_lengths.append(active_graph_path_length)
                                active_target_node_id = None if path_plan["target_node_id"] is None else int(path_plan["target_node_id"])
                                active_next_node_id = int(path_plan["next_node_id"])
                                active_maneuver_command = dict(path_plan["maneuver_command"])
                                active_maneuver_command_type = str(active_maneuver_command.get("command_type"))
                                if hazard_phase_active:
                                    hazard_phase_intervened = True

                if (
                    args.commit_to_corridor
                    and args.agent_mode == "songline"
                    and not hazard_commit_active
                    and should_activate_hazard_commit(
                        scene=scene,
                        hazard_phase_active=hazard_phase_active,
                        subgoal_xy=subgoal_xy,
                    )
                ):
                    hazard_commit_active = True
                    hazard_commit_dir = int(env.unwrapped.agent_dir)
                    hazard_commit_steps_left = max(hazard_commit_steps_left, 3)

            action = random_safe_action(rng)
            if args.agent_mode == "greedy":
                command = {"command_type": "go_to_waypoint", "waypoint_xy": goal_xy}
                action = trajectory_planner.next_action(env, command) if goal_xy is not None else random_safe_action(rng)
            elif args.agent_mode == "greedy_episodic":
                episodic_target = None
                if step % args.suggest_every == 0:
                    episodic_target = choose_episodic_waypoint(ep_memory)
                    if episodic_target is not None:
                        interventions_this_episode += 1
                        active_intervention_distance = distance_before
                        active_subgoal_key = (int(episodic_target[0]), int(episodic_target[1]))
                if episodic_target is None and ep_memory["active_waypoint"] is not None:
                    episodic_target = ep_memory["active_waypoint"]["pos"]
                if episodic_target is not None and rng.rand() > args.epsilon:
                    subgoal_xy = episodic_target.copy()
                    action = trajectory_planner.next_action(
                        env,
                        {"command_type": "go_to_waypoint", "waypoint_xy": subgoal_xy},
                    )
                else:
                    action = trajectory_planner.next_action(
                        env,
                        {"command_type": "go_to_waypoint", "waypoint_xy": goal_xy},
                    ) if goal_xy is not None else random_safe_action(rng)
            elif args.agent_mode == "songline":
                if args.songline_policy == "no_override":
                    action = trajectory_planner.next_action(
                        env,
                        {"command_type": "go_to_waypoint", "waypoint_xy": goal_xy},
                    ) if goal_xy is not None else random_safe_action(rng)
                elif args.songline_policy == "subgoal_controller":
                    if subgoal_xy is not None and rng.rand() > args.epsilon:
                        action = trajectory_planner.next_action(
                            env,
                            {"command_type": "go_to_waypoint", "waypoint_xy": subgoal_xy},
                        )
                    elif goal_xy is not None and rng.rand() > 0.5:
                        action = trajectory_planner.next_action(
                            env,
                            {"command_type": "go_to_waypoint", "waypoint_xy": goal_xy},
                        )
                    else:
                        action = random_safe_action(rng)
                elif args.songline_policy == "graph_path":
                    if resume_to_goal_active and resume_to_goal_steps_left > 0:
                        command = {
                            "command_type": "resume_to_goal",
                            "waypoint_xy": goal_xy if goal_xy is not None else subgoal_xy,
                            "final_exit_mode": args.final_exit_mode,
                        }
                        active_maneuver_command_type = str(command.get("command_type"))
                        action = trajectory_planner.next_action(env, command)
                    elif final_exit_active and final_exit_steps_left > 0:
                        command = {
                            "command_type": "final_exit_maneuver",
                            "waypoint_xy": goal_xy if goal_xy is not None else subgoal_xy,
                            "hazard_commit_dir": final_exit_dir,
                            "final_exit_mode": args.final_exit_mode,
                        }
                        active_maneuver_command_type = str(command.get("command_type"))
                        action = trajectory_planner.next_action(env, command)
                    elif exit_hazard_commit_active and exit_hazard_commit_steps_left > 0:
                        command = {
                            "command_type": "exit_hazard_commit",
                            "waypoint_xy": goal_xy if goal_xy is not None else subgoal_xy,
                            "hazard_commit_dir": exit_hazard_commit_dir,
                        }
                        active_maneuver_command_type = str(command.get("command_type"))
                        action = trajectory_planner.next_action(env, command)
                    elif subgoal_xy is not None and rng.rand() > args.epsilon:
                        if hazard_commit_active and hazard_commit_steps_left > 0:
                            command = {
                                "command_type": "committed_corridor_cross",
                                "waypoint_xy": subgoal_xy,
                                "hazard_commit_dir": hazard_commit_dir,
                            }
                        else:
                            command = active_maneuver_command or {"command_type": "follow_graph_path", "waypoint_xy": subgoal_xy}
                        active_maneuver_command_type = str(command.get("command_type"))
                        action = trajectory_planner.next_action(env, command)
                    elif goal_xy is not None:
                        active_maneuver_command_type = "go_to_waypoint"
                        action = trajectory_planner.next_action(
                            env,
                            {"command_type": "go_to_waypoint", "waypoint_xy": goal_xy},
                        )
                    else:
                        active_maneuver_command_type = "random"
                        action = random_safe_action(rng)
                else:
                    raise ValueError(f"Unknown songline policy: {args.songline_policy}")

            if token_label == "gap_aligned":
                phase_flags["gap_aligned"] = 1
            elif token_label == "safe_crossing":
                phase_flags["safe_crossing"] = 1
            elif token_label == "post_hazard":
                phase_flags["post_hazard"] = 1
                post_hazard_armed = 1
            if active_maneuver_command_type == "final_exit_maneuver":
                phase_flags["final_exit_maneuver"] = 1
            elif active_maneuver_command_type == "resume_to_goal":
                phase_flags["resume_to_goal"] = 1
                resume_to_goal_armed = 1

            obs, reward, terminated, truncated, info = env.step(action)
            del obs, info

            total_step_idx += 1
            episode_return += float(reward)
            agent_xy_new = np.array(env.unwrapped.agent_pos, dtype=np.int32)
            distance_after = None if goal_xy is None else manhattan(agent_xy_new, goal_xy)
            delta_distance = None
            if distance_before is not None and distance_after is not None:
                delta_distance = float(distance_before - distance_after)
                if post_hazard_armed and delta_distance > 0.0:
                    has_post_hazard_progress = 1
                    post_hazard_armed = 0
                if resume_to_goal_armed and delta_distance > 0.0:
                    has_resume_to_goal_progress = 1
                    resume_to_goal_armed = 0

            if args.agent_mode == "songline":
                if previous_goal_distance is not None and distance_after is not None:
                    improved = distance_after < previous_goal_distance
                    memory.record_plan_outcome(improved)

                transition_success = None
                transition_risk = None
                transition_cost = None
                if delta_distance is not None:
                    transition_success = 1.0 if delta_distance > 0.0 else 0.0
                    transition_cost = 1.0
                if args.token_source != "symbolic_hash":
                    post_scene = scene_encoder.encode(env)
                    transition_risk = scene_risk_proxy(post_scene)
                else:
                    post_scene = None

                memory.observe(
                    total_step_idx,
                    pose_xy=agent_xy_new,
                    goal_xy=goal_xy,
                    reward=float(reward),
                    progress=delta_distance,
                    risk=transition_risk,
                    success=1.0 if float(reward) > 0.0 else 0.0,
                    comfort_cost=scene_comfort_proxy(post_scene) if post_scene is not None else None,
                    goal_alignment=post_scene.route_context.goal_alignment if post_scene is not None else None,
                    phase_label=token_label,
                )
                memory.observe_transition(
                    src=prev_graph_node_id,
                    dst=current_graph_node_id,
                    step_idx=total_step_idx,
                    transition_success=transition_success,
                    transition_risk=transition_risk,
                    transition_cost=transition_cost,
                )

            if hazard_commit_active:
                still_hazard = hazard_phase_active
                front_safe_now = False
                if args.agent_mode == "songline" and scene_encoder is not None:
                    if post_scene is None:
                        post_scene = scene_encoder.encode(env)
                    post_risk = post_scene.risk_features
                    still_hazard = bool(
                        int(post_risk.get("hazard_front", 0.0)) == 1
                        or int(post_risk.get("hazard_near", 0.0)) == 1
                    )
                    front_safe_now = int(post_scene.risk_features.get("front_safe", 0.0)) == 1
                if still_hazard and front_safe_now and hazard_commit_steps_left > 0:
                    hazard_commit_steps_left -= 1
                else:
                    hazard_commit_active = False
                    hazard_commit_dir = None
                    hazard_commit_steps_left = 0
                if hazard_commit_steps_left <= 0:
                    hazard_commit_active = False
                    hazard_commit_dir = None

            if exit_hazard_commit_active:
                still_hazard = hazard_phase_active
                if args.agent_mode == "songline" and scene_encoder is not None:
                    post_scene = scene_encoder.encode(env)
                    post_risk = post_scene.risk_features
                    still_hazard = bool(
                        int(post_risk.get("hazard_front", 0.0)) == 1
                        or int(post_risk.get("hazard_near", 0.0)) == 1
                    )
                if still_hazard and exit_hazard_commit_steps_left > 0:
                    exit_hazard_commit_steps_left -= 1
                else:
                    exit_hazard_commit_active = False
                    exit_hazard_commit_dir = None
                    exit_hazard_commit_steps_left = 0
                if exit_hazard_commit_steps_left <= 0:
                    exit_hazard_commit_active = False
                    exit_hazard_commit_dir = None

            if final_exit_active:
                maintained_progress = True
                if distance_before is not None and distance_after is not None:
                    maintained_progress = distance_after <= distance_before
                if maintained_progress and final_exit_steps_left > 0:
                    final_exit_steps_left -= 1
                else:
                    final_exit_active = False
                    final_exit_dir = None
                    final_exit_steps_left = 0
                if final_exit_steps_left <= 0:
                    if maintained_progress:
                        resume_to_goal_active = True
                        # Give the bridge one full future action-selection step to fire.
                        resume_to_goal_steps_left = max(resume_to_goal_steps_left, 2)
                    final_exit_active = False
                    final_exit_dir = None

            if resume_to_goal_active:
                if resume_to_goal_steps_left > 0:
                    resume_to_goal_steps_left -= 1
                if resume_to_goal_steps_left <= 0:
                    resume_to_goal_active = False

            if args.agent_mode == "greedy_episodic":
                episodic_observe(ep_memory, agent_xy_new, distance_before, distance_after)

            if active_subgoal_key is not None:
                if int(agent_xy_new[0]) == active_subgoal_key[0] and int(agent_xy_new[1]) == active_subgoal_key[1]:
                    subgoal_hits_this_episode += 1
                    if active_intervention_distance is not None and distance_after is not None:
                        distance_delta_sum += float(active_intervention_distance - distance_after)
                    active_subgoal_key = None
                    active_intervention_distance = None
                    active_target_node_id = None
                    active_next_node_id = None
                    active_maneuver_command = None
                    active_maneuver_command_type = None
                    hazard_commit_active = False
                    hazard_commit_dir = None
                    hazard_commit_steps_left = 0
                    exit_hazard_commit_active = False
                    exit_hazard_commit_dir = None
                    exit_hazard_commit_steps_left = 0
                    final_exit_active = False
                    final_exit_dir = None
                    final_exit_steps_left = 0
                    resume_to_goal_active = False
                    resume_to_goal_steps_left = 0
                    if args.agent_mode == "greedy_episodic" and ep_memory["active_waypoint"] is not None:
                        ep_memory["active_waypoint"] = None
                elif active_intervention_distance is not None and distance_after is not None:
                    if distance_after > active_intervention_distance + args.intervention_patience:
                        active_subgoal_key = None
                        active_intervention_distance = None
                        active_target_node_id = None
                        active_next_node_id = None
                        active_maneuver_command = None
                        active_maneuver_command_type = None
                        hazard_commit_active = False
                        hazard_commit_dir = None
                        hazard_commit_steps_left = 0
                        exit_hazard_commit_active = False
                        exit_hazard_commit_dir = None
                        exit_hazard_commit_steps_left = 0
                        final_exit_active = False
                        final_exit_dir = None
                        final_exit_steps_left = 0
                        resume_to_goal_active = False
                        resume_to_goal_steps_left = 0
                        if args.agent_mode == "greedy_episodic" and ep_memory["active_waypoint"] is not None:
                            ep_memory["active_waypoint"] = None

            if args.debug_trace and args.agent_mode == "songline":
                trace_enabled = args.debug_trace_env_filter in ("", args.env_id)
                if trace_enabled:
                    risk = {} if scene is None else scene.risk_features
                    planner_debug = trajectory_planner.debug_state() if hasattr(trajectory_planner, "debug_state") else {
                        "corridor_commit_active": 0,
                        "corridor_commit_dir": None,
                        "corridor_commit_steps_left": 0,
                    }
                    debug_trace_rows.append(
                        {
                            "episode": int(ep + 1),
                            "step": int(step + 1),
                            "token": token_label,
                            "prev_token": previous_token_label,
                            "action": int(action),
                            "pos_x": int(agent_xy[0]),
                            "pos_y": int(agent_xy[1]),
                            "next_pos_x": int(agent_xy_new[0]),
                            "next_pos_y": int(agent_xy_new[1]),
                            "front_cell": int(risk.get("front_cell", -1)),
                            "left_cell": int(risk.get("left_cell", -1)),
                            "right_cell": int(risk.get("right_cell", -1)),
                            "back_cell": int(risk.get("back_cell", -1)),
                            "distance_before": -1 if distance_before is None else int(distance_before),
                            "distance_after": -1 if distance_after is None else int(distance_after),
                            "delta_distance": 0.0 if delta_distance is None else float(delta_distance),
                            "subgoal_x": None if subgoal_xy is None else int(subgoal_xy[0]),
                            "subgoal_y": None if subgoal_xy is None else int(subgoal_xy[1]),
                            "intervention_active": int(active_subgoal_key is not None),
                            "reward": float(reward),
                            "graph_node_id": None if memory.current_phrase_id is None else int(memory.current_phrase_id),
                            "target_node_id": active_target_node_id,
                            "next_graph_node_id": active_next_node_id,
                            "maneuver_command_type": active_maneuver_command_type,
                            "goal_visible": int(risk.get("goal_visible", 0.0)),
                            "hazard_front": int(risk.get("hazard_front", 0.0)),
                            "hazard_near": int(risk.get("hazard_near", 0.0)),
                            "front_safe": int(risk.get("front_safe", 0.0)),
                            "narrow_safe_channel": int(risk.get("narrow_safe_channel", 0.0)),
                            "goal_heading_alignment": float(risk.get("goal_heading_alignment", 0.0)),
                            "hazard_phase_active": int(hazard_phase_active),
                            "corridor_commit_active": int(hazard_commit_active),
                            "corridor_commit_dir": hazard_commit_dir,
                            "corridor_commit_steps_left": int(hazard_commit_steps_left),
                            "exit_hazard_commit_active": int(exit_hazard_commit_active),
                            "exit_hazard_commit_dir": exit_hazard_commit_dir,
                            "exit_hazard_commit_steps_left": int(exit_hazard_commit_steps_left),
                            "final_exit_active": int(final_exit_active),
                            "final_exit_dir": final_exit_dir,
                            "final_exit_steps_left": int(final_exit_steps_left),
                            "resume_to_goal_active": int(resume_to_goal_active),
                            "resume_to_goal_steps_left": int(resume_to_goal_steps_left),
                        }
                    )
                    previous_token_label = token_label

            if distance_after is not None:
                previous_goal_distance = distance_after

            if reward > 0:
                success = 1
                if goal_xy is not None:
                    subgoal_xy = goal_xy.copy()

            if terminated or truncated:
                break

        edge_count = 0 if memory is None else sum(len(v) for v in memory.edges.values())
        intervention_rate = 0.0
        if args.agent_mode == "songline":
            intervention_rate = safe_rate(memory.interventions, memory.intervention_attempts)
            if args.songline_policy == "no_override":
                intervention_rate = safe_rate(no_override_suggestions, max(1, step_count))
        elif args.agent_mode == "greedy_episodic":
            intervention_rate = safe_rate(interventions_this_episode, max(1, step_count))

        plan_hit_rate = 0.0 if memory is None else safe_rate(memory.plan_hits, memory.plan_total)
        phrase_length_mean = get_phrase_length_mean(memory)
        node_reuse_rate = 0.0
        if args.agent_mode == "songline" and memory is not None:
            reused = sum(1 for node in memory.nodes.values() if node["visits"] > 1)
            node_reuse_rate = safe_rate(reused, len(memory.nodes))

        new_nodes = 0 if memory is None else len(memory.nodes) - new_nodes_start
        graph_path_length_mean = float(np.mean(graph_path_lengths)) if graph_path_lengths else 0.0
        subgoal_reach_rate = safe_rate(subgoal_hits_this_episode, interventions_this_episode)
        distance_delta_per_intervention = safe_rate(distance_delta_sum, interventions_this_episode)
        max_phase_depth = 0
        if phase_flags["gap_aligned"]:
            max_phase_depth = 1
        if phase_flags["safe_crossing"]:
            max_phase_depth = 2
        if phase_flags["post_hazard"]:
            max_phase_depth = 3
        if phase_flags["final_exit_maneuver"]:
            max_phase_depth = 4
        if phase_flags["resume_to_goal"]:
            max_phase_depth = 5

        episode_metrics = {
            "episode": ep + 1,
            "env_id": args.env_id,
            "change_active": int(change_active),
            "agent_mode": args.agent_mode,
            "songline_policy": args.songline_policy,
            "method": method_name,
            "seed": args.seed,
            "return": float(episode_return),
            "steps_to_goal": int(step_count),
            "steps": int(step_count),
            "success": int(success),
            "intervention_rate": float(intervention_rate),
            "plan_hit_rate": float(plan_hit_rate),
            "graph_nodes": 0 if memory is None else int(len(memory.nodes)),
            "graph_edges": int(edge_count),
            "phrase_length_mean": float(phrase_length_mean),
            "subgoal_reach_rate": float(subgoal_reach_rate),
            "goal_distance_delta_per_intervention": float(distance_delta_per_intervention),
            "node_reuse_rate": float(node_reuse_rate),
            "new_nodes_per_episode": int(new_nodes),
            "graph_path_length": float(graph_path_length_mean),
            "has_gap_aligned": int(phase_flags["gap_aligned"]),
            "has_safe_crossing": int(phase_flags["safe_crossing"]),
            "has_post_hazard": int(phase_flags["post_hazard"]),
            "has_final_exit_maneuver": int(phase_flags["final_exit_maneuver"]),
            "has_resume_to_goal": int(phase_flags["resume_to_goal"]),
            "has_post_hazard_progress": int(has_post_hazard_progress),
            "has_resume_to_goal_progress": int(has_resume_to_goal_progress),
            "post_hazard_to_success": int(phase_flags["post_hazard"] and success),
            "resume_to_goal_to_success": int(phase_flags["resume_to_goal"] and success),
            "max_phase_depth": int(max_phase_depth),
        }

        run_summary["episode_returns"].append(float(episode_return))
        run_summary["episode_lengths"].append(int(step_count))
        run_summary["successes"].append(int(success))
        if change_active:
            run_summary["successes_post_change"].append(int(success))
        else:
            run_summary["successes_pre_change"].append(int(success))
        run_summary["graph_nodes"].append(episode_metrics["graph_nodes"])
        run_summary["graph_edges"].append(episode_metrics["graph_edges"])
        run_summary["intervention_rate"].append(float(intervention_rate))
        run_summary["plan_hit_rate"].append(float(plan_hit_rate))
        run_summary["phrase_length_mean"].append(float(phrase_length_mean))
        run_summary["subgoal_reached"].append(float(subgoal_reach_rate))
        run_summary["goal_distance_delta_per_intervention"].append(float(distance_delta_per_intervention))
        run_summary["node_reuse_rate"].append(float(node_reuse_rate))
        run_summary["new_nodes_per_episode"].append(int(new_nodes))
        run_summary["graph_path_length"].append(float(graph_path_length_mean))
        run_summary["has_gap_aligned"].append(int(phase_flags["gap_aligned"]))
        run_summary["has_safe_crossing"].append(int(phase_flags["safe_crossing"]))
        run_summary["has_post_hazard"].append(int(phase_flags["post_hazard"]))
        run_summary["has_final_exit_maneuver"].append(int(phase_flags["final_exit_maneuver"]))
        run_summary["has_resume_to_goal"].append(int(phase_flags["resume_to_goal"]))
        run_summary["has_post_hazard_progress"].append(int(has_post_hazard_progress))
        run_summary["has_resume_to_goal_progress"].append(int(has_resume_to_goal_progress))
        run_summary["post_hazard_to_success"].append(int(phase_flags["post_hazard"] and success))
        run_summary["resume_to_goal_to_success"].append(int(phase_flags["resume_to_goal"] and success))
        run_summary["max_phase_depth"].append(int(max_phase_depth))
        run_summary["episode_metrics"].append(episode_metrics)

        if verbose:
            print(
                f"Episode {ep + 1:03d} | "
                f"method={method_name} | "
                f"return={episode_return:.3f} | "
                f"steps={step_count:03d} | "
                f"success={success} | "
                f"nodes={episode_metrics['graph_nodes']} | "
                f"edges={episode_metrics['graph_edges']} | "
                f"subgoal_hit={subgoal_reach_rate:.3f}"
            )

    summary = summarise_run(run_summary, memory)
    summary.update(
        {
            "env_id": args.env_id,
            "agent_mode": args.agent_mode,
            "songline_policy": args.songline_policy,
            "method": method_name,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
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

    env.close()
    return run_summary, summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-Random-6x6-v0")
    parser.add_argument(
        "--agent_mode",
        type=str,
        default="songline",
        choices=["random", "greedy", "greedy_episodic", "songline"],
    )
    parser.add_argument(
        "--songline_policy",
        type=str,
        default="subgoal_controller",
        choices=["no_override", "subgoal_controller", "graph_path"],
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--suggest_every", type=int, default=8)
    parser.add_argument("--intervention_patience", type=int, default=4)
    parser.add_argument("--min_goal_visits", type=int, default=2)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--graph_rollout_horizon", type=int, default=4)
    parser.add_argument(
        "--token_source",
        type=str,
        default="symbolic_hash",
        choices=["symbolic_hash", "scene_semantic", "scene_patch_hash"],
    )
    parser.add_argument(
        "--milestone_mode",
        type=str,
        default="none",
        choices=["none", "semantic_handoff_v1"],
    )
    parser.add_argument(
        "--final_exit_mode",
        type=str,
        default="none",
        choices=["none", "v1"],
    )
    parser.add_argument(
        "--graph_update_mode",
        type=str,
        default="static",
        choices=["static", "adaptive"],
    )
    parser.add_argument(
        "--env_change_mode",
        type=str,
        default="none",
        choices=["none", "goal_shift_v1"],
    )
    parser.add_argument("--change_after_episode", type=int, default=-1)
    parser.add_argument("--scene_radius", type=int, default=1)
    parser.add_argument("--export_phase_metrics", action="store_true")
    parser.add_argument("--early_hazard_intervention", action="store_true")
    parser.add_argument("--commit_to_corridor", action="store_true")
    parser.add_argument("--debug_trace", action="store_true")
    parser.add_argument("--debug_trace_env_filter", type=str, default="")
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign", choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="tmp/songline_minigrid")
    return parser.parse_args()


def main():
    args = parse_args()
    _, summary = run_songline_experiment(args, export_outputs=True, verbose=True)

    print("\nFinal summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
