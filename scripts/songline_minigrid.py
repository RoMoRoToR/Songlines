import argparse
import json
import os
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid
import numpy as np

from utils.lz_memory import LZMapMemory, SymbolicTokenizer


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


def manhattan(a, b):
    return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))


def choose_action_toward(env, target_xy):
    """
    Простой локальный контроллер:
    поворачивает агента к target_xy, затем идёт вперёд.
    """
    unwrapped = env.unwrapped
    ax, ay = unwrapped.agent_pos
    agent_dir = unwrapped.agent_dir

    tx, ty = int(target_xy[0]), int(target_xy[1])
    dx = tx - ax
    dy = ty - ay

    if dx == 0 and dy == 0:
        return 2  # forward

    if abs(dx) >= abs(dy):
        desired_dir = 0 if dx > 0 else 2
    else:
        desired_dir = 1 if dy > 0 else 3

    if desired_dir == agent_dir:
        return 2  # forward

    right_turns = (desired_dir - agent_dir) % 4
    left_turns = (agent_dir - desired_dir) % 4

    if left_turns <= right_turns:
        return 0  # left
    return 1  # right


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


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


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
    tokenizer = SymbolicTokenizer(
        mode=args.tokenizer_mode,
        proj_dim=args.tokenizer_proj_dim,
        seed=args.seed,
    )
    memory = LZMapMemory(min_goal_visits=args.min_goal_visits)
    return tokenizer, memory


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


def remember_graph_path(memory, src, dst):
    if memory is None or src is None or dst is None:
        return None
    if src == dst:
        return [src]

    parents = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nxt in memory.edges.get(cur, {}):
            if nxt in parents:
                continue
            parents[nxt] = cur
            if nxt == dst:
                path = [dst]
                while parents[path[-1]] is not None:
                    path.append(parents[path[-1]])
                path.reverse()
                return path
            q.append(nxt)
    return None


def choose_songline_target_node(memory, top_k):
    if memory is None or memory.current_phrase_id is None:
        return None

    scored = []
    for node_id, node in memory.nodes.items():
        if node["visits"] < memory.min_goal_visits:
            continue
        goal_xy = get_mean_xy(node, "goal")
        pose_xy = get_mean_xy(node, "pose")
        if goal_xy is None or pose_xy is None:
            continue
        mean_reward = node["reward_sum"] / max(1, node["reward_count"])
        visit_bonus = np.log1p(node["visits"])
        goal_alignment = -manhattan(np.rint(pose_xy).astype(np.int32), np.rint(goal_xy).astype(np.int32))
        score = (2.0 * mean_reward) + (0.15 * visit_bonus) + (0.05 * goal_alignment)
        scored.append((score, node_id))

    if not scored:
        return None

    scored.sort(reverse=True)
    candidate_ids = [node_id for _, node_id in scored[:top_k]]
    src = current_phrase_node(memory)
    best = None
    for node_id in candidate_ids:
        path = remember_graph_path(memory, src, node_id)
        if path is None:
            continue
        path_len = max(0, len(path) - 1)
        node = memory.nodes[node_id]
        mean_reward = node["reward_sum"] / max(1, node["reward_count"])
        score = scored[[nid for _, nid in scored].index(node_id)][0]
        candidate = {
            "node_id": node_id,
            "path": path,
            "path_len": path_len,
            "score": float(score),
            "mean_reward": float(mean_reward),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def select_graph_waypoint(memory, top_k):
    target = choose_songline_target_node(memory, top_k=top_k)
    if target is None:
        return None

    path = target["path"]
    next_node_id = path[1] if len(path) > 1 else path[0]
    node = memory.nodes[next_node_id]
    pose_xy = get_mean_xy(node, "pose")
    if pose_xy is None:
        return None

    waypoint = np.rint(pose_xy).astype(np.int32)
    return {
        "waypoint_xy": waypoint,
        "target_node_id": int(target["node_id"]),
        "next_node_id": int(next_node_id),
        "graph_path_length": int(target["path_len"]),
    }


def make_method_name(agent_mode, songline_policy):
    if agent_mode == "songline":
        return f"songline_{songline_policy}"
    return agent_mode


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
        "final_nodes": final_nodes,
        "final_edges": final_edges,
    }
    return summary


def run_songline_experiment(args, export_outputs=True, verbose=True):
    ensure_dir(args.out_dir)

    env = gym.make(args.env_id, render_mode="rgb_array")
    rng = np.random.RandomState(args.seed)
    method_name = make_method_name(args.agent_mode, args.songline_policy)

    tokenizer = None
    memory = None
    if args.agent_mode == "songline":
        tokenizer, memory = build_memory(args)

    run_summary = {
        "env_id": args.env_id,
        "agent_mode": args.agent_mode,
        "songline_policy": args.songline_policy,
        "method": method_name,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "episode_returns": [],
        "episode_lengths": [],
        "successes": [],
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
        "episode_metrics": [],
    }

    total_step_idx = 0
    seen_songline_nodes = set()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        del obs, info

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

        for step in range(args.max_steps):
            step_count = step + 1
            unwrapped = env.unwrapped
            agent_xy = np.array(unwrapped.agent_pos, dtype=np.int32)
            distance_before = None if goal_xy is None else manhattan(agent_xy, goal_xy)

            if args.agent_mode == "songline":
                obs_vec = local_symbolic_observation(env, radius=1)
                token = tokenizer.encode(obs_vec)
                is_new_node = memory.update_token(token, total_step_idx)
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
                )

                if step % args.suggest_every == 0:
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

                    if args.songline_policy == "graph_path":
                        path_plan = select_graph_waypoint(memory, top_k=args.top_k_goals)
                        if path_plan is not None:
                            subgoal_xy = path_plan["waypoint_xy"].copy()
                            interventions_this_episode += 1
                            active_intervention_distance = distance_before
                            active_subgoal_key = (int(subgoal_xy[0]), int(subgoal_xy[1]))
                            active_graph_path_length = int(path_plan["graph_path_length"])
                            graph_path_lengths.append(active_graph_path_length)

            action = random_safe_action(rng)
            if args.agent_mode == "greedy":
                action = choose_action_toward(env, goal_xy) if goal_xy is not None else random_safe_action(rng)
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
                    action = choose_action_toward(env, subgoal_xy)
                else:
                    action = choose_action_toward(env, goal_xy) if goal_xy is not None else random_safe_action(rng)
            elif args.agent_mode == "songline":
                if args.songline_policy == "no_override":
                    action = choose_action_toward(env, goal_xy) if goal_xy is not None else random_safe_action(rng)
                elif args.songline_policy == "subgoal_controller":
                    if subgoal_xy is not None and rng.rand() > args.epsilon:
                        action = choose_action_toward(env, subgoal_xy)
                    elif goal_xy is not None and rng.rand() > 0.5:
                        action = choose_action_toward(env, goal_xy)
                    else:
                        action = random_safe_action(rng)
                elif args.songline_policy == "graph_path":
                    if subgoal_xy is not None and rng.rand() > args.epsilon:
                        action = choose_action_toward(env, subgoal_xy)
                    elif goal_xy is not None:
                        action = choose_action_toward(env, goal_xy)
                    else:
                        action = random_safe_action(rng)
                else:
                    raise ValueError(f"Unknown songline policy: {args.songline_policy}")

            obs, reward, terminated, truncated, info = env.step(action)
            del obs, info

            total_step_idx += 1
            episode_return += float(reward)
            agent_xy_new = np.array(env.unwrapped.agent_pos, dtype=np.int32)
            distance_after = None if goal_xy is None else manhattan(agent_xy_new, goal_xy)

            if args.agent_mode == "songline":
                if previous_goal_distance is not None and distance_after is not None:
                    improved = distance_after < previous_goal_distance
                    memory.record_plan_outcome(improved)

                memory.observe(
                    total_step_idx,
                    pose_xy=agent_xy_new,
                    goal_xy=goal_xy,
                    reward=float(reward),
                )

            if args.agent_mode == "greedy_episodic":
                episodic_observe(ep_memory, agent_xy_new, distance_before, distance_after)

            if active_subgoal_key is not None:
                if int(agent_xy_new[0]) == active_subgoal_key[0] and int(agent_xy_new[1]) == active_subgoal_key[1]:
                    subgoal_hits_this_episode += 1
                    if active_intervention_distance is not None and distance_after is not None:
                        distance_delta_sum += float(active_intervention_distance - distance_after)
                    active_subgoal_key = None
                    active_intervention_distance = None
                    if args.agent_mode == "greedy_episodic" and ep_memory["active_waypoint"] is not None:
                        ep_memory["active_waypoint"] = None
                elif active_intervention_distance is not None and distance_after is not None:
                    if distance_after > active_intervention_distance + args.intervention_patience:
                        active_subgoal_key = None
                        active_intervention_distance = None
                        if args.agent_mode == "greedy_episodic" and ep_memory["active_waypoint"] is not None:
                            ep_memory["active_waypoint"] = None

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

        episode_metrics = {
            "episode": ep + 1,
            "env_id": args.env_id,
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
        }

        run_summary["episode_returns"].append(float(episode_return))
        run_summary["episode_lengths"].append(int(step_count))
        run_summary["successes"].append(int(success))
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
