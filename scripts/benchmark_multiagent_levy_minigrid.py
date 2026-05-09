import argparse
import csv
import json
import os
from collections import defaultdict
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from scripts.songline_minigrid import (
    build_env,
    ensure_dir,
    get_goal_position,
    get_rest_position,
    get_water_position,
    manhattan,
    random_safe_action,
)
from songline_drive.trajectory_planner import TrajectoryPlanner


DEFAULT_ENV_IDS = [
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-FourRooms-v0",
]

DEFAULT_TASK_MODES = [
    "default",
    "water_search_v1",
    "rest_search_v1",
]

TEAM_CONFIGS = [
    "single",
    "two_no_comm",
    "two_comm",
]

POLICIES = [
    "random",
    "levy",
]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def safe_std(values):
    if not values:
        return 0.0
    return float(np.std(values))


def build_env_for_task(env_id: str, task_mode: str, seed: int, args):
    env_args = SimpleNamespace(
        env_id=env_id,
        task_mode=task_mode,
        water_success_radius=args.water_success_radius,
        rest_success_radius=args.rest_success_radius,
    )
    env = build_env(env_args)
    env.reset(seed=seed)
    return env


def target_position(env, task_mode: str):
    if task_mode == "water_search_v1":
        return get_water_position(env)
    if task_mode == "rest_search_v1":
        return get_rest_position(env)
    return get_goal_position(env)


def _cell_traversable(env, x: int, y: int) -> bool:
    grid = env.unwrapped.grid
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    cell = grid.get(x, y)
    if cell is None:
        return True
    return cell.type in {"goal", "ball", "box", "door"}


def traversable_cells(env):
    grid = env.unwrapped.grid
    cells = set()
    for y in range(grid.height):
        for x in range(grid.width):
            if _cell_traversable(env, x, y):
                cells.add((int(x), int(y)))
    return cells


def dir_to_delta(agent_dir: int):
    return {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1),
    }[int(agent_dir)]


def shortest_turn_action(agent_dir: int, desired_dir: int):
    right_turns = (desired_dir - agent_dir) % 4
    left_turns = (agent_dir - desired_dir) % 4
    if left_turns <= right_turns:
        return 0
    return 1


def forward_open(env, agent_dir: int):
    ax, ay = env.unwrapped.agent_pos
    dx, dy = dir_to_delta(agent_dir)
    return _cell_traversable(env, int(ax + dx), int(ay + dy))


def visible_target(agent_xy, target_xy, radius: int):
    if target_xy is None:
        return False
    return manhattan(agent_xy, target_xy) <= int(radius)


def sample_levy_run_length(rng, alpha: float, min_len: int, max_len: int):
    raw = (rng.pareto(alpha) + 1.0) * float(min_len)
    clipped = int(max(min_len, min(max_len, round(raw))))
    return clipped


def heading_score(env, candidate_dir: int, visited_counts, horizon: int):
    ax, ay = env.unwrapped.agent_pos
    dx, dy = dir_to_delta(candidate_dir)
    score = 0.0
    free_run = 0
    for step in range(1, int(horizon) + 1):
        x = int(ax + step * dx)
        y = int(ay + step * dy)
        if not _cell_traversable(env, x, y):
            break
        free_run += 1
        score += 1.0 / (1.0 + float(visited_counts.get((x, y), 0)))
    return score + (0.35 * float(free_run))


def choose_levy_heading(env, visited_counts, rng, horizon: int):
    agent_dir = int(env.unwrapped.agent_dir)
    candidates = []
    for candidate_dir in range(4):
        score = heading_score(env, candidate_dir, visited_counts, horizon=horizon)
        if candidate_dir == agent_dir:
            score += 0.05
        score += 1e-3 * float(rng.rand())
        candidates.append((score, candidate_dir))
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def levy_action(agent_state, env, rng, visited_counts, args):
    if agent_state["known_target_xy"] is not None:
        return int(
            agent_state["planner"].next_action(
                env,
                {"waypoint_xy": tuple(int(v) for v in agent_state["known_target_xy"])},
            )
        )

    agent_dir = int(env.unwrapped.agent_dir)
    need_new_segment = (
        agent_state["levy_heading"] is None
        or agent_state["levy_remaining"] <= 0
        or not forward_open(env, int(agent_state["levy_heading"]))
    )
    if need_new_segment:
        desired_dir = choose_levy_heading(
            env,
            visited_counts=visited_counts,
            rng=rng,
            horizon=args.levy_horizon,
        )
        run_length = sample_levy_run_length(
            rng,
            alpha=args.levy_alpha,
            min_len=args.levy_min_run,
            max_len=args.levy_max_run,
        )
        agent_state["levy_heading"] = int(desired_dir)
        agent_state["levy_remaining"] = int(run_length)
        agent_state["levy_run_lengths"].append(int(run_length))

    desired_dir = int(agent_state["levy_heading"])
    if agent_dir != desired_dir:
        return int(shortest_turn_action(agent_dir, desired_dir))
    if forward_open(env, agent_dir):
        agent_state["levy_remaining"] = max(0, int(agent_state["levy_remaining"]) - 1)
        return 2

    agent_state["levy_remaining"] = 0
    return int(rng.choice([0, 1]))


def update_visited(agent_state, pos_xy, shared_state, team_config: str):
    key = (int(pos_xy[0]), int(pos_xy[1]))
    agent_state["visited_counts"][key] += 1
    agent_state["visited_sequence"].append(key)
    if team_config == "two_comm":
        shared_state["visited_counts"][key] += 1


def current_target_for_agent(agent_state, shared_state, team_config: str):
    if team_config == "two_comm" and shared_state["known_target_xy"] is not None:
        return np.asarray(shared_state["known_target_xy"], dtype=np.int32)
    if agent_state["known_target_xy"] is not None:
        return np.asarray(agent_state["known_target_xy"], dtype=np.int32)
    return None


def maybe_share_target(agent_state, shared_state, agent_xy, target_xy, team_config: str, args):
    if not visible_target(agent_xy, target_xy, radius=args.discovery_radius):
        return
    agent_state["known_target_xy"] = np.asarray(target_xy, dtype=np.int32)
    agent_state["target_discoveries"] += 1
    if team_config == "two_comm":
        shared_state["known_target_xy"] = np.asarray(target_xy, dtype=np.int32)
        shared_state["target_shared"] = 1


def init_agent(env_seed: int):
    return {
        "planner": TrajectoryPlanner(),
        "visited_counts": defaultdict(int),
        "visited_sequence": [],
        "known_target_xy": None,
        "levy_heading": None,
        "levy_remaining": 0,
        "levy_run_lengths": [],
        "target_discoveries": 0,
        "env_seed": int(env_seed),
        "reached_target": 0,
        "reached_step": -1,
    }


def run_episode(args, env_id: str, task_mode: str, team_config: str, policy: str, seed: int, episode_idx: int):
    episode_seed = int(seed * 1000 + episode_idx)
    team_size = 1 if team_config == "single" else 2
    envs = [build_env_for_task(env_id, task_mode, episode_seed, args) for _ in range(team_size)]
    agents = [init_agent(episode_seed) for _ in range(team_size)]
    rngs = [np.random.RandomState(episode_seed + 97 * idx) for idx in range(team_size)]
    traversable = traversable_cells(envs[0])
    target_xy = target_position(envs[0], task_mode)
    shared_state = {
        "visited_counts": defaultdict(int),
        "known_target_xy": None,
        "target_shared": 0,
    }

    success = 0
    first_hit_step = int(args.max_steps + 1)
    team_success = 0
    team_completion_step = int(args.max_steps + 1)
    success_agent_idx = -1
    steps_taken = int(args.max_steps)

    for global_step in range(1, int(args.max_steps) + 1):
        for agent_idx, env in enumerate(envs):
            agent = agents[agent_idx]
            rng = rngs[agent_idx]
            if int(agent["reached_target"]) == 1:
                continue
            agent_xy = np.asarray(env.unwrapped.agent_pos, dtype=np.int32)
            update_visited(agent, agent_xy, shared_state, team_config)
            maybe_share_target(agent, shared_state, agent_xy, target_xy, team_config, args)

            target_for_agent = current_target_for_agent(agent, shared_state, team_config)
            if target_for_agent is not None:
                action = int(agent["planner"].next_action(env, {"waypoint_xy": tuple(int(v) for v in target_for_agent)}))
            elif policy == "random":
                action = int(random_safe_action(rng))
            elif policy == "levy":
                visited_counts = shared_state["visited_counts"] if team_config == "two_comm" else agent["visited_counts"]
                action = int(levy_action(agent, env, rng, visited_counts, args))
            else:
                raise ValueError(f"Unknown policy: {policy}")

            _, reward, terminated, truncated, _ = env.step(action)
            new_xy = np.asarray(env.unwrapped.agent_pos, dtype=np.int32)
            update_visited(agent, new_xy, shared_state, team_config)
            maybe_share_target(agent, shared_state, new_xy, target_xy, team_config, args)

            if float(reward) > 0.0 or (
                target_xy is not None
                and task_mode != "default"
                and manhattan(new_xy, target_xy) <= int(args.success_radius)
            ):
                agent["reached_target"] = 1
                agent["reached_step"] = int(global_step)
                if success == 0:
                    success = 1
                    first_hit_step = int(global_step)
                    success_agent_idx = int(agent_idx)
            if bool(terminated) or bool(truncated):
                continue
        if all(int(agent["reached_target"]) == 1 for agent in agents):
            team_success = 1
            team_completion_step = int(global_step)
            steps_taken = int(global_step)
            break

    agent_visited_sets = [set(agent["visited_counts"].keys()) for agent in agents]
    union_visited = set()
    for visited in agent_visited_sets:
        union_visited.update(visited)
    overlap_rate = 0.0
    if len(agent_visited_sets) == 2:
        overlap = len(agent_visited_sets[0].intersection(agent_visited_sets[1]))
        overlap_rate = float(overlap) / float(max(1, len(union_visited)))
    total_steps_observed = sum(len(agent["visited_sequence"]) for agent in agents)
    redundant_visit_rate = 1.0 - (float(len(union_visited)) / float(max(1, total_steps_observed)))

    all_levy_run_lengths = []
    for agent in agents:
        all_levy_run_lengths.extend(agent["levy_run_lengths"])
    long_run_fraction = 0.0
    if all_levy_run_lengths:
        long_run_fraction = safe_mean(
            [1.0 if int(run_len) >= int(args.long_run_threshold) else 0.0 for run_len in all_levy_run_lengths]
        )

    row = {
        "env_id": env_id,
        "task_mode": task_mode,
        "team_config": team_config,
        "team_size": int(team_size),
        "communication_enabled": int(team_config == "two_comm"),
        "policy": policy,
        "seed": int(seed),
        "episode": int(episode_idx),
        "success": int(success),
        "team_success": int(team_success if team_size > 1 else success),
        "first_hit_step": int(first_hit_step),
        "discovery_latency": float(first_hit_step if success else (args.max_steps + 1)),
        "team_completion_step": int(team_completion_step if team_size > 1 else first_hit_step),
        "team_completion_latency": float(
            (team_completion_step if team_size > 1 else first_hit_step)
            if (team_success if team_size > 1 else success)
            else (args.max_steps + 1)
        ),
        "steps_taken": int(steps_taken),
        "success_agent_idx": int(success_agent_idx),
        "agents_reached_fraction": safe_mean([float(agent["reached_target"]) for agent in agents]),
        "coverage_rate": float(len(union_visited)) / float(max(1, len(traversable))),
        "overlap_rate": float(overlap_rate),
        "redundant_visit_rate": float(redundant_visit_rate),
        "target_shared": int(shared_state["target_shared"]),
        "mean_levy_run_length": safe_mean([float(v) for v in all_levy_run_lengths]),
        "long_run_fraction": float(long_run_fraction),
        "num_levy_segments": int(len(all_levy_run_lengths)),
    }

    for env in envs:
        env.close()
    return row, all_levy_run_lengths


def aggregate_rows(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
    out_rows = []
    for key, group in sorted(grouped.items()):
        out = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        out["num_episodes"] = int(len(group))
        for metric in metric_keys:
            vals = [float(item[metric]) for item in group]
            out[f"{metric}_mean"] = safe_mean(vals)
            out[f"{metric}_std"] = safe_std(vals)
        out_rows.append(out)
    return out_rows


def compute_speedup_rows(aggregate_rows_list):
    indexed = {}
    for row in aggregate_rows_list:
        key = (row["env_id"], row["task_mode"], row["policy"], row["team_config"])
        indexed[key] = row
    speedup_rows = []
    base_keys = sorted({(env_id, task_mode, policy) for env_id, task_mode, policy, _ in indexed.keys()})
    for env_id, task_mode, policy in base_keys:
        single = indexed.get((env_id, task_mode, policy, "single"))
        if single is None:
            continue
        for team_config in ("two_no_comm", "two_comm"):
            team = indexed.get((env_id, task_mode, policy, team_config))
            if team is None:
                continue
            single_latency = float(single["discovery_latency_mean"])
            team_latency = float(team["discovery_latency_mean"])
            speedup = 0.0
            if team_latency > 1e-6:
                speedup = single_latency / team_latency
            speedup_rows.append(
                {
                    "env_id": env_id,
                    "task_mode": task_mode,
                    "policy": policy,
                    "team_config": team_config,
                    "single_success_rate": float(single["success_mean"]),
                    "team_success_rate": float(team["success_mean"]),
                    "single_discovery_latency": single_latency,
                    "team_discovery_latency": team_latency,
                    "discovery_speedup_vs_single": float(speedup),
                    "team_completion_rate": float(team["team_success_mean"]),
                    "team_completion_latency": float(team["team_completion_latency_mean"]),
                }
            )
    return speedup_rows


def plot_metric(aggregate_rows_list, metric: str, ylabel: str, out_path: str):
    env_task_labels = sorted(
        {(str(row["env_id"]), str(row["task_mode"])) for row in aggregate_rows_list}
    )
    series_labels = sorted(
        {f"{row['policy']}:{row['team_config']}" for row in aggregate_rows_list}
    )
    x = np.arange(len(env_task_labels))
    width = 0.82 / max(1, len(series_labels))
    fig, ax = plt.subplots(figsize=(13, 5.2))
    center = (len(series_labels) - 1) / 2.0
    for idx, series in enumerate(series_labels):
        vals = []
        for env_id, task_mode in env_task_labels:
            match = next(
                (
                    row for row in aggregate_rows_list
                    if str(row["env_id"]) == env_id
                    and str(row["task_mode"]) == task_mode
                    and f"{row['policy']}:{row['team_config']}" == series
                ),
                None,
            )
            vals.append(0.0 if match is None else float(match[f"{metric}_mean"]))
        ax.bar(x + (idx - center) * width, vals, width=width, label=series)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{env_id}\n{task_mode}" for env_id, task_mode in env_task_labels], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_levy_histogram(levy_run_lengths, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    if levy_run_lengths:
        bins = np.arange(1, max(levy_run_lengths) + 2) - 0.5
        ax.hist(levy_run_lengths, bins=bins, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Levy segment length")
    ax.set_ylabel("count")
    ax.set_title("Levy walk segment-length distribution")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_report(aggregate_rows_list, speedup_rows, args, out_path: str):
    grouped = {}
    for row in aggregate_rows_list:
        key = (row["env_id"], row["task_mode"])
        grouped.setdefault(key, []).append(row)

    lines = []
    lines.append("# Multi-Agent and Levy Exploration Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- env_ids: {', '.join(args.env_ids)}")
    lines.append(f"- task_modes: {', '.join(args.task_modes)}")
    lines.append(f"- seeds: {', '.join(str(v) for v in args.seeds)}")
    lines.append(f"- episodes_per_seed: {args.episodes}")
    lines.append(f"- max_steps: {args.max_steps}")
    lines.append(f"- levy_alpha: {args.levy_alpha}")
    lines.append("")
    lines.append("## Why Levy")
    lines.append("Levy-like exploration uses many short runs mixed with occasional long relocations.")
    lines.append("This pattern is often used as a compact model of animal search motion when resources are sparse or weakly localized.")
    lines.append("The benchmark operationalizes that idea with Pareto-distributed straight-line segments.")
    lines.append("")
    lines.append("## Main findings")
    for (env_id, task_mode), rows in sorted(grouped.items()):
        lines.append(f"### {env_id} / {task_mode}")
        best_success = max(rows, key=lambda row: float(row["success_mean"]))
        best_latency = min(rows, key=lambda row: float(row["discovery_latency_mean"]))
        two_agent_rows = [row for row in rows if int(row.get("team_size", 1)) > 1]
        best_team = max(two_agent_rows, key=lambda row: float(row.get("team_success_mean", 0.0))) if two_agent_rows else best_success
        lines.append(
            f"- best success: `{best_success['policy']}:{best_success['team_config']}` = {best_success['success_mean']:.3f}"
        )
        lines.append(
            f"- best discovery latency: `{best_latency['policy']}:{best_latency['team_config']}` = {best_latency['discovery_latency_mean']:.2f}"
        )
        lines.append(
            f"- best two-agent completion: `{best_team['policy']}:{best_team['team_config']}` = "
            f"{best_team.get('team_success_mean', 0.0):.3f} "
            f"(latency {best_team.get('team_completion_latency_mean', 0.0):.2f})"
        )
        lines.append("")

    lines.append("## Two-agent speedup")
    for row in speedup_rows:
        lines.append(
            f"- {row['env_id']} / {row['task_mode']} / {row['policy']} / {row['team_config']}: "
            f"speedup={row['discovery_speedup_vs_single']:.3f}, "
            f"success={row['team_success_rate']:.3f} vs {row['single_success_rate']:.3f}"
        )
    lines.append("")

    with open(out_path, "w") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENV_IDS)
    parser.add_argument("--task_modes", nargs="+", default=DEFAULT_TASK_MODES, choices=DEFAULT_TASK_MODES)
    parser.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    parser.add_argument("--team_configs", nargs="+", default=TEAM_CONFIGS, choices=TEAM_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 7, 11])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--water_success_radius", type=int, default=1)
    parser.add_argument("--rest_success_radius", type=int, default=1)
    parser.add_argument("--success_radius", type=int, default=1)
    parser.add_argument("--discovery_radius", type=int, default=2)
    parser.add_argument("--levy_alpha", type=float, default=1.5)
    parser.add_argument("--levy_min_run", type=int, default=1)
    parser.add_argument("--levy_max_run", type=int, default=12)
    parser.add_argument("--levy_horizon", type=int, default=4)
    parser.add_argument("--long_run_threshold", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="tmp/multiagent_levy_minigrid")
    return parser.parse_args()


def run_benchmark(args):
    ensure_dir(args.out_dir)
    episode_rows = []
    levy_run_lengths = []
    for env_id in args.env_ids:
        for task_mode in args.task_modes:
            for policy in args.policies:
                for team_config in args.team_configs:
                    for seed in args.seeds:
                        for episode_idx in range(int(args.episodes)):
                            row, run_lengths = run_episode(
                                args=args,
                                env_id=env_id,
                                task_mode=task_mode,
                                team_config=team_config,
                                policy=policy,
                                seed=seed,
                                episode_idx=episode_idx,
                            )
                            episode_rows.append(row)
                            levy_run_lengths.extend(run_lengths)

    episode_fieldnames = [
        "env_id",
        "task_mode",
        "team_config",
        "team_size",
        "communication_enabled",
        "policy",
        "seed",
        "episode",
        "success",
        "team_success",
        "first_hit_step",
        "discovery_latency",
        "team_completion_step",
        "team_completion_latency",
        "steps_taken",
        "success_agent_idx",
        "agents_reached_fraction",
        "coverage_rate",
        "overlap_rate",
        "redundant_visit_rate",
        "target_shared",
        "mean_levy_run_length",
        "long_run_fraction",
        "num_levy_segments",
    ]
    write_csv(os.path.join(args.out_dir, "episode_results.csv"), episode_rows, episode_fieldnames)
    with open(os.path.join(args.out_dir, "episode_results.json"), "w") as file_obj:
        json.dump(episode_rows, file_obj, indent=2)

    aggregate_metric_keys = [
        "success",
        "team_success",
        "discovery_latency",
        "team_completion_latency",
        "agents_reached_fraction",
        "coverage_rate",
        "overlap_rate",
        "redundant_visit_rate",
        "target_shared",
        "mean_levy_run_length",
        "long_run_fraction",
        "num_levy_segments",
    ]
    aggregate_rows_list = aggregate_rows(
        episode_rows,
        group_keys=["env_id", "task_mode", "policy", "team_config", "team_size", "communication_enabled"],
        metric_keys=aggregate_metric_keys,
    )
    write_csv(
        os.path.join(args.out_dir, "aggregate_overall.csv"),
        aggregate_rows_list,
        list(aggregate_rows_list[0].keys()) if aggregate_rows_list else [],
    )
    with open(os.path.join(args.out_dir, "aggregate_overall.json"), "w") as file_obj:
        json.dump(aggregate_rows_list, file_obj, indent=2)

    speedup_rows = compute_speedup_rows(aggregate_rows_list)
    if speedup_rows:
        write_csv(
            os.path.join(args.out_dir, "speedup_vs_single.csv"),
            speedup_rows,
            list(speedup_rows[0].keys()),
        )
        with open(os.path.join(args.out_dir, "speedup_vs_single.json"), "w") as file_obj:
            json.dump(speedup_rows, file_obj, indent=2)

    plot_metric(
        aggregate_rows_list,
        metric="success",
        ylabel="Success rate",
        out_path=os.path.join(args.out_dir, "success_rate.png"),
    )
    plot_metric(
        aggregate_rows_list,
        metric="discovery_latency",
        ylabel="Discovery latency",
        out_path=os.path.join(args.out_dir, "discovery_latency.png"),
    )
    plot_metric(
        aggregate_rows_list,
        metric="team_success",
        ylabel="Team completion rate",
        out_path=os.path.join(args.out_dir, "team_completion_rate.png"),
    )
    plot_metric(
        aggregate_rows_list,
        metric="team_completion_latency",
        ylabel="Team completion latency",
        out_path=os.path.join(args.out_dir, "team_completion_latency.png"),
    )
    plot_metric(
        aggregate_rows_list,
        metric="coverage_rate",
        ylabel="Coverage rate",
        out_path=os.path.join(args.out_dir, "coverage_rate.png"),
    )
    plot_metric(
        aggregate_rows_list,
        metric="overlap_rate",
        ylabel="Two-agent overlap rate",
        out_path=os.path.join(args.out_dir, "overlap_rate.png"),
    )
    plot_levy_histogram(
        levy_run_lengths,
        out_path=os.path.join(args.out_dir, "levy_run_length_histogram.png"),
    )
    build_report(
        aggregate_rows_list=aggregate_rows_list,
        speedup_rows=speedup_rows,
        args=args,
        out_path=os.path.join(args.out_dir, "report.md"),
    )
    return {
        "episode_rows": episode_rows,
        "aggregate_rows": aggregate_rows_list,
        "speedup_rows": speedup_rows,
        "levy_run_lengths": levy_run_lengths,
        "out_dir": args.out_dir,
    }


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
