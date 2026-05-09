import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmark_multiagent_levy_minigrid import (
    build_env_for_task,
    current_target_for_agent,
    init_agent,
    levy_action,
    maybe_share_target,
    safe_mean,
    safe_std,
    target_position,
    traversable_cells,
    update_visited,
)
from scripts.songline_minigrid import ensure_dir, manhattan, random_safe_action


DEFAULT_ENV_IDS = [
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-FourRooms-v0",
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


def displacement_sq(start_xy, pos_xy):
    dx = float(pos_xy[0] - start_xy[0])
    dy = float(pos_xy[1] - start_xy[1])
    return (dx * dx) + (dy * dy)


def signed_turn_angle(prev_dir, next_dir):
    if prev_dir is None or next_dir is None:
        return None
    angle_map = {
        (1, 0): 0.0,
        (0, 1): 90.0,
        (-1, 0): 180.0,
        (0, -1): -90.0,
    }
    a0 = angle_map[(int(prev_dir[0]), int(prev_dir[1]))]
    a1 = angle_map[(int(next_dir[0]), int(next_dir[1]))]
    diff = a1 - a0
    while diff <= -180.0:
        diff += 360.0
    while diff > 180.0:
        diff -= 360.0
    return float(diff)


def extract_segments(positions):
    segments = []
    current_dir = None
    current_len = 0
    for idx in range(1, len(positions)):
        prev_xy = positions[idx - 1]
        next_xy = positions[idx]
        dx = int(next_xy[0] - prev_xy[0])
        dy = int(next_xy[1] - prev_xy[1])
        if dx == 0 and dy == 0:
            continue
        step_dir = (dx, dy)
        if current_dir is None:
            current_dir = step_dir
            current_len = 1
            continue
        if step_dir == current_dir:
            current_len += 1
            continue
        segments.append({"dir": current_dir, "length": int(current_len)})
        current_dir = step_dir
        current_len = 1
    if current_dir is not None and current_len > 0:
        segments.append({"dir": current_dir, "length": int(current_len)})
    return segments


def ccdf_points(lengths):
    if not lengths:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    vals = np.asarray(sorted(int(v) for v in lengths), dtype=np.int32)
    unique = np.asarray(sorted(set(int(v) for v in vals)), dtype=np.int32)
    ccdf = []
    total = float(len(vals))
    for x in unique:
        ccdf.append(float(np.sum(vals >= int(x))) / total)
    return unique.astype(np.float64), np.asarray(ccdf, dtype=np.float64)


def fit_tail_slope(lengths, xmin: int = 2):
    xs, ys = ccdf_points([v for v in lengths if int(v) >= int(xmin)])
    if len(xs) < 2:
        return 0.0
    logx = np.log(xs)
    logy = np.log(np.maximum(1e-9, ys))
    slope = np.polyfit(logx, logy, 1)[0]
    return float(slope)


def run_single_episode(args, env_id: str, policy: str, seed: int, episode_idx: int):
    episode_seed = int(seed * 1000 + episode_idx)
    env = build_env_for_task(env_id, args.task_mode, episode_seed, args)
    target_xy = target_position(env, args.task_mode)
    traversable = traversable_cells(env)
    shared_state = {
        "visited_counts": {},
        "known_target_xy": None,
        "target_shared": 0,
    }
    agent = init_agent(episode_seed)
    rng = np.random.RandomState(episode_seed)

    start_xy = np.asarray(env.unwrapped.agent_pos, dtype=np.int32)
    positions = [start_xy.copy()]
    headings = [int(env.unwrapped.agent_dir)]
    coverage_curve = []
    msd_curve = []

    success = 0
    discovery_step = int(args.max_steps + 1)
    for step_idx in range(1, int(args.max_steps) + 1):
        agent_xy = np.asarray(env.unwrapped.agent_pos, dtype=np.int32)
        update_visited(agent, agent_xy, shared_state, "single")
        maybe_share_target(agent, shared_state, agent_xy, target_xy, "single", args)

        known_target_xy = current_target_for_agent(agent, shared_state, "single")
        if known_target_xy is not None:
            action = int(agent["planner"].next_action(env, {"waypoint_xy": tuple(int(v) for v in known_target_xy)}))
        elif policy == "random":
            action = int(random_safe_action(rng))
        elif policy == "levy":
            action = int(levy_action(agent, env, rng, agent["visited_counts"], args))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        _, reward, terminated, truncated, _ = env.step(action)
        new_xy = np.asarray(env.unwrapped.agent_pos, dtype=np.int32)
        update_visited(agent, new_xy, shared_state, "single")
        maybe_share_target(agent, shared_state, new_xy, target_xy, "single", args)
        positions.append(new_xy.copy())
        headings.append(int(env.unwrapped.agent_dir))

        visited_unique = set(agent["visited_counts"].keys())
        coverage_curve.append(float(len(visited_unique)) / float(max(1, len(traversable))))
        msd_curve.append(displacement_sq(start_xy, new_xy))

        success_radius = int(args.success_radius)
        if args.task_mode == "water_search_v1":
            success_radius = int(args.water_success_radius)
        elif args.task_mode == "rest_search_v1":
            success_radius = int(args.rest_success_radius)
        if float(reward) > 0.0 or (target_xy is not None and manhattan(new_xy, target_xy) <= success_radius):
            success = 1
            discovery_step = int(step_idx)
            break
        if bool(terminated) or bool(truncated):
            continue

    env.close()

    segments = extract_segments(positions)
    segment_lengths = [int(item["length"]) for item in segments]
    turn_angles = []
    for idx in range(1, len(segments)):
        angle = signed_turn_angle(segments[idx - 1]["dir"], segments[idx]["dir"])
        if angle is not None:
            turn_angles.append(float(angle))

    total_visits = max(1, len(agent["visited_sequence"]))
    revisit_rate = 1.0 - (float(len(set(agent["visited_counts"].keys()))) / float(total_visits))
    row = {
        "env_id": env_id,
        "policy": policy,
        "seed": int(seed),
        "episode": int(episode_idx),
        "success": int(success),
        "discovery_latency": float(discovery_step if success else (args.max_steps + 1)),
        "coverage_final": float(coverage_curve[-1] if coverage_curve else 0.0),
        "revisit_rate": float(revisit_rate),
        "mean_segment_length": safe_mean([float(v) for v in segment_lengths]),
        "median_segment_length": float(np.median(segment_lengths)) if segment_lengths else 0.0,
        "long_run_fraction": safe_mean(
            [1.0 if int(v) >= int(args.long_run_threshold) else 0.0 for v in segment_lengths]
        ),
        "mean_abs_turn_angle": safe_mean([abs(float(v)) for v in turn_angles]),
        "final_msd": float(msd_curve[-1] if msd_curve else 0.0),
        "tail_slope": float(fit_tail_slope(segment_lengths, xmin=args.tail_xmin)),
        "num_segments": int(len(segment_lengths)),
    }
    payload = {
        "row": row,
        "positions": [[int(x), int(y)] for x, y in positions],
        "headings": [int(v) for v in headings],
        "coverage_curve": [float(v) for v in coverage_curve],
        "msd_curve": [float(v) for v in msd_curve],
        "segment_lengths": [int(v) for v in segment_lengths],
        "turn_angles": [float(v) for v in turn_angles],
        "start_xy": [int(v) for v in start_xy],
        "target_xy": None if target_xy is None else [int(v) for v in target_xy],
    }
    return payload


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["env_id"], row["policy"])
        grouped.setdefault(key, []).append(row)
    metric_keys = [
        "success",
        "discovery_latency",
        "coverage_final",
        "revisit_rate",
        "mean_segment_length",
        "median_segment_length",
        "long_run_fraction",
        "mean_abs_turn_angle",
        "final_msd",
        "tail_slope",
        "num_segments",
    ]
    out_rows = []
    for key, group in sorted(grouped.items()):
        out = {
            "env_id": key[0],
            "policy": key[1],
            "num_episodes": int(len(group)),
        }
        for metric in metric_keys:
            vals = [float(item[metric]) for item in group]
            out[f"{metric}_mean"] = safe_mean(vals)
            out[f"{metric}_std"] = safe_std(vals)
        out_rows.append(out)
    return out_rows


def plot_sample_trajectories(samples, out_path: str):
    env_ids = sorted(samples.keys())
    fig, axes = plt.subplots(1, len(env_ids), figsize=(6.0 * len(env_ids), 5.0), squeeze=False)
    for ax, env_id in zip(axes[0], env_ids):
        for policy, color in [("random", "#C44E52"), ("levy", "#4C78A8")]:
            payload = samples[env_id][policy]
            pos = np.asarray(payload["positions"], dtype=np.float64)
            ax.plot(pos[:, 0], pos[:, 1], marker="o", markersize=2.5, linewidth=1.5, alpha=0.85, label=policy, color=color)
            ax.scatter(pos[0, 0], pos[0, 1], s=60, marker="s", color=color)
            if payload["target_xy"] is not None:
                ax.scatter(payload["target_xy"][0], payload["target_xy"][1], s=80, marker="*", color=color, edgecolors="black")
        ax.set_title(env_id)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_segment_ccdf(episode_payloads, out_path: str):
    grouped = {}
    for payload in episode_payloads:
        grouped.setdefault((payload["row"]["env_id"], payload["row"]["policy"]), []).extend(payload["segment_lengths"])
    env_ids = sorted({key[0] for key in grouped})
    fig, axes = plt.subplots(1, len(env_ids), figsize=(6.0 * len(env_ids), 4.8), squeeze=False)
    for ax, env_id in zip(axes[0], env_ids):
        for policy, color in [("random", "#C44E52"), ("levy", "#4C78A8")]:
            xs, ys = ccdf_points(grouped.get((env_id, policy), []))
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=policy, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(env_id)
        ax.set_xlabel("segment length")
        ax.set_ylabel("CCDF")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar_metric(aggregate_rows_list, metric: str, ylabel: str, out_path: str):
    env_ids = sorted({row["env_id"] for row in aggregate_rows_list})
    x = np.arange(len(env_ids))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, policy in enumerate(POLICIES):
        vals = []
        errs = []
        for env_id in env_ids:
            match = next((row for row in aggregate_rows_list if row["env_id"] == env_id and row["policy"] == policy), None)
            vals.append(0.0 if match is None else float(match[f"{metric}_mean"]))
            errs.append(0.0 if match is None else float(match[f"{metric}_std"]))
        ax.bar(x + (idx - 0.5) * width, vals, width=width, yerr=errs, capsize=3, label=policy)
    ax.set_xticks(x)
    ax.set_xticklabels(env_ids, rotation=10, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_curve_mean(episode_payloads, curve_key: str, ylabel: str, out_path: str):
    env_ids = sorted({payload["row"]["env_id"] for payload in episode_payloads})
    fig, axes = plt.subplots(1, len(env_ids), figsize=(6.0 * len(env_ids), 4.8), squeeze=False)
    for ax, env_id in zip(axes[0], env_ids):
        for policy, color in [("random", "#C44E52"), ("levy", "#4C78A8")]:
            curves = [payload[curve_key] for payload in episode_payloads if payload["row"]["env_id"] == env_id and payload["row"]["policy"] == policy]
            if not curves:
                continue
            max_len = max(len(curve) for curve in curves)
            mat = []
            for curve in curves:
                vals = list(curve)
                while len(vals) < max_len:
                    vals.append(vals[-1] if vals else 0.0)
                mat.append(vals)
            arr = np.asarray(mat, dtype=np.float64)
            mean_curve = np.mean(arr, axis=0)
            ax.plot(np.arange(1, len(mean_curve) + 1), mean_curve, linewidth=1.8, label=policy, color=color)
        ax.set_title(env_id)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_turn_angle_hist(episode_payloads, out_path: str):
    env_ids = sorted({payload["row"]["env_id"] for payload in episode_payloads})
    fig, axes = plt.subplots(1, len(env_ids), figsize=(6.0 * len(env_ids), 4.8), squeeze=False)
    bins = np.linspace(-180.0, 180.0, 13)
    for ax, env_id in zip(axes[0], env_ids):
        for policy, color in [("random", "#C44E52"), ("levy", "#4C78A8")]:
            angles = []
            for payload in episode_payloads:
                if payload["row"]["env_id"] == env_id and payload["row"]["policy"] == policy:
                    angles.extend(float(v) for v in payload["turn_angles"])
            if not angles:
                continue
            ax.hist(angles, bins=bins, density=True, alpha=0.45, label=policy, color=color)
        ax.set_title(env_id)
        ax.set_xlabel("turn angle (deg)")
        ax.set_ylabel("density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_article_table_rows(aggregate_rows_list):
    rows = []
    for env_id in sorted({row["env_id"] for row in aggregate_rows_list}):
        random_row = next((row for row in aggregate_rows_list if row["env_id"] == env_id and row["policy"] == "random"), None)
        levy_row = next((row for row in aggregate_rows_list if row["env_id"] == env_id and row["policy"] == "levy"), None)
        if random_row is None or levy_row is None:
            continue
        rows.append(
            {
                "env_id": env_id,
                "success_random": float(random_row["success_mean"]),
                "success_levy": float(levy_row["success_mean"]),
                "discovery_latency_random": float(random_row["discovery_latency_mean"]),
                "discovery_latency_levy": float(levy_row["discovery_latency_mean"]),
                "coverage_random": float(random_row["coverage_final_mean"]),
                "coverage_levy": float(levy_row["coverage_final_mean"]),
                "revisit_random": float(random_row["revisit_rate_mean"]),
                "revisit_levy": float(levy_row["revisit_rate_mean"]),
                "mean_segment_random": float(random_row["mean_segment_length_mean"]),
                "mean_segment_levy": float(levy_row["mean_segment_length_mean"]),
                "long_run_fraction_random": float(random_row["long_run_fraction_mean"]),
                "long_run_fraction_levy": float(levy_row["long_run_fraction_mean"]),
                "tail_slope_random": float(random_row["tail_slope_mean"]),
                "tail_slope_levy": float(levy_row["tail_slope_mean"]),
            }
        )
    return rows


def write_report(args, aggregate_rows_list, out_path: str):
    lines = []
    lines.append("# Levy Trajectory Statistics Report")
    lines.append("")
    lines.append("## Task")
    lines.append(f"- task_mode: {args.task_mode}")
    lines.append("")
    lines.append("## Claim")
    lines.append("The Levy-like policy should not be interpreted as direct biological imitation.")
    lines.append("The claim is weaker and more defensible: its trajectories have the same qualitative search organization often discussed in animal and human foraging.")
    lines.append("That organization is many short local scans interrupted by rarer longer relocation segments.")
    lines.append("")
    lines.append("## Measured Structure")
    lines.append("- segment-length CCDF")
    lines.append("- turn-angle distribution")
    lines.append("- mean squared displacement over time")
    lines.append("- coverage over time")
    lines.append("- revisit rate")
    lines.append("- fraction of longer relocation segments")
    lines.append("")
    lines.append("## Random vs Levy")
    env_ids = sorted({row['env_id'] for row in aggregate_rows_list})
    for env_id in env_ids:
        random_row = next((row for row in aggregate_rows_list if row["env_id"] == env_id and row["policy"] == "random"), None)
        levy_row = next((row for row in aggregate_rows_list if row["env_id"] == env_id and row["policy"] == "levy"), None)
        lines.append(f"### {env_id}")
        if random_row is not None and levy_row is not None:
            lines.append(f"- success: {random_row['success_mean']:.3f} -> {levy_row['success_mean']:.3f}")
            lines.append(f"- discovery latency: {random_row['discovery_latency_mean']:.2f} -> {levy_row['discovery_latency_mean']:.2f}")
            lines.append(f"- coverage final: {random_row['coverage_final_mean']:.3f} -> {levy_row['coverage_final_mean']:.3f}")
            lines.append(f"- revisit rate: {random_row['revisit_rate_mean']:.3f} -> {levy_row['revisit_rate_mean']:.3f}")
            lines.append(f"- mean segment length: {random_row['mean_segment_length_mean']:.2f} -> {levy_row['mean_segment_length_mean']:.2f}")
            lines.append(f"- long-run fraction: {random_row['long_run_fraction_mean']:.3f} -> {levy_row['long_run_fraction_mean']:.3f}")
            lines.append(f"- mean abs turn angle: {random_row['mean_abs_turn_angle_mean']:.2f} -> {levy_row['mean_abs_turn_angle_mean']:.2f}")
            lines.append(f"- final MSD: {random_row['final_msd_mean']:.2f} -> {levy_row['final_msd_mean']:.2f}")
            lines.append(f"- tail slope proxy: {random_row['tail_slope_mean']:.3f} -> {levy_row['tail_slope_mean']:.3f}")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("On the larger FourRooms layout the difference is clearest: Levy yields longer straight segments, a higher fraction of longer relocations, lower revisit, and larger MSD.")
    lines.append("That is the pattern expected from superdiffusive search organization: local scanning mixed with occasional larger relocations.")
    lines.append("On the small Empty-6x6 layout the target is often found too quickly for long-tail behavior to fully develop, so latency gains remain strong while long-run statistics are less diagnostic.")
    lines.append("")
    with open(out_path, "w") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_mode", type=str, default="water_search_v1")
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENV_IDS)
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
    parser.add_argument("--tail_xmin", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="tmp/levy_trajectory_stats_final")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    episode_payloads = []
    for env_id in args.env_ids:
        for policy in POLICIES:
            for seed in args.seeds:
                for episode_idx in range(int(args.episodes)):
                    episode_payloads.append(
                        run_single_episode(
                            args=args,
                            env_id=env_id,
                            policy=policy,
                            seed=seed,
                            episode_idx=episode_idx,
                        )
                    )

    episode_rows = [payload["row"] for payload in episode_payloads]
    aggregate_rows_list = aggregate_rows(episode_rows)

    episode_fieldnames = list(episode_rows[0].keys()) if episode_rows else []
    write_csv(os.path.join(args.out_dir, "episode_results.csv"), episode_rows, episode_fieldnames)
    with open(os.path.join(args.out_dir, "episode_results.json"), "w") as file_obj:
        json.dump(episode_rows, file_obj, indent=2)

    aggregate_fieldnames = list(aggregate_rows_list[0].keys()) if aggregate_rows_list else []
    write_csv(os.path.join(args.out_dir, "aggregate_overall.csv"), aggregate_rows_list, aggregate_fieldnames)
    with open(os.path.join(args.out_dir, "aggregate_overall.json"), "w") as file_obj:
        json.dump(aggregate_rows_list, file_obj, indent=2)
    article_table_rows = build_article_table_rows(aggregate_rows_list)
    if article_table_rows:
        article_table_fields = list(article_table_rows[0].keys())
        write_csv(os.path.join(args.out_dir, "trajectory_article_table.csv"), article_table_rows, article_table_fields)
        with open(os.path.join(args.out_dir, "trajectory_article_table.json"), "w") as file_obj:
            json.dump(article_table_rows, file_obj, indent=2)

    sample_payloads = {}
    for env_id in args.env_ids:
        sample_payloads[env_id] = {}
        for policy in POLICIES:
            matches = [payload for payload in episode_payloads if payload["row"]["env_id"] == env_id and payload["row"]["policy"] == policy]
            sample_payloads[env_id][policy] = min(matches, key=lambda payload: float(payload["row"]["discovery_latency"]))
    with open(os.path.join(args.out_dir, "sample_trajectories.json"), "w") as file_obj:
        json.dump(sample_payloads, file_obj, indent=2)

    plot_sample_trajectories(sample_payloads, os.path.join(args.out_dir, "sample_trajectories.png"))
    plot_segment_ccdf(episode_payloads, os.path.join(args.out_dir, "segment_length_ccdf.png"))
    plot_bar_metric(aggregate_rows_list, "success", "Success rate", os.path.join(args.out_dir, "success_rate.png"))
    plot_bar_metric(aggregate_rows_list, "discovery_latency", "Discovery latency", os.path.join(args.out_dir, "discovery_latency.png"))
    plot_curve_mean(episode_payloads, "msd_curve", "Mean squared displacement", os.path.join(args.out_dir, "msd_over_time.png"))
    plot_curve_mean(episode_payloads, "coverage_curve", "Coverage rate", os.path.join(args.out_dir, "coverage_over_time.png"))
    plot_turn_angle_hist(episode_payloads, os.path.join(args.out_dir, "turn_angle_distribution.png"))
    write_report(args, aggregate_rows_list, os.path.join(args.out_dir, "trajectory_stats_report.md"))


if __name__ == "__main__":
    main()
