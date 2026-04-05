import argparse
import csv
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from scripts.songline_minigrid import ensure_dir, run_songline_experiment


DEFAULT_ENV_IDS = [
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-LavaGapS7-v0",
]
DEFAULT_METHODS = [
    "random",
    "greedy",
    "greedy_episodic",
    "songline_no_override",
    "songline_subgoal_controller",
    "songline_graph_path",
    "milestone_semantic_handoff_v1",
    "milestone_semantic_handoff_v1_adaptive_graph",
    "milestone_semantic_handoff_v1_plus_final_exit",
    "milestone_semantic_intent_safe_exit_v1",
    "milestone_state_conditioned_intent_v1",
]


def safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def safe_std(values):
    if not values:
        return 0.0
    return float(np.std(values))


def slope(values):
    if len(values) <= 1:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def growth_stats(graph_nodes):
    if len(graph_nodes) <= 1:
        return {
            "graph_growth_slope_first_half": 0.0,
            "graph_growth_slope_second_half": 0.0,
            "graph_growth_slowdown": 0.0,
        }

    mid = max(1, len(graph_nodes) // 2)
    first = graph_nodes[: mid + 1]
    second = graph_nodes[mid:]
    first_slope = slope(first)
    second_slope = slope(second)
    return {
        "graph_growth_slope_first_half": first_slope,
        "graph_growth_slope_second_half": second_slope,
        "graph_growth_slowdown": first_slope - second_slope,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for key, group in grouped.items():
        out = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        out["num_runs"] = len(group)
        for metric in metric_keys:
            vals = [float(item[metric]) for item in group]
            out[f"{metric}_mean"] = safe_mean(vals)
            out[f"{metric}_std"] = safe_std(vals)
        aggregated.append(out)
    return aggregated


def plot_metric_by_env(rows, env_ids, methods, metric, ylabel, out_path):
    x = np.arange(len(env_ids))
    width = 0.12 if len(methods) > 4 else 0.2
    fig, ax = plt.subplots(figsize=(10, 4.8))

    center = (len(methods) - 1) / 2.0
    for idx, method in enumerate(methods):
        means = []
        stds = []
        for env_id in env_ids:
            match = next(
                (
                    row for row in rows
                    if row["env_id"] == env_id and row["method"] == method
                ),
                None,
            )
            means.append(0.0 if match is None else float(match[f"{metric}_mean"]))
            stds.append(0.0 if match is None else float(match[f"{metric}_std"]))
        ax.bar(
            x + (idx - center) * width,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(env_ids, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Environment")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_songline_growth(songline_run_rows, out_path):
    grouped = {}
    for row in songline_run_rows:
        grouped.setdefault(row["method"], {}).setdefault(row["env_id"], []).append(row)

    if not grouped:
        return

    num_rows = len(grouped)
    fig, axes = plt.subplots(num_rows, 1, figsize=(9, 3.5 * num_rows), squeeze=False)
    for ax, method in zip(axes[:, 0], sorted(grouped.keys())):
        for env_id, runs in sorted(grouped[method].items()):
            max_len = max(len(run["graph_nodes_curve"]) for run in runs)
            node_mat = []
            for run in runs:
                nodes = list(run["graph_nodes_curve"])
                while len(nodes) < max_len:
                    nodes.append(nodes[-1] if nodes else 0)
                node_mat.append(nodes)
            node_mean = np.mean(np.asarray(node_mat, dtype=np.float64), axis=0)
            episodes = np.arange(1, max_len + 1)
            ax.plot(episodes, node_mean, label=env_id)
        ax.set_title(f"{method} node growth")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Nodes")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_by_method(rows, methods, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(methods))
    means = []
    stds = []
    for method in methods:
        match = next((row for row in rows if row["method"] == method), None)
        means.append(0.0 if match is None else float(match[f"{metric}_mean"]))
        stds.append(0.0 if match is None else float(match[f"{metric}_std"]))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def method_to_config(method):
    if method == "random":
        return {"agent_mode": "random", "songline_policy": "subgoal_controller"}
    if method == "greedy":
        return {"agent_mode": "greedy", "songline_policy": "subgoal_controller"}
    if method == "greedy_episodic":
        return {"agent_mode": "greedy_episodic", "songline_policy": "subgoal_controller"}
    if method == "songline_no_override":
        return {"agent_mode": "songline", "songline_policy": "no_override"}
    if method == "songline_subgoal_controller":
        return {"agent_mode": "songline", "songline_policy": "subgoal_controller"}
    if method == "songline_graph_path":
        return {"agent_mode": "songline", "songline_policy": "graph_path"}
    if method == "milestone_semantic_handoff_v1":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "token_source": "scene_semantic",
            "milestone_mode": "semantic_handoff_v1",
            "early_hazard_intervention": True,
            "final_exit_mode": "none",
            "graph_update_mode": "static",
        }
    if method == "milestone_semantic_handoff_v1_adaptive_graph":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "token_source": "scene_semantic",
            "milestone_mode": "semantic_handoff_v1",
            "early_hazard_intervention": True,
            "final_exit_mode": "none",
            "graph_update_mode": "adaptive",
        }
    if method == "milestone_semantic_handoff_v1_plus_final_exit":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "token_source": "scene_semantic",
            "milestone_mode": "semantic_handoff_v1",
            "early_hazard_intervention": True,
            "final_exit_mode": "v1",
            "graph_update_mode": "static",
        }
    if method == "milestone_semantic_intent_safe_exit_v1":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "token_source": "scene_semantic",
            "milestone_mode": "semantic_handoff_v1",
            "early_hazard_intervention": True,
            "final_exit_mode": "none",
            "graph_update_mode": "static",
            "intent_mode": "safe_exit_v1",
            "intent_type": "reach_safe_exit",
        }
    if method == "milestone_state_conditioned_intent_v1":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "token_source": "scene_semantic",
            "milestone_mode": "semantic_handoff_v1",
            "early_hazard_intervention": True,
            "final_exit_mode": "none",
            "graph_update_mode": "static",
            "intent_mode": "safe_exit_v1",
            "intent_selection_mode": "state_v1",
            "intent_type": "reach_safe_exit",
        }
    raise ValueError(f"Unknown method: {method}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENV_IDS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=120)
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
    parser.add_argument("--scene_radius", type=int, default=1)
    parser.add_argument("--intent_mode", type=str, default="none", choices=["none", "safe_exit_v1", "goal_region_v1"])
    parser.add_argument("--intent_selection_mode", type=str, default="fixed", choices=["fixed", "state_v1"])
    parser.add_argument(
        "--intent_type",
        type=str,
        default="reach_safe_exit",
        choices=["reach_safe_exit", "find_goal_region", "find_water_source"],
    )
    parser.add_argument("--env_change_mode", type=str, default="none", choices=["none", "goal_shift_v1"])
    parser.add_argument("--change_after_episode", type=int, default=-1)
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign", choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="tmp/songline_minigrid_compare")
    return parser.parse_args()


def run_comparison(args):
    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.num_seeds))
    ensure_dir(args.out_dir)

    episode_rows = []
    run_rows = []
    songline_growth_rows = []

    for env_id in args.env_ids:
        for seed in seeds:
            for method in args.methods:
                cfg = method_to_config(method)
                run_out_dir = os.path.join(args.out_dir, env_id, f"seed_{seed}", method)
                run_args = SimpleNamespace(
                    env_id=env_id,
                    agent_mode=cfg["agent_mode"],
                    songline_policy=cfg["songline_policy"],
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    seed=seed,
                    epsilon=args.epsilon,
                    suggest_every=args.suggest_every,
                    intervention_patience=args.intervention_patience,
                    min_goal_visits=args.min_goal_visits,
                    top_k_goals=args.top_k_goals,
                    graph_rollout_horizon=args.graph_rollout_horizon,
                    token_source=cfg.get("token_source", args.token_source),
                    milestone_mode=cfg.get("milestone_mode", "none"),
                    final_exit_mode=cfg.get("final_exit_mode", "none"),
                    graph_update_mode=cfg.get("graph_update_mode", "static"),
                    intent_mode=cfg.get("intent_mode", getattr(args, "intent_mode", "none")),
                    intent_selection_mode=cfg.get("intent_selection_mode", getattr(args, "intent_selection_mode", "fixed")),
                    intent_type=cfg.get("intent_type", getattr(args, "intent_type", "reach_safe_exit")),
                    env_change_mode=args.env_change_mode,
                    change_after_episode=args.change_after_episode,
                    export_phase_metrics=True,
                    scene_radius=args.scene_radius,
                    tokenizer_mode=args.tokenizer_mode,
                    tokenizer_proj_dim=args.tokenizer_proj_dim,
                    out_dir=run_out_dir,
                    early_hazard_intervention=cfg.get("early_hazard_intervention", False),
                    commit_to_corridor=cfg.get("commit_to_corridor", False),
                    debug_trace=False,
                    debug_trace_env_filter="",
                )

                print(f"[run] env={env_id} seed={seed} method={method}")
                run_summary, summary = run_songline_experiment(run_args, export_outputs=True, verbose=False)

                for item in run_summary["episode_metrics"]:
                    episode_rows.append(item)

                run_row = {
                    "env_id": env_id,
                    "seed": seed,
                    "method": method,
                    "token_source": cfg.get("token_source", args.token_source),
                    "graph_update_mode": cfg.get("graph_update_mode", "static"),
                    "intent_mode": cfg.get("intent_mode", getattr(args, "intent_mode", "none")),
                    "intent_selection_mode": cfg.get("intent_selection_mode", getattr(args, "intent_selection_mode", "fixed")),
                    "intent_type": cfg.get("intent_type", getattr(args, "intent_type", "reach_safe_exit")),
                    "env_change_mode": args.env_change_mode,
                    "change_after_episode": int(args.change_after_episode),
                    "agent_mode": summary["agent_mode"],
                    "songline_policy": summary["songline_policy"],
                    "success_rate": float(summary["success_rate"]),
                    "avg_steps_to_goal": float(summary["avg_steps_to_goal"]),
                    "avg_return": float(summary["avg_return"]),
                    "success_rate_pre_change": float(summary["success_rate_pre_change"]),
                    "success_rate_post_change": float(summary["success_rate_post_change"]),
                    "success_rate_change_delta": float(summary["success_rate_change_delta"]),
                    "intervention_rate": float(summary["intervention_rate"]),
                    "plan_hit_rate": float(summary["plan_hit_rate"]),
                    "graph_nodes": float(summary["graph_nodes"]),
                    "graph_edges": float(summary["graph_edges"]),
                    "phrase_length_mean": float(summary["phrase_length_mean"]),
                    "subgoal_reach_rate": float(summary["subgoal_reach_rate"]),
                    "goal_distance_delta_per_intervention": float(summary["goal_distance_delta_per_intervention"]),
                    "node_reuse_rate": float(summary["node_reuse_rate"]),
                    "new_nodes_per_episode": float(summary["new_nodes_per_episode"]),
                    "graph_path_length": float(summary["graph_path_length"]),
                    "fraction_gap_aligned": float(summary["fraction_gap_aligned"]),
                    "fraction_safe_crossing": float(summary["fraction_safe_crossing"]),
                    "fraction_post_hazard": float(summary["fraction_post_hazard"]),
                    "fraction_final_exit_maneuver": float(summary["fraction_final_exit_maneuver"]),
                    "fraction_resume_to_goal": float(summary["fraction_resume_to_goal"]),
                    "fraction_post_hazard_progress": float(summary["fraction_post_hazard_progress"]),
                    "fraction_resume_to_goal_progress": float(summary["fraction_resume_to_goal_progress"]),
                    "fraction_post_hazard_to_success": float(summary["fraction_post_hazard_to_success"]),
                    "fraction_resume_to_goal_to_success": float(summary["fraction_resume_to_goal_to_success"]),
                    "conditional_post_hazard_success": float(summary["conditional_post_hazard_success"]),
                    "conditional_resume_to_goal_success": float(summary["conditional_resume_to_goal_success"]),
                    "mean_max_phase_depth": float(summary["mean_max_phase_depth"]),
                }
                run_row.update(growth_stats(run_summary["graph_nodes"]))
                run_rows.append(run_row)

                if summary["agent_mode"] == "songline":
                    songline_growth_rows.append(
                        {
                            "env_id": env_id,
                            "seed": seed,
                            "method": method,
                            "graph_nodes_curve": list(run_summary["graph_nodes"]),
                        }
                    )

    episode_fieldnames = [
        "episode",
        "env_id",
        "change_active",
        "env_change_mode",
        "change_after_episode",
        "intent_mode",
        "intent_selection_mode",
        "intent_type",
        "intent_active",
        "active_intent_type",
        "agent_task_phase",
        "agent_thirst",
        "agent_energy",
        "agent_risk_budget",
        "agent_mode",
        "songline_policy",
        "method",
        "token_source",
        "seed",
        "return",
        "steps_to_goal",
        "steps",
        "success",
        "intervention_rate",
        "plan_hit_rate",
        "graph_nodes",
        "graph_edges",
        "phrase_length_mean",
        "subgoal_reach_rate",
        "goal_distance_delta_per_intervention",
        "node_reuse_rate",
        "new_nodes_per_episode",
        "graph_path_length",
        "has_gap_aligned",
        "has_safe_crossing",
        "has_post_hazard",
        "has_final_exit_maneuver",
        "has_resume_to_goal",
        "has_post_hazard_progress",
        "has_resume_to_goal_progress",
        "post_hazard_to_success",
        "resume_to_goal_to_success",
        "max_phase_depth",
    ]
    write_csv(os.path.join(args.out_dir, "episode_results.csv"), episode_rows, episode_fieldnames)
    with open(os.path.join(args.out_dir, "episode_results.json"), "w") as f:
        json.dump(episode_rows, f, indent=2)

    run_metric_keys = [
        "success_rate",
        "avg_steps_to_goal",
        "avg_return",
        "success_rate_pre_change",
        "success_rate_post_change",
        "success_rate_change_delta",
        "intervention_rate",
        "plan_hit_rate",
        "graph_nodes",
        "graph_edges",
        "phrase_length_mean",
        "subgoal_reach_rate",
        "goal_distance_delta_per_intervention",
        "node_reuse_rate",
        "new_nodes_per_episode",
        "graph_path_length",
        "fraction_gap_aligned",
        "fraction_safe_crossing",
        "fraction_post_hazard",
        "fraction_final_exit_maneuver",
        "fraction_resume_to_goal",
        "fraction_post_hazard_progress",
        "fraction_resume_to_goal_progress",
        "fraction_post_hazard_to_success",
        "fraction_resume_to_goal_to_success",
        "conditional_post_hazard_success",
        "conditional_resume_to_goal_success",
        "mean_max_phase_depth",
        "graph_growth_slope_first_half",
        "graph_growth_slope_second_half",
        "graph_growth_slowdown",
    ]
    run_fieldnames = [
        "env_id",
        "seed",
        "method",
        "token_source",
        "graph_update_mode",
        "intent_mode",
        "intent_selection_mode",
        "intent_type",
        "env_change_mode",
        "change_after_episode",
        "agent_mode",
        "songline_policy",
    ] + run_metric_keys
    write_csv(os.path.join(args.out_dir, "run_results.csv"), run_rows, run_fieldnames)
    with open(os.path.join(args.out_dir, "run_results.json"), "w") as f:
        json.dump(run_rows, f, indent=2)

    aggregated_by_env = aggregate_rows(
        run_rows,
        [
            "env_id",
            "method",
            "token_source",
            "graph_update_mode",
            "intent_mode",
            "intent_selection_mode",
            "intent_type",
            "env_change_mode",
            "change_after_episode",
        ],
        run_metric_keys,
    )
    aggregated_overall = aggregate_rows(
        run_rows,
        ["method", "token_source", "graph_update_mode", "intent_mode", "intent_selection_mode", "intent_type", "env_change_mode", "change_after_episode"],
        run_metric_keys,
    )

    agg_fieldnames = [
        "env_id",
        "method",
        "token_source",
        "graph_update_mode",
        "intent_mode",
        "intent_selection_mode",
        "intent_type",
        "env_change_mode",
        "change_after_episode",
        "num_runs",
    ]
    for metric in run_metric_keys:
        agg_fieldnames.append(f"{metric}_mean")
        agg_fieldnames.append(f"{metric}_std")

    write_csv(os.path.join(args.out_dir, "aggregate_by_env.csv"), aggregated_by_env, agg_fieldnames)
    with open(os.path.join(args.out_dir, "aggregate_by_env.json"), "w") as f:
        json.dump(aggregated_by_env, f, indent=2)

    overall_fieldnames = [
        "method",
        "token_source",
        "graph_update_mode",
        "intent_mode",
        "intent_selection_mode",
        "intent_type",
        "env_change_mode",
        "change_after_episode",
        "num_runs",
    ]
    for metric in run_metric_keys:
        overall_fieldnames.append(f"{metric}_mean")
        overall_fieldnames.append(f"{metric}_std")
    write_csv(os.path.join(args.out_dir, "aggregate_overall.csv"), aggregated_overall, overall_fieldnames)
    with open(os.path.join(args.out_dir, "aggregate_overall.json"), "w") as f:
        json.dump(aggregated_overall, f, indent=2)

    summary_table_rows = []
    for row in aggregated_overall:
        summary_table_rows.append(
            {
                "Method": row["method"],
                "Token source": row["token_source"],
                "Graph update": row["graph_update_mode"],
                "Intent mode": row["intent_mode"],
                "Intent selection": row["intent_selection_mode"],
                "Intent type": row["intent_type"],
                "Env change": row["env_change_mode"],
                "Change after": row["change_after_episode"],
                "Success rate": row["success_rate_mean"],
                "Pre-change success": row["success_rate_pre_change_mean"],
                "Post-change success": row["success_rate_post_change_mean"],
                "Success delta": row["success_rate_change_delta_mean"],
                "Avg steps": row["avg_steps_to_goal_mean"],
                "Avg return": row["avg_return_mean"],
                "Phase depth": row["mean_max_phase_depth_mean"],
                "Gap aligned frac": row["fraction_gap_aligned_mean"],
                "Safe crossing frac": row["fraction_safe_crossing_mean"],
                "Post hazard frac": row["fraction_post_hazard_mean"],
                "Resume to goal frac": row["fraction_resume_to_goal_mean"],
                "Post hazard progress frac": row["fraction_post_hazard_progress_mean"],
                "Resume progress frac": row["fraction_resume_to_goal_progress_mean"],
                "Cond post hazard success": row["conditional_post_hazard_success_mean"],
                "Cond resume success": row["conditional_resume_to_goal_success_mean"],
                "Std success": row["success_rate_std"],
                "Std steps": row["avg_steps_to_goal_std"],
                "Std return": row["avg_return_std"],
            }
        )
    write_csv(
        os.path.join(args.out_dir, "summary_table.csv"),
        summary_table_rows,
        [
            "Method",
            "Token source",
            "Graph update",
            "Intent mode",
            "Intent selection",
            "Intent type",
            "Env change",
            "Change after",
            "Success rate",
            "Pre-change success",
            "Post-change success",
            "Success delta",
            "Avg steps",
            "Avg return",
            "Phase depth",
            "Gap aligned frac",
            "Safe crossing frac",
            "Post hazard frac",
            "Resume to goal frac",
            "Post hazard progress frac",
            "Resume progress frac",
            "Cond post hazard success",
            "Cond resume success",
            "Std success",
            "Std steps",
            "Std return",
        ],
    )
    with open(os.path.join(args.out_dir, "summary_table.json"), "w") as f:
        json.dump(summary_table_rows, f, indent=2)

    plot_metric_by_env(
        aggregated_by_env,
        args.env_ids,
        args.methods,
        metric="success_rate",
        ylabel="Success Rate",
        out_path=os.path.join(args.out_dir, "comparison_success_rate.png"),
    )
    plot_metric_by_env(
        aggregated_by_env,
        args.env_ids,
        args.methods,
        metric="success_rate_pre_change",
        ylabel="Pre-change Success Rate",
        out_path=os.path.join(args.out_dir, "comparison_success_rate_pre_change.png"),
    )
    plot_metric_by_env(
        aggregated_by_env,
        args.env_ids,
        args.methods,
        metric="success_rate_post_change",
        ylabel="Post-change Success Rate",
        out_path=os.path.join(args.out_dir, "comparison_success_rate_post_change.png"),
    )
    plot_metric_by_env(
        aggregated_by_env,
        args.env_ids,
        args.methods,
        metric="success_rate_change_delta",
        ylabel="Success Delta (Post - Pre)",
        out_path=os.path.join(args.out_dir, "comparison_success_rate_change_delta.png"),
    )
    plot_metric_by_env(
        aggregated_by_env,
        args.env_ids,
        args.methods,
        metric="avg_steps_to_goal",
        ylabel="Avg Steps to Goal",
        out_path=os.path.join(args.out_dir, "comparison_avg_steps.png"),
    )
    plot_metric_by_env(
        aggregated_by_env,
        args.env_ids,
        args.methods,
        metric="avg_return",
        ylabel="Avg Return",
        out_path=os.path.join(args.out_dir, "comparison_avg_return.png"),
    )
    plot_songline_growth(songline_growth_rows, os.path.join(args.out_dir, "songline_graph_growth.png"))
    plot_metric_by_method(
        aggregated_overall,
        args.methods,
        metric="mean_max_phase_depth",
        ylabel="Phase Depth",
        out_path=os.path.join(args.out_dir, "comparison_phase_depth.png"),
    )
    plot_metric_by_method(
        aggregated_overall,
        args.methods,
        metric="fraction_safe_crossing",
        ylabel="Safe Crossing Fraction",
        out_path=os.path.join(args.out_dir, "comparison_safe_crossing_fraction.png"),
    )

    print("\nOverall summary:")
    for row in summary_table_rows:
        print(
            f"{row['Method']}: "
            f"token_source={row['Token source']} | "
            f"graph_update={row['Graph update']} | "
            f"intent={row['Intent mode']}:{row['Intent selection']}:{row['Intent type']} | "
            f"env_change={row['Env change']}@{row['Change after']} | "
            f"success={row['Success rate']:.3f} +- {row['Std success']:.3f}, "
            f"delta={row['Success delta']:.3f}, "
            f"steps={row['Avg steps']:.3f} +- {row['Std steps']:.3f}, "
            f"return={row['Avg return']:.3f} +- {row['Std return']:.3f}"
        )


def main():
    args = parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
