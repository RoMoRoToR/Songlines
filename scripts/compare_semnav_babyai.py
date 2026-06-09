import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.songline_minigrid import ensure_dir, run_songline_experiment


DEFAULT_ENV_IDS = [
    "BabyAI-GoToObjMaze-v0",
]

DEFAULT_METHODS = [
    "random",
    "greedy",
    "babyai_semantic_node_v1",
    "babyai_semantic_plan_v1",
]


def safe_mean(values):
    return 0.0 if not values else float(np.mean(values))


def safe_std(values):
    return 0.0 if not values else float(np.std(values))


def write_csv(path, rows, fieldnames):
    extras = []
    seen = set(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                extras.append(key)
    fieldnames = list(fieldnames) + extras
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def method_to_config(method):
    if method == "random":
        return {
            "agent_mode": "random",
            "songline_policy": "graph_path",
            "intent_mode": "babyai_mission_v1",
            "intent_type": "find_goal_region",
            "semantic_retrieval_mode": "node_only",
        }
    if method == "greedy":
        return {
            "agent_mode": "greedy",
            "songline_policy": "graph_path",
            "intent_mode": "babyai_mission_v1",
            "intent_type": "find_goal_region",
            "semantic_retrieval_mode": "node_only",
        }
    if method == "babyai_semantic_node_v1":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "intent_mode": "babyai_mission_v1",
            "intent_type": "find_goal_region",
            "semantic_retrieval_mode": "node_only",
        }
    if method == "babyai_semantic_plan_v1":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "intent_mode": "babyai_mission_v1",
            "intent_type": "find_goal_region",
            "semantic_retrieval_mode": "concept_plan_v1",
        }
    raise ValueError(f"Unknown method: {method}")


def aggregate_rows(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out_rows = []
    for key, group in grouped.items():
        out = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        out["num_runs"] = len(group)
        for metric in metric_keys:
            vals = [float(item[metric]) for item in group]
            out[f"{metric}_mean"] = safe_mean(vals)
            out[f"{metric}_std"] = safe_std(vals)
        out_rows.append(out)
    return out_rows


def plot_success(rows, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = [row["method"] for row in rows]
    means = [float(row["success_rate_mean"]) for row in rows]
    stds = [float(row["success_rate_std"]) for row in rows]
    x = np.arange(len(labels))
    colors = ["#9AA5B1", "#74828F", "#3B82F6", "#1D4ED8"]
    ax.bar(x, means, yerr=stds, capsize=3, color=colors[: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, max(1.0, max(means) + 0.1))
    ax.set_title("BabyAI GoToObjMaze success by method")
    for idx, mean in enumerate(means):
        ax.text(idx, mean + 0.02, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_stage_profile(rows, out_path):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    stage_keys = [
        ("query_nonempty_rate_mean", "Q"),
        ("query_satisfaction_rate_mean", "R"),
        ("semantic_target_materialization_rate_mean", "M"),
        ("post_retrieval_completion_rate_mean", "C"),
    ]
    x = np.arange(len(stage_keys))
    for row in rows:
        if not str(row["method"]).startswith("babyai_semantic_"):
            continue
        y = [float(row[key]) for key, _ in stage_keys]
        ax.plot(x, y, marker="o", linewidth=2, label=row["method"])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in stage_keys])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Operational stage rate")
    ax.set_title("BabyAI semantic stage profile")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENV_IDS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--suggest_every", type=int, default=8)
    parser.add_argument("--min_goal_visits", type=int, default=2)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--graph_rollout_horizon", type=int, default=4)
    parser.add_argument("--token_source", type=str, default="scene_semantic", choices=["symbolic_hash", "scene_semantic", "scene_patch_hash"])
    parser.add_argument("--graph_update_mode", type=str, default="static", choices=["static", "adaptive"])
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign", choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="tmp/songline_babyai_compare")
    return parser.parse_args()


def run_comparison(args):
    ensure_dir(args.out_dir)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    episode_rows = []
    run_rows = []

    for env_id in args.env_ids:
        for seed in seeds:
            for method in args.methods:
                cfg = method_to_config(method)
                run_out_dir = os.path.join(args.out_dir, env_id, f"seed_{seed}", method)
                ensure_dir(run_out_dir)
                run_args = SimpleNamespace(
                    env_id=env_id,
                    task_mode="babyai_semantic_v1",
                    agent_mode=cfg["agent_mode"],
                    songline_policy=cfg["songline_policy"],
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    seed=seed,
                    epsilon=0.15,
                    suggest_every=args.suggest_every,
                    intervention_patience=4,
                    min_goal_visits=args.min_goal_visits,
                    top_k_goals=args.top_k_goals,
                    graph_rollout_horizon=args.graph_rollout_horizon,
                    token_source=args.token_source,
                    milestone_mode="none",
                    final_exit_mode="none",
                    graph_update_mode=args.graph_update_mode,
                    intent_mode=cfg["intent_mode"],
                    intent_selection_mode="fixed",
                    intent_type=cfg["intent_type"],
                    intent_handoff_mode="none",
                    goal_rejoin_guard_mode="none",
                    goal_rejoin_guard_steps=4,
                    goal_rejoin_target_mode="none",
                    semantic_retrieval_mode=cfg["semantic_retrieval_mode"],
                    env_change_mode="none",
                    change_after_episode=-1,
                    scene_radius=1,
                    export_phase_metrics=False,
                    water_success_radius=1,
                    rest_success_radius=1,
                    thirst_on_threshold=0.10,
                    thirst_off_threshold=0.04,
                    water_local_activation_threshold=0.0,
                    water_local_hold_threshold=0.0,
                    rest_energy_on_threshold=0.95,
                    rest_energy_off_threshold=0.98,
                    rest_local_activation_threshold=0.0,
                    rest_local_hold_threshold=0.0,
                    early_hazard_intervention=False,
                    commit_to_corridor=False,
                    disable_local_resource_guidance=False,
                    disable_goal_rejoin_fallback_assists=False,
                    oracle_retrieval=False,
                    oracle_materialization=False,
                    oracle_controller=False,
                    semantic_tag_dropout_prob=0.0,
                    semantic_tag_false_positive_prob=0.0,
                    semantic_tag_false_positive_value=0.35,
                    debug_trace=False,
                    debug_trace_env_filter="",
                    record_demo=False,
                    demo_episode=1,
                    demo_frame_stride=1,
                    demo_fps=3,
                    intent_replan_cooldown_steps=4,
                    tokenizer_mode=args.tokenizer_mode,
                    tokenizer_proj_dim=args.tokenizer_proj_dim,
                    out_dir=run_out_dir,
                )
                print(f"[run] env={env_id} seed={seed} method={method}")
                run_summary, summary = run_songline_experiment(run_args, export_outputs=True, verbose=False)
                for item in run_summary["episode_metrics"]:
                    episode_rows.append(item)
                run_rows.append(
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "method": method,
                        "agent_mode": summary["agent_mode"],
                        "songline_policy": summary["songline_policy"],
                        "task_mode": summary["task_mode"],
                        "intent_mode": summary["intent_mode"],
                        "intent_type": summary["intent_type"],
                        "semantic_retrieval_mode": summary["semantic_retrieval_mode"],
                        "token_source": args.token_source,
                        "graph_update_mode": args.graph_update_mode,
                        "success_rate": float(summary["success_rate"]),
                        "avg_steps_to_goal": float(summary["avg_steps_to_goal"]),
                        "avg_return": float(summary["avg_return"]),
                        "query_nonempty_rate": float(summary["query_nonempty_rate"]),
                        "query_satisfaction_rate": float(summary["query_satisfaction_rate"]),
                        "semantic_target_materialization_rate": float(summary["semantic_target_materialization_rate"]),
                        "post_retrieval_completion_rate": float(summary["post_retrieval_completion_rate"]),
                        "graph_nodes": float(summary["graph_nodes"]),
                        "graph_edges": float(summary["graph_edges"]),
                    }
                )

    episode_fieldnames = sorted(set().union(*(row.keys() for row in episode_rows))) if episode_rows else []
    if episode_fieldnames:
        write_csv(os.path.join(args.out_dir, "episode_results.csv"), episode_rows, episode_fieldnames)
    with open(os.path.join(args.out_dir, "episode_results.json"), "w") as f:
        json.dump(episode_rows, f, indent=2)

    run_metric_keys = [
        "success_rate",
        "avg_steps_to_goal",
        "avg_return",
        "query_nonempty_rate",
        "query_satisfaction_rate",
        "semantic_target_materialization_rate",
        "post_retrieval_completion_rate",
        "graph_nodes",
        "graph_edges",
    ]
    run_fieldnames = [
        "env_id",
        "seed",
        "method",
        "agent_mode",
        "songline_policy",
        "task_mode",
        "intent_mode",
        "intent_type",
        "semantic_retrieval_mode",
        "token_source",
        "graph_update_mode",
    ] + run_metric_keys
    write_csv(os.path.join(args.out_dir, "run_results.csv"), run_rows, run_fieldnames)
    with open(os.path.join(args.out_dir, "run_results.json"), "w") as f:
        json.dump(run_rows, f, indent=2)

    aggregated_overall = aggregate_rows(
        run_rows,
        ["env_id", "method", "task_mode", "intent_mode", "intent_type", "semantic_retrieval_mode"],
        run_metric_keys,
    )
    overall_fieldnames = [
        "env_id",
        "method",
        "task_mode",
        "intent_mode",
        "intent_type",
        "semantic_retrieval_mode",
        "num_runs",
    ]
    for metric in run_metric_keys:
        overall_fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(os.path.join(args.out_dir, "aggregate_overall.csv"), aggregated_overall, overall_fieldnames)
    with open(os.path.join(args.out_dir, "aggregate_overall.json"), "w") as f:
        json.dump(aggregated_overall, f, indent=2)

    summary_table = []
    for row in aggregated_overall:
        summary_table.append(
            {
                "env_id": row["env_id"],
                "method": row["method"],
                "success": round(float(row["success_rate_mean"]), 4),
                "steps": round(float(row["avg_steps_to_goal_mean"]), 4),
                "query": round(float(row["query_nonempty_rate_mean"]), 4),
                "retrieval": round(float(row["query_satisfaction_rate_mean"]), 4),
                "materialized": round(float(row["semantic_target_materialization_rate_mean"]), 4),
                "post": round(float(row["post_retrieval_completion_rate_mean"]), 4),
            }
        )
    write_csv(
        os.path.join(args.out_dir, "summary_table.csv"),
        summary_table,
        ["env_id", "method", "success", "steps", "query", "retrieval", "materialized", "post"],
    )
    with open(os.path.join(args.out_dir, "summary_table.json"), "w") as f:
        json.dump(summary_table, f, indent=2)

    plot_success(aggregated_overall, os.path.join(args.out_dir, "babyai_success.png"))
    plot_stage_profile(aggregated_overall, os.path.join(args.out_dir, "babyai_stage_profile.png"))


def main():
    args = parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
