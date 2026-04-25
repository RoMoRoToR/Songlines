import argparse
import csv
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from scripts.songline_miniworld import run_songline_miniworld_experiment
from scripts.songline_minigrid import ensure_dir


DEFAULT_ENV_IDS = [
    "MiniWorld-Hallway-v0",
    "MiniWorld-TMaze-v0",
    "MiniWorld-WallGap-v0",
    "MiniWorld-FourRooms-v0",
]

DEFAULT_METHODS = [
    "random",
    "greedy",
    "miniworld_goal_region_node_v1",
    "miniworld_goal_region_concept_v1",
    "miniworld_goal_region_plan_v1",
]


def safe_mean(values):
    return 0.0 if not values else float(np.mean(values))


def safe_std(values):
    return 0.0 if not values else float(np.std(values))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def method_to_config(method):
    if method == "random":
        return {"agent_mode": "random", "songline_policy": "graph_path"}
    if method == "greedy":
        return {"agent_mode": "greedy", "songline_policy": "graph_path"}
    if method == "miniworld_goal_region_node_v1":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "intent_mode": "goal_region_v1",
            "intent_type": "find_goal_region",
            "semantic_retrieval_mode": "node_only",
        }
    if method == "miniworld_goal_region_concept_v1":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "intent_mode": "goal_region_v1",
            "intent_type": "find_goal_region",
            "semantic_retrieval_mode": "concept_recall_v1",
        }
    if method == "miniworld_goal_region_plan_v1":
        return {
            "agent_mode": "songline",
            "songline_policy": "graph_path",
            "intent_mode": "goal_region_v1",
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


def plot_metric(rows, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [row["method"] for row in rows]
    means = [float(row[f"{metric}_mean"]) for row in rows]
    stds = [float(row[f"{metric}_std"]) for row in rows]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENV_IDS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--suggest_every", type=int, default=8)
    parser.add_argument("--min_goal_visits", type=int, default=2)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--graph_rollout_horizon", type=int, default=4)
    parser.add_argument("--token_source", type=str, default="scene_semantic", choices=["symbolic_hash", "scene_semantic", "scene_patch_hash"])
    parser.add_argument("--graph_update_mode", type=str, default="static", choices=["static", "adaptive"])
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign", choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="tmp/songline_miniworld_compare")
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
                run_args = SimpleNamespace(
                    env_id=env_id,
                    agent_mode=cfg["agent_mode"],
                    songline_policy=cfg["songline_policy"],
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    seed=seed,
                    suggest_every=args.suggest_every,
                    min_goal_visits=args.min_goal_visits,
                    top_k_goals=args.top_k_goals,
                    graph_rollout_horizon=args.graph_rollout_horizon,
                    token_source=args.token_source,
                    graph_update_mode=args.graph_update_mode,
                    intent_mode=cfg.get("intent_mode", "none"),
                    intent_selection_mode="fixed",
                    intent_type=cfg.get("intent_type", "find_goal_region"),
                    semantic_retrieval_mode=cfg.get("semantic_retrieval_mode", "concept_recall_v1"),
                    debug_trace=False,
                    record_demo=False,
                    demo_episode=1,
                    demo_fps=3,
                    tokenizer_mode=args.tokenizer_mode,
                    tokenizer_proj_dim=args.tokenizer_proj_dim,
                    out_dir=run_out_dir,
                    check_dependencies=False,
                    list_envs=False,
                )
                print(f"[run] env={env_id} seed={seed} method={method}")
                run_summary, summary = run_songline_miniworld_experiment(run_args, export_outputs=True, verbose=False)
                for item in run_summary["episode_metrics"]:
                    episode_rows.append(item)
                run_rows.append(
                    {
                        "env_id": env_id,
                        "seed": seed,
                        "method": method,
                        "agent_mode": summary["agent_mode"],
                        "songline_policy": summary["songline_policy"],
                        "token_source": args.token_source,
                        "graph_update_mode": args.graph_update_mode,
                        "intent_mode": cfg.get("intent_mode", "none"),
                        "intent_type": cfg.get("intent_type", "find_goal_region"),
                        "semantic_retrieval_mode": cfg.get("semantic_retrieval_mode", "concept_recall_v1"),
                        "success_rate": float(summary["success_rate"]),
                        "avg_steps_to_goal": float(summary["avg_steps_to_goal"]),
                        "avg_return": float(summary["avg_return"]),
                        "intervention_rate": float(summary["intervention_rate"]),
                        "plan_hit_rate": float(summary["plan_hit_rate"]),
                        "graph_nodes": float(summary["graph_nodes"]),
                        "graph_edges": float(summary["graph_edges"]),
                        "phrase_length_mean": float(summary["phrase_length_mean"]),
                        "subgoal_reach_rate": float(summary["subgoal_reach_rate"]),
                        "node_reuse_rate": float(summary["node_reuse_rate"]),
                        "new_nodes_per_episode": float(summary["new_nodes_per_episode"]),
                        "graph_path_length": float(summary["graph_path_length"]),
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
        "intervention_rate",
        "plan_hit_rate",
        "graph_nodes",
        "graph_edges",
        "phrase_length_mean",
        "subgoal_reach_rate",
        "node_reuse_rate",
        "new_nodes_per_episode",
        "graph_path_length",
    ]
    run_fieldnames = [
        "env_id",
        "seed",
        "method",
        "agent_mode",
        "songline_policy",
        "token_source",
        "graph_update_mode",
        "intent_mode",
        "intent_type",
        "semantic_retrieval_mode",
    ] + run_metric_keys
    write_csv(os.path.join(args.out_dir, "run_results.csv"), run_rows, run_fieldnames)
    with open(os.path.join(args.out_dir, "run_results.json"), "w") as f:
        json.dump(run_rows, f, indent=2)

    aggregated_overall = aggregate_rows(
        run_rows,
        ["method", "token_source", "graph_update_mode", "intent_mode", "intent_type", "semantic_retrieval_mode"],
        run_metric_keys,
    )
    overall_fieldnames = [
        "method",
        "token_source",
        "graph_update_mode",
        "intent_mode",
        "intent_type",
        "semantic_retrieval_mode",
        "num_runs",
    ]
    for metric in run_metric_keys:
        overall_fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(os.path.join(args.out_dir, "aggregate_overall.csv"), aggregated_overall, overall_fieldnames)

    summary_table_rows = []
    for row in aggregated_overall:
        summary_table_rows.append(
            {
                "Method": row["method"],
                "Semantic retrieval": row["semantic_retrieval_mode"],
                "Success rate": row["success_rate_mean"],
                "Avg steps": row["avg_steps_to_goal_mean"],
                "Avg return": row["avg_return_mean"],
                "Graph nodes": row["graph_nodes_mean"],
                "Graph edges": row["graph_edges_mean"],
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
            "Semantic retrieval",
            "Success rate",
            "Avg steps",
            "Avg return",
            "Graph nodes",
            "Graph edges",
            "Std success",
            "Std steps",
            "Std return",
        ],
    )
    with open(os.path.join(args.out_dir, "summary_table.json"), "w") as f:
        json.dump(summary_table_rows, f, indent=2)

    plot_metric(aggregated_overall, "success_rate", "Success Rate", os.path.join(args.out_dir, "success_rate.png"))
    plot_metric(aggregated_overall, "avg_steps_to_goal", "Average Steps", os.path.join(args.out_dir, "avg_steps.png"))
    plot_metric(aggregated_overall, "avg_return", "Average Return", os.path.join(args.out_dir, "avg_return.png"))


def main():
    args = parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
