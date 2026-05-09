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

from scripts.compare_songline_minigrid import run_comparison
from scripts.songline_minigrid import ensure_dir


TASK_SUITES = {
    "water": {
        "task_mode": "water_search_v1",
        "env_ids": ["MiniGrid-Empty-Random-6x6-v0"],
        "methods": [
            "random",
            "songline_graph_path",
            "external_sptm_like_patch_graph",
            "external_learned_bc_grid_obs",
            "milestone_semantic_intent_water_node_v1",
            "milestone_semantic_intent_water_concept_plan_v1",
            "milestone_state_conditioned_water_v2",
        ],
    },
    "rest": {
        "task_mode": "rest_search_v1",
        "env_ids": ["MiniGrid-Empty-Random-6x6-v0"],
        "methods": [
            "random",
            "songline_graph_path",
            "external_sptm_like_patch_graph",
            "external_learned_bc_grid_obs",
            "milestone_semantic_intent_rest_node_v1",
            "milestone_semantic_intent_rest_concept_plan_v1",
            "milestone_state_conditioned_rest_v2",
        ],
    },
    "goal_region": {
        "task_mode": "default",
        "env_ids": ["MiniGrid-Empty-Random-6x6-v0", "MiniGrid-FourRooms-v0"],
        "methods": [
            "random",
            "songline_graph_path",
            "external_sptm_like_patch_graph",
            "external_learned_bc_grid_obs",
            "milestone_semantic_intent_goal_region_node_v1",
            "milestone_semantic_intent_goal_region_concept_v1",
            "milestone_semantic_intent_goal_region_plan_v1",
        ],
    },
    "hazard_recovery": {
        "task_mode": "default",
        "env_ids": ["MiniGrid-LavaGapS7-v0"],
        "methods": [
            "random",
            "songline_graph_path",
            "external_sptm_like_patch_graph",
            "external_learned_bc_grid_obs",
            "milestone_semantic_intent_hazard_recovery_node_v1",
            "milestone_semantic_intent_hazard_recovery_concept_v1",
            "milestone_semantic_intent_hazard_recovery_plan_v1",
            "milestone_state_conditioned_hazard_recovery_v7",
        ],
    },
}

STAGE_SPECS = [
    ("query_formed_rate", "query_nonempty_rate", "Query formed"),
    ("retrieval_satisfied_rate", "query_satisfaction_rate", "Retrieval satisfied"),
    ("target_materialized_rate", "semantic_target_materialization_rate", "Target materialized"),
    ("completion_after_retrieval_rate", "post_retrieval_completion_rate", "Completion after retrieval"),
]

ASSIST_MODES = ("on", "off")


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


def load_json(path):
    with open(path, "r") as file_obj:
        return json.load(file_obj)


def assist_flags(assist_mode):
    if assist_mode == "on":
        return False, False
    if assist_mode == "off":
        return True, True
    raise ValueError(f"Unknown assist_mode: {assist_mode}")


def series_label(row, include_assist_suffix):
    if include_assist_suffix:
        return f"{row['method']} [{row['assist_mode']}]"
    return str(row["method"])


def best_row(rows):
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("success_rate", 0.0)),
            float(row.get("post_retrieval_completion_rate", 0.0)),
            float(row.get("semantic_target_materialization_rate", 0.0)),
            float(row.get("query_satisfaction_rate", 0.0)),
        ),
    )


def weakest_stage_from_row(row):
    stage_name, metric_name, _ = min(STAGE_SPECS, key=lambda item: float(row.get(item[1], 0.0)))
    return stage_name, float(row.get(metric_name, 0.0))


def run_suite(args, task_name, suite_cfg, assist_mode):
    suite_out_dir = os.path.join(args.out_dir, task_name, f"assists_{assist_mode}")
    ensure_dir(suite_out_dir)
    disable_local_resource_guidance, disable_goal_rejoin_fallback_assists = assist_flags(assist_mode)
    compare_args = SimpleNamespace(
        env_ids=list(suite_cfg["env_ids"]),
        methods=list(suite_cfg["methods"]),
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        seeds=args.seeds,
        episodes=args.episodes,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        suggest_every=args.suggest_every,
        intervention_patience=args.intervention_patience,
        min_goal_visits=args.min_goal_visits,
        top_k_goals=args.top_k_goals,
        graph_rollout_horizon=args.graph_rollout_horizon,
        token_source=args.token_source,
        scene_radius=args.scene_radius,
        disable_local_resource_guidance=disable_local_resource_guidance,
        disable_goal_rejoin_fallback_assists=disable_goal_rejoin_fallback_assists,
        task_mode=str(suite_cfg.get("task_mode", "default")),
        intent_mode="none",
        intent_selection_mode="fixed",
        intent_type="reach_safe_exit",
        intent_handoff_mode="none",
        goal_rejoin_guard_mode="none",
        goal_rejoin_guard_steps=4,
        goal_rejoin_target_mode="none",
        semantic_retrieval_mode="concept_recall_v1",
        water_success_radius=args.water_success_radius,
        rest_success_radius=args.rest_success_radius,
        thirst_on_threshold=0.10,
        thirst_off_threshold=0.04,
        water_local_activation_threshold=0.0,
        water_local_hold_threshold=0.0,
        rest_energy_on_threshold=0.95,
        rest_energy_off_threshold=0.98,
        rest_local_activation_threshold=0.0,
        rest_local_hold_threshold=0.0,
        env_change_mode=args.env_change_mode,
        change_after_episode=args.change_after_episode,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_proj_dim=args.tokenizer_proj_dim,
        out_dir=suite_out_dir,
    )
    run_comparison(compare_args)
    return suite_out_dir


def build_article_rows(task_name, suite_out_dir, assist_mode):
    aggregate_rows = load_json(os.path.join(suite_out_dir, "aggregate_overall.json"))
    rows = []
    for row in aggregate_rows:
        rows.append(
            {
                "task_name": task_name,
                "method": row["method"],
                "assist_mode": assist_mode,
                "controller_assists_enabled": int(assist_mode == "on"),
                "env_change_mode": row["env_change_mode"],
                "local_resource_guidance_enabled": int(row["local_resource_guidance_enabled"]),
                "goal_rejoin_fallback_assists_enabled": int(row["goal_rejoin_fallback_assists_enabled"]),
                "success_rate": float(row["success_rate_mean"]),
                "avg_steps": float(row["avg_steps_to_goal_mean"]),
                "avg_return": float(row["avg_return_mean"]),
                "query_nonempty_rate": float(row["query_nonempty_rate_mean"]),
                "retrieval_precision_at_k": float(row["retrieval_precision_at_k_mean"]),
                "query_satisfaction_rate": float(row["query_satisfaction_rate_mean"]),
                "semantic_target_materialization_rate": float(row["semantic_target_materialization_rate_mean"]),
                "post_retrieval_completion_rate": float(row["post_retrieval_completion_rate_mean"]),
                "completion_given_materialized": float(row["completion_given_materialized_mean"]),
                "retrieval_failure_empty_rate": float(row["retrieval_failure_empty_rate_mean"]),
                "retrieval_failure_unsatisfied_rate": float(row["retrieval_failure_unsatisfied_rate_mean"]),
                "semantic_materialization_failure_rate": float(row["semantic_materialization_failure_rate_mean"]),
                "control_failure_after_retrieval_rate": float(row["control_failure_after_retrieval_rate_mean"]),
            }
        )
    return rows


def build_failure_taxonomy_rows(task_name, suite_out_dir, assist_mode):
    episode_rows = load_json(os.path.join(suite_out_dir, "episode_results.json"))
    grouped = {}
    for row in episode_rows:
        key = (str(task_name), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    taxonomy_rows = []
    for (task, method), rows in sorted(grouped.items()):
        taxonomy_counts = {}
        for row in rows:
            label = str(row.get("failure_taxonomy", "unknown"))
            taxonomy_counts[label] = taxonomy_counts.get(label, 0) + 1
        total = float(max(1, len(rows)))
        taxonomy_rows.append(
            {
                "task_name": task,
                "method": method,
                "assist_mode": assist_mode,
                "controller_assists_enabled": int(assist_mode == "on"),
                "num_episodes": int(len(rows)),
                "success_rate": safe_mean([float(row.get("success", 0.0)) for row in rows]),
                "retrieval_failure_empty_rate": float(taxonomy_counts.get("retrieval_failure_empty", 0) / total),
                "retrieval_failure_unsatisfied_rate": float(taxonomy_counts.get("retrieval_failure_unsatisfied", 0) / total),
                "semantic_materialization_failure_rate": float(taxonomy_counts.get("semantic_materialization_failure", 0) / total),
                "control_failure_after_retrieval_rate": float(taxonomy_counts.get("control_failure_after_retrieval", 0) / total),
                "no_retrieval_attempt_rate": float(taxonomy_counts.get("no_retrieval_attempt", 0) / total),
            }
        )
    return taxonomy_rows


def build_task_family_summary(article_rows):
    rows = []
    for task_name in sorted({str(row["task_name"]) for row in article_rows}):
        match = best_row([row for row in article_rows if str(row["task_name"]) == task_name])
        if match is None:
            continue
        weakest_stage, weakest_stage_rate = weakest_stage_from_row(match)
        rows.append(
            {
                "task_name": task_name,
                "best_method": match["method"],
                "assist_mode": match["assist_mode"],
                "controller_assists_enabled": int(match["controller_assists_enabled"]),
                "success_rate": float(match["success_rate"]),
                "avg_steps": float(match["avg_steps"]),
                "query_formed_rate": float(match["query_nonempty_rate"]),
                "retrieval_satisfied_rate": float(match["query_satisfaction_rate"]),
                "target_materialized_rate": float(match["semantic_target_materialization_rate"]),
                "completion_after_retrieval_rate": float(match["post_retrieval_completion_rate"]),
                "weakest_stage": weakest_stage,
                "weakest_stage_rate": weakest_stage_rate,
            }
        )
    return rows


def build_method_diagnostics(article_rows):
    grouped = {}
    for row in article_rows:
        key = (str(row["method"]), str(row["assist_mode"]))
        grouped.setdefault(key, []).append(row)
    rows = []
    for (method, assist_mode), group in sorted(grouped.items()):
        best_task_row = best_row(group)
        mean_stage_rates = {
            stage_name: safe_mean([float(row[metric_name]) for row in group])
            for stage_name, metric_name, _ in STAGE_SPECS
        }
        weakest_stage = min(mean_stage_rates, key=mean_stage_rates.get)
        rows.append(
            {
                "method": method,
                "assist_mode": assist_mode,
                "controller_assists_enabled": int(assist_mode == "on"),
                "num_tasks": int(len(group)),
                "best_task": best_task_row["task_name"],
                "best_task_success_rate": float(best_task_row["success_rate"]),
                "avg_success_rate": safe_mean([float(row["success_rate"]) for row in group]),
                "avg_query_formed_rate": safe_mean([float(row["query_nonempty_rate"]) for row in group]),
                "avg_retrieval_satisfied_rate": safe_mean([float(row["query_satisfaction_rate"]) for row in group]),
                "avg_target_materialized_rate": safe_mean([float(row["semantic_target_materialization_rate"]) for row in group]),
                "avg_completion_after_retrieval_rate": safe_mean([float(row["post_retrieval_completion_rate"]) for row in group]),
                "weakest_stage": weakest_stage,
                "weakest_stage_rate": float(mean_stage_rates[weakest_stage]),
            }
        )
    return rows


def build_assist_comparison_rows(article_rows):
    grouped = {}
    for row in article_rows:
        key = (str(row["task_name"]), str(row["method"]))
        grouped.setdefault(key, {})[str(row["assist_mode"])] = row
    rows = []
    for (task_name, method), pair in sorted(grouped.items()):
        row_on = pair.get("on")
        row_off = pair.get("off")
        if row_on is None or row_off is None:
            continue
        rows.append(
            {
                "task_name": task_name,
                "method": method,
                "success_rate_on": float(row_on["success_rate"]),
                "success_rate_off": float(row_off["success_rate"]),
                "success_rate_delta_on_minus_off": float(row_on["success_rate"] - row_off["success_rate"]),
                "query_satisfaction_rate_on": float(row_on["query_satisfaction_rate"]),
                "query_satisfaction_rate_off": float(row_off["query_satisfaction_rate"]),
                "query_satisfaction_rate_delta_on_minus_off": float(row_on["query_satisfaction_rate"] - row_off["query_satisfaction_rate"]),
                "semantic_target_materialization_rate_on": float(row_on["semantic_target_materialization_rate"]),
                "semantic_target_materialization_rate_off": float(row_off["semantic_target_materialization_rate"]),
                "semantic_target_materialization_rate_delta_on_minus_off": float(
                    row_on["semantic_target_materialization_rate"] - row_off["semantic_target_materialization_rate"]
                ),
                "post_retrieval_completion_rate_on": float(row_on["post_retrieval_completion_rate"]),
                "post_retrieval_completion_rate_off": float(row_off["post_retrieval_completion_rate"]),
                "post_retrieval_completion_rate_delta_on_minus_off": float(
                    row_on["post_retrieval_completion_rate"] - row_off["post_retrieval_completion_rate"]
                ),
            }
        )
    return rows


def build_assist_summary_rows(article_rows):
    rows = []
    for task_name in sorted({str(row["task_name"]) for row in article_rows}):
        task_rows = [row for row in article_rows if str(row["task_name"]) == task_name]
        row_on = best_row([row for row in task_rows if str(row["assist_mode"]) == "on"])
        row_off = best_row([row for row in task_rows if str(row["assist_mode"]) == "off"])
        if row_on is None and row_off is None:
            continue
        rows.append(
            {
                "task_name": task_name,
                "best_method_on": "" if row_on is None else row_on["method"],
                "best_method_off": "" if row_off is None else row_off["method"],
                "success_rate_on": 0.0 if row_on is None else float(row_on["success_rate"]),
                "success_rate_off": 0.0 if row_off is None else float(row_off["success_rate"]),
                "query_satisfaction_rate_on": 0.0 if row_on is None else float(row_on["query_satisfaction_rate"]),
                "query_satisfaction_rate_off": 0.0 if row_off is None else float(row_off["query_satisfaction_rate"]),
                "semantic_target_materialization_rate_on": 0.0 if row_on is None else float(row_on["semantic_target_materialization_rate"]),
                "semantic_target_materialization_rate_off": 0.0 if row_off is None else float(row_off["semantic_target_materialization_rate"]),
                "post_retrieval_completion_rate_on": 0.0 if row_on is None else float(row_on["post_retrieval_completion_rate"]),
                "post_retrieval_completion_rate_off": 0.0 if row_off is None else float(row_off["post_retrieval_completion_rate"]),
            }
        )
    for row in rows:
        row["success_rate_delta_on_minus_off"] = float(row["success_rate_on"] - row["success_rate_off"])
        row["query_satisfaction_rate_delta_on_minus_off"] = float(row["query_satisfaction_rate_on"] - row["query_satisfaction_rate_off"])
        row["semantic_target_materialization_rate_delta_on_minus_off"] = float(
            row["semantic_target_materialization_rate_on"] - row["semantic_target_materialization_rate_off"]
        )
        row["post_retrieval_completion_rate_delta_on_minus_off"] = float(
            row["post_retrieval_completion_rate_on"] - row["post_retrieval_completion_rate_off"]
        )
    return rows


def plot_article_metric(rows, metric, title, out_path):
    tasks = sorted({str(row["task_name"]) for row in rows})
    include_assist_suffix = len({str(row["assist_mode"]) for row in rows}) > 1
    series = sorted({series_label(row, include_assist_suffix) for row in rows})
    x = np.arange(len(tasks))
    width = 0.80 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    center = (len(series) - 1) / 2.0
    for idx, label in enumerate(series):
        vals = []
        for task in tasks:
            match = next(
                (
                    row for row in rows
                    if str(row["task_name"]) == task and series_label(row, include_assist_suffix) == label
                ),
                None,
            )
            vals.append(0.0 if match is None else float(match[metric]))
        ax.bar(x + (idx - center) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=0)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task_funnel(article_rows, task_name, out_path):
    task_rows = [row for row in article_rows if str(row["task_name"]) == str(task_name)]
    if not task_rows:
        return
    selected_rows = []
    for assist_mode in ASSIST_MODES:
        match = best_row([row for row in task_rows if str(row["assist_mode"]) == assist_mode])
        if match is not None:
            selected_rows.append(match)
    if not selected_rows:
        return

    stage_labels = [label for _, _, label in STAGE_SPECS] + ["Success"]
    x = np.arange(len(stage_labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for row in selected_rows:
        vals = [float(row[metric_name]) for _, metric_name, _ in STAGE_SPECS] + [float(row["success_rate"])]
        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2,
            label=f"{row['assist_mode']} [{row['method']}]",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title(f"{task_name} retrieval/control funnel")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_assist_comparison_grid(summary_rows, out_path):
    if not summary_rows:
        return
    tasks = [str(row["task_name"]) for row in summary_rows]
    metrics = [
        ("success_rate", "Success"),
        ("query_satisfaction_rate", "Query satisfied"),
        ("semantic_target_materialization_rate", "Target materialized"),
        ("post_retrieval_completion_rate", "Completion after retrieval"),
    ]
    x = np.arange(len(tasks))
    width = 0.34
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (metric_prefix, title) in zip(axes.ravel(), metrics):
        vals_on = [float(row[f"{metric_prefix}_on"]) for row in summary_rows]
        vals_off = [float(row[f"{metric_prefix}_off"]) for row in summary_rows]
        ax.bar(x - width / 2.0, vals_on, width=width, label="assists_on")
        ax.bar(x + width / 2.0, vals_off, width=width, label="assists_off")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASK_SUITES.keys()), choices=list(TASK_SUITES.keys()))
    parser.add_argument("--assist_modes", nargs="+", default=list(ASSIST_MODES), choices=list(ASSIST_MODES))
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--suggest_every", type=int, default=8)
    parser.add_argument("--intervention_patience", type=int, default=4)
    parser.add_argument("--min_goal_visits", type=int, default=2)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--graph_rollout_horizon", type=int, default=4)
    parser.add_argument("--token_source", type=str, default="scene_semantic", choices=["symbolic_hash", "scene_semantic", "scene_patch_hash"])
    parser.add_argument("--scene_radius", type=int, default=1)
    parser.add_argument("--water_success_radius", type=int, default=1)
    parser.add_argument("--rest_success_radius", type=int, default=1)
    parser.add_argument("--env_change_mode", type=str, default="none", choices=["none", "goal_shift_v1"])
    parser.add_argument("--change_after_episode", type=int, default=-1)
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign", choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="tmp/symbolic_memory_article_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    article_rows = []
    taxonomy_rows = []
    for task_name in args.tasks:
        for assist_mode in args.assist_modes:
            suite_out_dir = run_suite(args, task_name, TASK_SUITES[task_name], assist_mode)
            article_rows.extend(build_article_rows(task_name, suite_out_dir, assist_mode))
            taxonomy_rows.extend(build_failure_taxonomy_rows(task_name, suite_out_dir, assist_mode))

    task_family_summary_rows = build_task_family_summary(article_rows)
    method_diagnostic_rows = build_method_diagnostics(article_rows)
    assist_comparison_rows = build_assist_comparison_rows(article_rows)
    assist_summary_rows = build_assist_summary_rows(article_rows)

    article_fieldnames = [
        "task_name",
        "method",
        "assist_mode",
        "controller_assists_enabled",
        "env_change_mode",
        "local_resource_guidance_enabled",
        "goal_rejoin_fallback_assists_enabled",
        "success_rate",
        "avg_steps",
        "avg_return",
        "query_nonempty_rate",
        "retrieval_precision_at_k",
        "query_satisfaction_rate",
        "semantic_target_materialization_rate",
        "post_retrieval_completion_rate",
        "completion_given_materialized",
        "retrieval_failure_empty_rate",
        "retrieval_failure_unsatisfied_rate",
        "semantic_materialization_failure_rate",
        "control_failure_after_retrieval_rate",
    ]
    write_csv(os.path.join(args.out_dir, "article_overview.csv"), article_rows, article_fieldnames)
    with open(os.path.join(args.out_dir, "article_overview.json"), "w") as file_obj:
        json.dump(article_rows, file_obj, indent=2)

    taxonomy_fieldnames = [
        "task_name",
        "method",
        "assist_mode",
        "controller_assists_enabled",
        "num_episodes",
        "success_rate",
        "retrieval_failure_empty_rate",
        "retrieval_failure_unsatisfied_rate",
        "semantic_materialization_failure_rate",
        "control_failure_after_retrieval_rate",
        "no_retrieval_attempt_rate",
    ]
    write_csv(os.path.join(args.out_dir, "article_failure_taxonomy.csv"), taxonomy_rows, taxonomy_fieldnames)
    with open(os.path.join(args.out_dir, "article_failure_taxonomy.json"), "w") as file_obj:
        json.dump(taxonomy_rows, file_obj, indent=2)

    task_family_summary_fieldnames = [
        "task_name",
        "best_method",
        "assist_mode",
        "controller_assists_enabled",
        "success_rate",
        "avg_steps",
        "query_formed_rate",
        "retrieval_satisfied_rate",
        "target_materialized_rate",
        "completion_after_retrieval_rate",
        "weakest_stage",
        "weakest_stage_rate",
    ]
    write_csv(os.path.join(args.out_dir, "article_task_family_summary.csv"), task_family_summary_rows, task_family_summary_fieldnames)
    with open(os.path.join(args.out_dir, "article_task_family_summary.json"), "w") as file_obj:
        json.dump(task_family_summary_rows, file_obj, indent=2)

    method_diagnostic_fieldnames = [
        "method",
        "assist_mode",
        "controller_assists_enabled",
        "num_tasks",
        "best_task",
        "best_task_success_rate",
        "avg_success_rate",
        "avg_query_formed_rate",
        "avg_retrieval_satisfied_rate",
        "avg_target_materialized_rate",
        "avg_completion_after_retrieval_rate",
        "weakest_stage",
        "weakest_stage_rate",
    ]
    write_csv(os.path.join(args.out_dir, "article_method_diagnostics.csv"), method_diagnostic_rows, method_diagnostic_fieldnames)
    with open(os.path.join(args.out_dir, "article_method_diagnostics.json"), "w") as file_obj:
        json.dump(method_diagnostic_rows, file_obj, indent=2)

    assist_comparison_fieldnames = [
        "task_name",
        "method",
        "success_rate_on",
        "success_rate_off",
        "success_rate_delta_on_minus_off",
        "query_satisfaction_rate_on",
        "query_satisfaction_rate_off",
        "query_satisfaction_rate_delta_on_minus_off",
        "semantic_target_materialization_rate_on",
        "semantic_target_materialization_rate_off",
        "semantic_target_materialization_rate_delta_on_minus_off",
        "post_retrieval_completion_rate_on",
        "post_retrieval_completion_rate_off",
        "post_retrieval_completion_rate_delta_on_minus_off",
    ]
    write_csv(os.path.join(args.out_dir, "article_assist_comparison.csv"), assist_comparison_rows, assist_comparison_fieldnames)
    with open(os.path.join(args.out_dir, "article_assist_comparison.json"), "w") as file_obj:
        json.dump(assist_comparison_rows, file_obj, indent=2)

    assist_summary_fieldnames = [
        "task_name",
        "best_method_on",
        "best_method_off",
        "success_rate_on",
        "success_rate_off",
        "success_rate_delta_on_minus_off",
        "query_satisfaction_rate_on",
        "query_satisfaction_rate_off",
        "query_satisfaction_rate_delta_on_minus_off",
        "semantic_target_materialization_rate_on",
        "semantic_target_materialization_rate_off",
        "semantic_target_materialization_rate_delta_on_minus_off",
        "post_retrieval_completion_rate_on",
        "post_retrieval_completion_rate_off",
        "post_retrieval_completion_rate_delta_on_minus_off",
    ]
    write_csv(os.path.join(args.out_dir, "article_assist_summary.csv"), assist_summary_rows, assist_summary_fieldnames)
    with open(os.path.join(args.out_dir, "article_assist_summary.json"), "w") as file_obj:
        json.dump(assist_summary_rows, file_obj, indent=2)

    plot_article_metric(
        article_rows,
        metric="success_rate",
        title="Article Benchmark Success Rate",
        out_path=os.path.join(args.out_dir, "article_success_rate.png"),
    )
    plot_article_metric(
        article_rows,
        metric="query_satisfaction_rate",
        title="Article Benchmark Query Satisfaction Rate",
        out_path=os.path.join(args.out_dir, "article_query_satisfaction_rate.png"),
    )
    plot_article_metric(
        article_rows,
        metric="semantic_target_materialization_rate",
        title="Article Benchmark Target Materialization Rate",
        out_path=os.path.join(args.out_dir, "article_target_materialization_rate.png"),
    )
    plot_article_metric(
        article_rows,
        metric="post_retrieval_completion_rate",
        title="Article Benchmark Post-Retrieval Completion Rate",
        out_path=os.path.join(args.out_dir, "article_post_retrieval_completion_rate.png"),
    )
    for task_name in args.tasks:
        plot_task_funnel(
            article_rows,
            task_name=task_name,
            out_path=os.path.join(args.out_dir, f"article_funnel_{task_name}.png"),
        )
    plot_assist_comparison_grid(
        assist_summary_rows,
        out_path=os.path.join(args.out_dir, "article_assist_on_off_comparison.png"),
    )


if __name__ == "__main__":
    main()
