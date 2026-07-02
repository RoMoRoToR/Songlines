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

from scripts.compare_songline_minigrid import method_to_config
from scripts.songline_minigrid import ensure_dir, run_songline_experiment


TASKS = {
    "water": {
        "env_id": "MiniGrid-Empty-Random-6x6-v0",
        "task_mode": "water_search_v1",
        "method": "milestone_semantic_intent_water_concept_plan_v1",
    },
    "rest": {
        "env_id": "MiniGrid-Empty-Random-6x6-v0",
        "task_mode": "rest_search_v1",
        "method": "milestone_semantic_intent_rest_concept_plan_v1",
    },
    "goal_region": {
        "env_id": "MiniGrid-FourRooms-v0",
        "task_mode": "default",
        "method": "milestone_semantic_intent_goal_region_concept_v1",
    },
    "hazard_recovery": {
        "env_id": "MiniGrid-LavaGapS7-v0",
        "task_mode": "default",
        "method": "milestone_state_conditioned_hazard_recovery_v7",
    },
}

PERTURBATIONS = {
    "clean": {"semantic_tag_dropout_prob": 0.0, "semantic_tag_false_positive_prob": 0.0},
    "tag_dropout": {"semantic_tag_false_positive_prob": 0.0},
    "false_positive": {"semantic_tag_dropout_prob": 0.0},
}

PLOT_COLORS = {
    "clean": "#5f5f5f",
    "tag_dropout": "#2f6db3",
    "false_positive": "#b23a48",
}

STAGE_METRICS = [
    ("query_nonempty_rate", "Query formed"),
    ("query_satisfaction_rate", "Retrieval"),
    ("semantic_target_materialization_rate", "Materialized"),
    ("post_retrieval_completion_rate", "Post"),
]


def safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def bootstrap_ci(values, seed, num_bootstrap=2000):
    values = [float(v) for v in values]
    if not values:
        return 0.0, 0.0, 0.0
    mean = safe_mean(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.RandomState(int(seed))
    arr = np.asarray(values, dtype=np.float64)
    draws = []
    for _ in range(int(num_bootstrap)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        draws.append(float(np.mean(sample)))
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_run_args(args, task_name, task_cfg, perturbation_name, level, seed):
    cfg = method_to_config(task_cfg["method"])
    out_dir = os.path.join(
        args.out_dir,
        task_name,
        perturbation_name,
        f"level_{str(level).replace('.', 'p')}",
        f"seed_{seed}",
    )
    ensure_dir(out_dir)
    run_args = SimpleNamespace(
        env_id=str(task_cfg["env_id"]),
        task_mode=str(task_cfg["task_mode"]),
        agent_mode=cfg["agent_mode"],
        songline_policy=cfg["songline_policy"],
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(seed),
        epsilon=float(args.epsilon),
        suggest_every=int(args.suggest_every),
        intervention_patience=int(args.intervention_patience),
        min_goal_visits=int(args.min_goal_visits),
        top_k_goals=int(args.top_k_goals),
        graph_rollout_horizon=int(args.graph_rollout_horizon),
        token_source=cfg.get("token_source", args.token_source),
        milestone_mode=cfg.get("milestone_mode", "none"),
        final_exit_mode=cfg.get("final_exit_mode", "none"),
        graph_update_mode=cfg.get("graph_update_mode", "static"),
        intent_mode=cfg.get("intent_mode", args.intent_mode),
        intent_selection_mode=cfg.get("intent_selection_mode", args.intent_selection_mode),
        intent_type=cfg.get("intent_type", args.intent_type),
        intent_handoff_mode=cfg.get("intent_handoff_mode", args.intent_handoff_mode),
        goal_rejoin_guard_mode=cfg.get("goal_rejoin_guard_mode", args.goal_rejoin_guard_mode),
        goal_rejoin_guard_steps=int(args.goal_rejoin_guard_steps),
        goal_rejoin_target_mode=cfg.get("goal_rejoin_target_mode", args.goal_rejoin_target_mode),
        semantic_retrieval_mode=cfg.get("semantic_retrieval_mode", args.semantic_retrieval_mode),
        water_success_radius=int(args.water_success_radius),
        rest_success_radius=int(args.rest_success_radius),
        thirst_on_threshold=float(cfg.get("thirst_on_threshold", args.thirst_on_threshold)),
        thirst_off_threshold=float(cfg.get("thirst_off_threshold", args.thirst_off_threshold)),
        water_local_activation_threshold=float(cfg.get("water_local_activation_threshold", args.water_local_activation_threshold)),
        water_local_hold_threshold=float(cfg.get("water_local_hold_threshold", args.water_local_hold_threshold)),
        rest_energy_on_threshold=float(cfg.get("rest_energy_on_threshold", args.rest_energy_on_threshold)),
        rest_energy_off_threshold=float(cfg.get("rest_energy_off_threshold", args.rest_energy_off_threshold)),
        rest_local_activation_threshold=float(cfg.get("rest_local_activation_threshold", args.rest_local_activation_threshold)),
        rest_local_hold_threshold=float(cfg.get("rest_local_hold_threshold", args.rest_local_hold_threshold)),
        env_change_mode=str(args.env_change_mode),
        change_after_episode=int(args.change_after_episode),
        export_phase_metrics=True,
        scene_radius=int(args.scene_radius),
        disable_local_resource_guidance=bool(args.disable_local_resource_guidance),
        disable_goal_rejoin_fallback_assists=bool(args.disable_goal_rejoin_fallback_assists),
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_proj_dim=int(args.tokenizer_proj_dim),
        out_dir=out_dir,
        early_hazard_intervention=cfg.get("early_hazard_intervention", False),
        commit_to_corridor=cfg.get("commit_to_corridor", False),
        debug_trace=False,
        debug_trace_env_filter="",
        record_demo=False,
        demo_episode=1,
        demo_frame_stride=1,
        demo_fps=3,
        semantic_tag_dropout_prob=0.0,
        semantic_tag_false_positive_prob=0.0,
        semantic_tag_false_positive_value=float(args.semantic_tag_false_positive_value),
    )
    if perturbation_name == "tag_dropout":
        run_args.semantic_tag_dropout_prob = float(level)
    elif perturbation_name == "false_positive":
        run_args.semantic_tag_false_positive_prob = float(level)
    return run_args


def plot_success(rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    task_names = list(TASKS.keys())
    for ax, task_name in zip(axes.flat, task_names):
        task_rows = [row for row in rows if row["task_name"] == task_name]
        by_mode = {}
        for row in task_rows:
            by_mode.setdefault(row["perturbation"], []).append(row)
        for perturbation_name, mode_rows in by_mode.items():
            mode_rows = sorted(mode_rows, key=lambda item: float(item["noise_level"]))
            xs = [float(item["noise_level"]) for item in mode_rows]
            ys = [float(item["success_rate"]) for item in mode_rows]
            yerr_low = [float(item["success_rate"] - item["success_rate_ci_low"]) for item in mode_rows]
            yerr_high = [float(item["success_rate_ci_high"] - item["success_rate"]) for item in mode_rows]
            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_low, yerr_high],
                marker="o",
                linewidth=2.0,
                color=PLOT_COLORS.get(perturbation_name, "#333333"),
                label=perturbation_name.replace("_", " "),
            )
        ax.set_title(task_name.replace("_", " "))
        ax.set_xlabel("Noise level")
        ax.set_ylabel("Success")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stage(rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    task_names = list(TASKS.keys())
    stage_colors = {
        "query_nonempty_rate": "#2f6db3",
        "query_satisfaction_rate": "#2f9e44",
        "semantic_target_materialization_rate": "#e88d1c",
        "post_retrieval_completion_rate": "#b23a48",
    }
    for ax, task_name in zip(axes.flat, task_names):
        task_rows = sorted(
            [row for row in rows if row["task_name"] == task_name and row["perturbation"] == "tag_dropout"],
            key=lambda item: float(item["noise_level"]),
        )
        xs = [float(item["noise_level"]) for item in task_rows]
        for metric_name, metric_label in STAGE_METRICS:
            ys = [float(item[metric_name]) for item in task_rows]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=stage_colors[metric_name], label=metric_label)
        ax.set_title(f"{task_name.replace('_', ' ')} | dropout")
        ax.set_xlabel("Dropout level")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="tmp/semantic_noise_robustness_20260430")
    parser.add_argument("--seeds", type=int, nargs="*", default=[2, 7, 11])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--noise_levels", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.35])
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--suggest_every", type=int, default=1)
    parser.add_argument("--intervention_patience", type=int, default=2)
    parser.add_argument("--min_goal_visits", type=int, default=1)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--graph_rollout_horizon", type=int, default=6)
    parser.add_argument("--token_source", type=str, default="scene_semantic")
    parser.add_argument("--scene_radius", type=int, default=3)
    parser.add_argument("--disable_local_resource_guidance", action="store_true")
    parser.add_argument("--disable_goal_rejoin_fallback_assists", action="store_true")
    parser.add_argument("--intent_mode", type=str, default="none")
    parser.add_argument("--intent_selection_mode", type=str, default="fixed")
    parser.add_argument("--intent_type", type=str, default="reach_safe_exit")
    parser.add_argument("--intent_handoff_mode", type=str, default="none")
    parser.add_argument("--goal_rejoin_guard_mode", type=str, default="none")
    parser.add_argument("--goal_rejoin_guard_steps", type=int, default=4)
    parser.add_argument("--goal_rejoin_target_mode", type=str, default="none")
    parser.add_argument("--semantic_retrieval_mode", type=str, default="concept_recall_v1")
    parser.add_argument("--water_success_radius", type=int, default=1)
    parser.add_argument("--rest_success_radius", type=int, default=1)
    parser.add_argument("--thirst_on_threshold", type=float, default=0.10)
    parser.add_argument("--thirst_off_threshold", type=float, default=0.04)
    parser.add_argument("--water_local_activation_threshold", type=float, default=0.0)
    parser.add_argument("--water_local_hold_threshold", type=float, default=0.0)
    parser.add_argument("--rest_energy_on_threshold", type=float, default=0.95)
    parser.add_argument("--rest_energy_off_threshold", type=float, default=0.98)
    parser.add_argument("--rest_local_activation_threshold", type=float, default=0.0)
    parser.add_argument("--rest_local_hold_threshold", type=float, default=0.0)
    parser.add_argument("--env_change_mode", type=str, default="none")
    parser.add_argument("--change_after_episode", type=int, default=0)
    parser.add_argument("--tokenizer_mode", type=str, default="pca")
    parser.add_argument("--tokenizer_proj_dim", type=int, default=12)
    parser.add_argument("--semantic_tag_false_positive_value", type=float, default=0.35)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    seed_rows = []

    for task_name, task_cfg in TASKS.items():
        for perturbation_name in ("clean", "tag_dropout", "false_positive"):
            levels = [0.0] if perturbation_name == "clean" else list(args.noise_levels)
            for noise_level in levels:
                for seed in args.seeds:
                    run_args = build_run_args(args, task_name, task_cfg, perturbation_name, noise_level, seed)
                    print(f"[noise] task={task_name} perturb={perturbation_name} level={noise_level:.2f} seed={seed}")
                    _, summary = run_songline_experiment(run_args, export_outputs=True, verbose=False)
                    seed_rows.append(
                        {
                            "task_name": str(task_name),
                            "env_id": str(task_cfg["env_id"]),
                            "method": str(task_cfg["method"]),
                            "perturbation": str(perturbation_name),
                            "noise_level": float(noise_level),
                            "seed": int(seed),
                            "episodes": int(args.episodes),
                            "success_rate": float(summary["success_rate"]),
                            "query_nonempty_rate": float(summary["query_nonempty_rate"]),
                            "query_satisfaction_rate": float(summary["query_satisfaction_rate"]),
                            "semantic_target_materialization_rate": float(summary["semantic_target_materialization_rate"]),
                            "post_retrieval_completion_rate": float(summary["post_retrieval_completion_rate"]),
                        }
                    )

    grouped = {}
    for row in seed_rows:
        key = (row["task_name"], row["perturbation"], float(row["noise_level"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (task_name, perturbation_name, noise_level), rows in sorted(grouped.items()):
        out = {
            "task_name": str(task_name),
            "perturbation": str(perturbation_name),
            "noise_level": float(noise_level),
            "method": str(rows[0]["method"]),
            "env_id": str(rows[0]["env_id"]),
            "num_seeds": int(len(rows)),
            "episodes_per_seed": int(rows[0]["episodes"]),
        }
        for idx, metric_name in enumerate(
            [
                "success_rate",
                "query_nonempty_rate",
                "query_satisfaction_rate",
                "semantic_target_materialization_rate",
                "post_retrieval_completion_rate",
            ]
        ):
            vals = [float(item[metric_name]) for item in rows]
            mean, low, high = bootstrap_ci(vals, seed=1729 + idx * 1000 + int(noise_level * 1000) + len(task_name))
            out[metric_name] = mean
            out[f"{metric_name}_ci_low"] = low
            out[f"{metric_name}_ci_high"] = high
        aggregate_rows.append(out)

    analysis_dir = os.path.join(args.out_dir, "analysis")
    ensure_dir(analysis_dir)
    with open(os.path.join(analysis_dir, "noise_seed_rows.json"), "w") as file_obj:
        json.dump(seed_rows, file_obj, indent=2)
    with open(os.path.join(analysis_dir, "noise_aggregate.json"), "w") as file_obj:
        json.dump(aggregate_rows, file_obj, indent=2)
    write_csv(
        os.path.join(analysis_dir, "noise_aggregate.csv"),
        aggregate_rows,
        [
            "task_name",
            "perturbation",
            "noise_level",
            "method",
            "env_id",
            "num_seeds",
            "episodes_per_seed",
            "success_rate",
            "success_rate_ci_low",
            "success_rate_ci_high",
            "query_nonempty_rate",
            "query_nonempty_rate_ci_low",
            "query_nonempty_rate_ci_high",
            "query_satisfaction_rate",
            "query_satisfaction_rate_ci_low",
            "query_satisfaction_rate_ci_high",
            "semantic_target_materialization_rate",
            "semantic_target_materialization_rate_ci_low",
            "semantic_target_materialization_rate_ci_high",
            "post_retrieval_completion_rate",
            "post_retrieval_completion_rate_ci_low",
            "post_retrieval_completion_rate_ci_high",
        ],
    )
    plot_success(aggregate_rows, os.path.join(analysis_dir, "noise_success.png"))
    plot_stage(aggregate_rows, os.path.join(analysis_dir, "noise_stage_dropout.png"))


if __name__ == "__main__":
    main()
