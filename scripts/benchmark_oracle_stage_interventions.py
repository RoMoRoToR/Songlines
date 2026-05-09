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
    "goal_region": {
        "env_id": "MiniGrid-FourRooms-v0",
        "method": "milestone_semantic_intent_goal_region_concept_v1",
        "task_mode": "default",
    },
    "hazard_recovery": {
        "env_id": "MiniGrid-LavaGapS7-v0",
        "method": "milestone_state_conditioned_hazard_recovery_v7",
        "task_mode": "default",
    },
}

REGIMES = [
    ("base", {}),
    ("oracle_retrieval", {"oracle_retrieval": True}),
    ("oracle_materialization", {"oracle_materialization": True}),
    ("oracle_controller", {"oracle_controller": True}),
]

PLOT_REGIME_LABELS = {
    "base": "Base",
    "oracle_retrieval": "Oracle retrieval",
    "oracle_materialization": "Oracle materialization",
    "oracle_controller": "Oracle controller",
}

REGIME_COLORS = {
    "base": "#7a7a7a",
    "oracle_retrieval": "#2f6db3",
    "oracle_materialization": "#e88d1c",
    "oracle_controller": "#b23a48",
}

TASK_DISPLAY_NAMES = {
    "goal_region": "Goal region",
    "hazard_recovery": "Hazard recovery",
}


def safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def bootstrap_ci(values, num_bootstrap, bootstrap_seed, ci=95.0):
    values = [float(v) for v in values]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.RandomState(int(bootstrap_seed))
    arr = np.asarray(values, dtype=np.float64)
    draws = []
    for _ in range(int(num_bootstrap)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        draws.append(float(np.mean(sample)))
    alpha = 0.5 * (100.0 - float(ci))
    return float(np.percentile(draws, alpha)), float(np.percentile(draws, 100.0 - alpha))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_run_args(args, task_name, task_cfg, regime_name, regime_flags, seed):
    cfg = method_to_config(task_cfg["method"])
    run_out_dir = os.path.join(args.out_dir, task_name, regime_name, f"seed_{seed}")
    ensure_dir(run_out_dir)
    return SimpleNamespace(
        env_id=str(task_cfg["env_id"]),
        task_mode=str(task_cfg.get("task_mode", "default")),
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
        tokenizer_mode=str(args.tokenizer_mode),
        tokenizer_proj_dim=int(args.tokenizer_proj_dim),
        out_dir=run_out_dir,
        early_hazard_intervention=cfg.get("early_hazard_intervention", False),
        commit_to_corridor=cfg.get("commit_to_corridor", False),
        debug_trace=False,
        debug_trace_env_filter="",
        record_demo=False,
        demo_episode=1,
        demo_frame_stride=1,
        demo_fps=3,
        oracle_retrieval=bool(regime_flags.get("oracle_retrieval", False)),
        oracle_materialization=bool(regime_flags.get("oracle_materialization", False)),
        oracle_controller=bool(regime_flags.get("oracle_controller", False)),
    )


def aggregate_seed_rows(seed_rows, num_bootstrap, bootstrap_seed):
    grouped = {}
    for row in seed_rows:
        key = (row["task_name"], row["regime"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (task_name, regime), rows in sorted(grouped.items()):
        success_vals = [float(item["success_rate"]) for item in rows]
        semantic_path_vals = [float(item["semantic_path_completion"]) for item in rows]
        retrieval_vals = [float(item["query_satisfaction_rate"]) for item in rows]
        materialized_vals = [float(item["semantic_target_materialization_rate"]) for item in rows]
        oracle_retrieval_activation_vals = [float(item["oracle_retrieval_activation_rate"]) for item in rows]
        oracle_materialization_activation_vals = [float(item["oracle_materialization_activation_rate"]) for item in rows]
        oracle_controller_activation_vals = [float(item["oracle_controller_activation_rate"]) for item in rows]
        success_ci_low, success_ci_high = bootstrap_ci(success_vals, num_bootstrap, bootstrap_seed + hash((task_name, regime, "success")) % 100000)
        path_ci_low, path_ci_high = bootstrap_ci(semantic_path_vals, num_bootstrap, bootstrap_seed + hash((task_name, regime, "path")) % 100000)
        aggregate_rows.append(
            {
                "task_name": str(task_name),
                "regime": str(regime),
                "env_id": str(rows[0]["env_id"]),
                "method": str(rows[0]["method"]),
                "num_seeds": int(len(rows)),
                "episodes_per_seed": int(rows[0]["episodes"]),
                "success_rate": safe_mean(success_vals),
                "success_ci_low": success_ci_low,
                "success_ci_high": success_ci_high,
                "semantic_path_completion": safe_mean(semantic_path_vals),
                "semantic_path_ci_low": path_ci_low,
                "semantic_path_ci_high": path_ci_high,
                "query_satisfaction_rate": safe_mean(retrieval_vals),
                "semantic_target_materialization_rate": safe_mean(materialized_vals),
                "oracle_retrieval_activation_rate": safe_mean(oracle_retrieval_activation_vals),
                "oracle_materialization_activation_rate": safe_mean(oracle_materialization_activation_vals),
                "oracle_controller_activation_rate": safe_mean(oracle_controller_activation_vals),
            }
        )
    return aggregate_rows


def select_case_studies(episode_rows):
    cases = []

    def pick(predicate):
        for row in episode_rows:
            if predicate(row):
                return row
        return None

    cases.append(
        (
            "goal_base_failure",
            pick(
                lambda row: row["task_name"] == "goal_region"
                and row["regime"] == "base"
                and int(row["success"]) == 0
            ),
        )
    )
    cases.append(
        (
            "goal_oracle_materialized_failure",
            pick(
                lambda row: row["task_name"] == "goal_region"
                and row["regime"] == "oracle_materialization"
                and int(row["success"]) == 0
                and int(row.get("semantic_target_materialized_any", 0)) == 1
            ),
        )
    )
    cases.append(
        (
            "hazard_base_failure",
            pick(
                lambda row: row["task_name"] == "hazard_recovery"
                and row["regime"] == "base"
                and int(row["success"]) == 0
                and float(row.get("query_satisfaction_rate", 0.0)) > 0.0
            ),
        )
    )
    cases.append(
        (
            "hazard_oracle_retrieval_success",
            pick(
                lambda row: row["task_name"] == "hazard_recovery"
                and row["regime"] == "oracle_retrieval"
                and int(row["success"]) == 1
            ),
        )
    )
    return [(name, row) for name, row in cases if row is not None]


def plot_case_studies(case_rows, out_path):
    if not case_rows:
        return
    stages = ["Q", "R", "M", "C"]
    fig, axes = plt.subplots(len(case_rows), 1, figsize=(10, 1.8 * len(case_rows)), squeeze=False)
    color_map = {"Q": "#2f6db3", "R": "#38a169", "M": "#e88d1c", "C": "#b23a48"}

    for ax, (case_name, row) in zip(axes[:, 0], case_rows):
        reached = {
            "Q": float(row.get("query_nonempty_rate", 0.0)) > 0.0 or int(row.get("query_attempt_count", 0)) > 0,
            "R": float(row.get("query_satisfaction_rate", 0.0)) > 0.0,
            "M": int(row.get("semantic_target_materialized_any", 0)) == 1 or float(row.get("semantic_target_materialization_rate", 0.0)) > 0.0,
            "C": int(row.get("post_retrieval_completion", 0)) == 1 or int(row.get("success", 0)) == 1,
        }
        for idx, stage in enumerate(stages):
            facecolor = color_map[stage] if reached[stage] else "#eeeeee"
            edgecolor = "#333333"
            rect = plt.Rectangle((idx, 0.1), 0.85, 0.8, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2)
            ax.add_patch(rect)
            ax.text(idx + 0.425, 0.50, stage, ha="center", va="center", fontsize=10, color="white" if reached[stage] else "#333333", fontweight="bold")
        title = (
            f"{row['task_name']} | {row['regime']} | seed {row['seed']} ep {row['episode']} | "
            f"success={int(row['success'])} materialized={int(row.get('semantic_target_materialized_any', 0))}"
        )
        ax.text(4.05, 0.50, title, ha="left", va="center", fontsize=9)
        ax.set_xlim(-0.1, 8.4)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_case_studies_markdown(case_rows, out_path):
    if not case_rows:
        return
    lines = [
        "# Oracle Case Studies",
        "",
        "| Case | Task | Regime | Seed | Episode | Query | Retrieval | Materialized | Post | Success | Oracle retrieval | Oracle materialization | Oracle controller |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_name, row in case_rows:
        lines.append(
            "| {case} | {task} | {regime} | {seed} | {episode} | {q:.2f} | {r:.2f} | {m:.2f} | {post:d} | {success:d} | {or_r:d} | {or_m:d} | {or_c:d} |".format(
                case=case_name,
                task=row["task_name"],
                regime=row["regime"],
                seed=int(row["seed"]),
                episode=int(row["episode"]),
                q=float(row.get("query_nonempty_rate", 0.0)),
                r=float(row.get("query_satisfaction_rate", 0.0)),
                m=float(row.get("semantic_target_materialization_rate", 0.0)),
                post=int(row.get("post_retrieval_completion", 0)),
                success=int(row.get("success", 0)),
                or_r=int(row.get("oracle_retrieval_activated", 0)),
                or_m=int(row.get("oracle_materialization_activated", 0)),
                or_c=int(row.get("oracle_controller_activated", 0)),
            )
        )
    with open(out_path, "w") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def plot_oracle_interventions(rows, out_path):
    task_order = ["goal_region", "hazard_recovery"]
    regime_order = [name for name, _ in REGIMES]
    metrics = [
        ("success_rate", "Success rate"),
        ("semantic_path_completion", "Semantic-path completion"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(len(task_order))
    width = 0.18
    center = (len(regime_order) - 1) / 2.0

    for ax, (metric_key, title) in zip(axes, metrics):
        for idx, regime_name in enumerate(regime_order):
            means = []
            err_low = []
            err_high = []
            for task_name in task_order:
                row = next(
                    (
                        item for item in rows
                        if item["task_name"] == task_name and item["regime"] == regime_name
                    ),
                    None,
                )
                mean = 0.0 if row is None else float(row[metric_key])
                ci_low = mean if row is None else float(row[f"{metric_key.replace('rate', 'ci_low')}" if metric_key == "success_rate" else "semantic_path_ci_low"])
                ci_high = mean if row is None else float(row[f"{metric_key.replace('rate', 'ci_high')}" if metric_key == "success_rate" else "semantic_path_ci_high"])
                means.append(mean)
                err_low.append(max(0.0, mean - ci_low))
                err_high.append(max(0.0, ci_high - mean))
            offsets = x + (idx - center) * width
            bars = ax.bar(
                offsets,
                means,
                width=width,
                label=PLOT_REGIME_LABELS[regime_name],
                color=REGIME_COLORS[regime_name],
                yerr=np.asarray([err_low, err_high]),
                capsize=3,
            )
            for bar, value in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    min(1.02, value + 0.03),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_DISPLAY_NAMES[name] for name in task_order])
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="tmp/oracle_stage_interventions_20260430")
    parser.add_argument("--seed_start", type=int, default=2)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--suggest_every", type=int, default=8)
    parser.add_argument("--intervention_patience", type=int, default=4)
    parser.add_argument("--min_goal_visits", type=int, default=2)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--graph_rollout_horizon", type=int, default=4)
    parser.add_argument("--token_source", type=str, default="symbolic_hash")
    parser.add_argument("--scene_radius", type=int, default=2)
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
    parser.add_argument("--change_after_episode", type=int, default=-1)
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign")
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--num_bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap_seed", type=int, default=123)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
    seed_rows = []
    episode_rows = []

    for task_name, task_cfg in TASKS.items():
        for regime_name, regime_flags in REGIMES:
            for seed in seeds:
                run_args = build_run_args(args, task_name, task_cfg, regime_name, regime_flags, seed)
                print(f"[oracle] task={task_name} regime={regime_name} seed={seed}")
                run_summary, summary = run_songline_experiment(run_args, export_outputs=True, verbose=False)
                seed_rows.append(
                    {
                        "task_name": str(task_name),
                        "regime": str(regime_name),
                        "env_id": str(task_cfg["env_id"]),
                        "method": str(task_cfg["method"]),
                        "seed": int(seed),
                        "episodes": int(args.episodes),
                        "success_rate": float(summary["success_rate"]),
                        "semantic_path_completion": float(summary["post_retrieval_completion_rate"]),
                        "query_satisfaction_rate": float(summary["query_satisfaction_rate"]),
                        "semantic_target_materialization_rate": float(summary["semantic_target_materialization_rate"]),
                        "oracle_retrieval_activation_rate": float(summary.get("oracle_retrieval_activation_rate", 0.0)),
                        "oracle_materialization_activation_rate": float(summary.get("oracle_materialization_activation_rate", 0.0)),
                        "oracle_controller_activation_rate": float(summary.get("oracle_controller_activation_rate", 0.0)),
                    }
                )
                for item in run_summary["episode_metrics"]:
                    stamped = dict(item)
                    stamped["task_name"] = str(task_name)
                    stamped["regime"] = str(regime_name)
                    stamped["env_id"] = str(task_cfg["env_id"])
                    stamped["method"] = str(task_cfg["method"])
                    episode_rows.append(stamped)

    aggregate_rows = aggregate_seed_rows(seed_rows, args.num_bootstrap, args.bootstrap_seed)
    case_rows = select_case_studies(episode_rows)
    analysis_dir = os.path.join(args.out_dir, "analysis")
    ensure_dir(analysis_dir)

    with open(os.path.join(analysis_dir, "oracle_stage_seed_rows.json"), "w") as file_obj:
        json.dump(seed_rows, file_obj, indent=2)
    with open(os.path.join(analysis_dir, "oracle_stage_episode_rows.json"), "w") as file_obj:
        json.dump(episode_rows, file_obj, indent=2)
    with open(os.path.join(analysis_dir, "oracle_stage_aggregate.json"), "w") as file_obj:
        json.dump(aggregate_rows, file_obj, indent=2)
    with open(os.path.join(analysis_dir, "oracle_case_studies.json"), "w") as file_obj:
        json.dump([{"case_name": name, "episode": row} for name, row in case_rows], file_obj, indent=2)

    write_csv(
        os.path.join(analysis_dir, "oracle_stage_aggregate.csv"),
        aggregate_rows,
        [
            "task_name",
            "regime",
            "env_id",
            "method",
            "num_seeds",
            "episodes_per_seed",
            "success_rate",
            "success_ci_low",
            "success_ci_high",
            "semantic_path_completion",
            "semantic_path_ci_low",
            "semantic_path_ci_high",
            "query_satisfaction_rate",
            "semantic_target_materialization_rate",
            "oracle_retrieval_activation_rate",
            "oracle_materialization_activation_rate",
            "oracle_controller_activation_rate",
        ],
    )
    plot_oracle_interventions(aggregate_rows, os.path.join(analysis_dir, "oracle_stage_interventions.png"))
    plot_case_studies(case_rows, os.path.join(analysis_dir, "oracle_case_studies.png"))
    write_case_studies_markdown(case_rows, os.path.join(analysis_dir, "oracle_case_studies.md"))


if __name__ == "__main__":
    main()
