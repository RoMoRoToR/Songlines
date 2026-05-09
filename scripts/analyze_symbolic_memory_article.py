import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.songline_minigrid import ensure_dir


BASELINE_METHODS = {
    "random",
    "songline_graph_path",
    "external_sptm_like_patch_graph",
    "external_learned_q_table_grid_obs",
    "external_learned_bc_grid_obs",
    "external_learned_dqn_grid_obs",
}
TASK_NAMES = ("water", "rest", "goal_region", "hazard_recovery")
ASSIST_MODES = ("on", "off")

METRIC_SPECS = [
    ("success_rate", "success"),
    ("avg_steps", "steps_to_goal"),
    ("avg_return", "return"),
    ("query_nonempty_rate", "query_nonempty_rate"),
    ("retrieval_precision_at_k", "retrieval_precision_at_k"),
    ("query_satisfaction_rate", "query_satisfaction_rate"),
    ("semantic_target_materialization_rate", "semantic_target_materialization_rate"),
    ("post_retrieval_completion_rate", "post_retrieval_completion"),
    ("completion_given_materialized", "completion_given_materialized"),
]

STAGE_METRICS = [
    ("query_nonempty_rate", "Query formed"),
    ("query_satisfaction_rate", "Retrieval satisfied"),
    ("semantic_target_materialization_rate", "Target materialized"),
    ("post_retrieval_completion_rate", "Completion after retrieval"),
]


def episode_stage_event_flags(row):
    attempts = int(row.get("query_attempt_count", 0))
    taxonomy = str(row.get("failure_taxonomy", ""))
    q_event = int(attempts > 0)
    r_event = int(q_event and taxonomy not in {"no_retrieval_attempt", "retrieval_failure_empty", "retrieval_failure_unsatisfied"})
    m_event = int(r_event and int(row.get("semantic_target_materialized_any", 0)) > 0)
    c_event = int(m_event and int(row.get("post_retrieval_completion", 0)) > 0)
    return q_event, r_event, m_event, c_event


def load_json(path):
    with open(path, "r") as file_obj:
        return json.load(file_obj)


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


def ordered_task_names(task_names):
    order = {name: idx for idx, name in enumerate(TASK_NAMES)}
    return sorted(task_names, key=lambda name: order.get(name, 999))


def bootstrap_seed_ci(seed_values, rng, num_bootstrap):
    vals = np.asarray(list(seed_values), dtype=np.float64)
    if vals.size <= 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, mean, mean
    indices = rng.randint(0, vals.size, size=(int(num_bootstrap), vals.size))
    samples = np.mean(vals[indices], axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return mean, float(low), float(high)


def holm_bonferroni(rows, p_key, out_key):
    indexed = [(idx, float(row.get(p_key, 1.0))) for idx, row in enumerate(rows)]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted = [1.0] * m
    running = 0.0
    for rank, (idx, pval) in enumerate(indexed):
        adj = min(1.0, (m - rank) * pval)
        running = max(running, adj)
        adjusted[rank] = running
    for (rank, (idx, _)) in enumerate(indexed):
        rows[idx][out_key] = float(adjusted[rank])


def collect_episode_rows(article_dir):
    rows = []
    for task_name in TASK_NAMES:
        for assist_mode in ASSIST_MODES:
            path = os.path.join(article_dir, task_name, f"assists_{assist_mode}", "episode_results.json")
            if not os.path.exists(path):
                continue
            for row in load_json(path):
                out = dict(row)
                out["task_name"] = task_name
                out["assist_mode"] = assist_mode
                out["controller_assists_enabled"] = int(assist_mode == "on")
                rows.append(out)
    return rows


def group_seed_values(episode_rows):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in episode_rows:
        key = (str(row["task_name"]), str(row["assist_mode"]), str(row["method"]))
        seed = int(row["seed"])
        for out_name, src_key in METRIC_SPECS:
            grouped[key][out_name][seed].append(float(row.get(src_key, 0.0)))
    return grouped


def build_article_rows(episode_rows, num_bootstrap, bootstrap_seed):
    rng = np.random.RandomState(int(bootstrap_seed))
    grouped = group_seed_values(episode_rows)
    rows = []
    for (task_name, assist_mode, method), metric_seed_map in sorted(grouped.items()):
        row = {
            "task_name": task_name,
            "assist_mode": assist_mode,
            "method": method,
            "is_baseline": int(method in BASELINE_METHODS),
            "controller_assists_enabled": int(assist_mode == "on"),
        }
        for out_name, _ in METRIC_SPECS:
            seed_means = [safe_mean(vals) for _, vals in sorted(metric_seed_map[out_name].items())]
            mean, low, high = bootstrap_seed_ci(seed_means, rng=rng, num_bootstrap=num_bootstrap)
            row[out_name] = float(mean)
            row[f"{out_name}_ci_low"] = float(low)
            row[f"{out_name}_ci_high"] = float(high)
            row[f"{out_name}_num_seeds"] = int(len(seed_means))
        rows.append(row)
    return rows, grouped


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


def build_task_summary(article_rows):
    rows = []
    for task_name in ordered_task_names({row["task_name"] for row in article_rows}):
        semantic_rows = [row for row in article_rows if row["task_name"] == task_name and not int(row["is_baseline"])]
        match = best_row(semantic_rows)
        if match is None:
            continue
        weakest_metric, weakest_label = min(
            STAGE_METRICS,
            key=lambda item: float(match.get(item[0], 0.0)),
        )
        rows.append(
            {
                "task_name": task_name,
                "best_method": match["method"],
                "assist_mode": match["assist_mode"],
                "success_rate": float(match["success_rate"]),
                "success_rate_ci_low": float(match["success_rate_ci_low"]),
                "success_rate_ci_high": float(match["success_rate_ci_high"]),
                "avg_steps": float(match["avg_steps"]),
                "query_formed_rate": float(match["query_nonempty_rate"]),
                "retrieval_satisfied_rate": float(match["query_satisfaction_rate"]),
                "target_materialized_rate": float(match["semantic_target_materialization_rate"]),
                "completion_after_retrieval_rate": float(match["post_retrieval_completion_rate"]),
                "weakest_stage": weakest_label,
                "weakest_stage_rate": float(match[weakest_metric]),
            }
        )
    return rows


def build_assist_significance(article_rows, grouped_seed_metrics):
    rows = []
    for task_name in ordered_task_names({row["task_name"] for row in article_rows}):
        task_rows = [row for row in article_rows if row["task_name"] == task_name and not int(row["is_baseline"])]
        methods = sorted({row["method"] for row in task_rows})
        best_method = None
        best_score = -1.0
        for method in methods:
            on_row = next((row for row in task_rows if row["method"] == method and row["assist_mode"] == "on"), None)
            off_row = next((row for row in task_rows if row["method"] == method and row["assist_mode"] == "off"), None)
            if on_row is None or off_row is None:
                continue
            score = max(float(on_row["success_rate"]), float(off_row["success_rate"]))
            if score > best_score:
                best_score = score
                best_method = method
        if best_method is None:
            continue
        row = {
            "task_name": task_name,
            "method": best_method,
        }
        for metric_name in ["success_rate", "query_satisfaction_rate", "semantic_target_materialization_rate", "post_retrieval_completion_rate"]:
            on_key = (task_name, "on", best_method)
            off_key = (task_name, "off", best_method)
            on_seed_map = grouped_seed_metrics.get(on_key, {}).get(metric_name, {})
            off_seed_map = grouped_seed_metrics.get(off_key, {}).get(metric_name, {})
            seeds = sorted(set(on_seed_map.keys()) & set(off_seed_map.keys()))
            on_vals = [safe_mean(on_seed_map[seed]) for seed in seeds]
            off_vals = [safe_mean(off_seed_map[seed]) for seed in seeds]
            row[f"{metric_name}_on"] = safe_mean(on_vals)
            row[f"{metric_name}_off"] = safe_mean(off_vals)
            row[f"{metric_name}_delta_on_minus_off"] = float(row[f"{metric_name}_on"] - row[f"{metric_name}_off"])
            if not seeds or np.allclose(on_vals, off_vals):
                pval = 1.0
            else:
                try:
                    pval = float(wilcoxon(on_vals, off_vals, zero_method="pratt").pvalue)
                except ValueError:
                    pval = 1.0
            row[f"{metric_name}_p_value"] = float(pval)
        rows.append(row)

    for metric_name in ["success_rate", "query_satisfaction_rate", "semantic_target_materialization_rate", "post_retrieval_completion_rate"]:
        holm_bonferroni(rows, f"{metric_name}_p_value", f"{metric_name}_p_value_holm")
    return rows


def build_baseline_comparison(article_rows):
    rows = []
    for task_name in ordered_task_names({row["task_name"] for row in article_rows}):
        semantic_rows = [row for row in article_rows if row["task_name"] == task_name and not int(row["is_baseline"])]
        best_semantic = best_row(semantic_rows)
        if best_semantic is None:
            continue
        assist_mode = str(best_semantic["assist_mode"])
        random_row = next((row for row in article_rows if row["task_name"] == task_name and row["assist_mode"] == assist_mode and row["method"] == "random"), None)
        graph_row = next((row for row in article_rows if row["task_name"] == task_name and row["assist_mode"] == assist_mode and row["method"] == "songline_graph_path"), None)
        sptm_row = next((row for row in article_rows if row["task_name"] == task_name and row["assist_mode"] == assist_mode and row["method"] == "external_sptm_like_patch_graph"), None)
        learned_candidates = [
            row
            for row in article_rows
            if row["task_name"] == task_name
            and row["assist_mode"] == assist_mode
            and row["method"] in {"external_learned_bc_grid_obs", "external_learned_dqn_grid_obs", "external_learned_q_table_grid_obs"}
        ]
        learned_row = best_row(learned_candidates)
        rows.append(
            {
                "task_name": task_name,
                "assist_mode": assist_mode,
                "semantic_method": best_semantic["method"],
                "learned_method": "" if learned_row is None else str(learned_row["method"]),
                "success_random": 0.0 if random_row is None else float(random_row["success_rate"]),
                "success_graph": 0.0 if graph_row is None else float(graph_row["success_rate"]),
                "success_sptm": 0.0 if sptm_row is None else float(sptm_row["success_rate"]),
                "success_learned": 0.0 if learned_row is None else float(learned_row["success_rate"]),
                "success_semantic": float(best_semantic["success_rate"]),
                "query_random": 0.0 if random_row is None else float(random_row["query_nonempty_rate"]),
                "query_graph": 0.0 if graph_row is None else float(graph_row["query_nonempty_rate"]),
                "query_sptm": 0.0 if sptm_row is None else float(sptm_row["query_nonempty_rate"]),
                "query_learned": 0.0 if learned_row is None else float(learned_row["query_nonempty_rate"]),
                "query_semantic": float(best_semantic["query_nonempty_rate"]),
                "retrieval_random": 0.0 if random_row is None else float(random_row["query_satisfaction_rate"]),
                "retrieval_graph": 0.0 if graph_row is None else float(graph_row["query_satisfaction_rate"]),
                "retrieval_sptm": 0.0 if sptm_row is None else float(sptm_row["query_satisfaction_rate"]),
                "retrieval_learned": 0.0 if learned_row is None else float(learned_row["query_satisfaction_rate"]),
                "retrieval_semantic": float(best_semantic["query_satisfaction_rate"]),
                "materialized_random": 0.0 if random_row is None else float(random_row["semantic_target_materialization_rate"]),
                "materialized_graph": 0.0 if graph_row is None else float(graph_row["semantic_target_materialization_rate"]),
                "materialized_sptm": 0.0 if sptm_row is None else float(sptm_row["semantic_target_materialization_rate"]),
                "materialized_learned": 0.0 if learned_row is None else float(learned_row["semantic_target_materialization_rate"]),
                "materialized_semantic": float(best_semantic["semantic_target_materialization_rate"]),
                "post_random": 0.0 if random_row is None else float(random_row["post_retrieval_completion_rate"]),
                "post_graph": 0.0 if graph_row is None else float(graph_row["post_retrieval_completion_rate"]),
                "post_sptm": 0.0 if sptm_row is None else float(sptm_row["post_retrieval_completion_rate"]),
                "post_learned": 0.0 if learned_row is None else float(learned_row["post_retrieval_completion_rate"]),
                "post_semantic": float(best_semantic["post_retrieval_completion_rate"]),
            }
        )
    return rows


def build_cross_layout_rows(episode_rows, num_bootstrap, bootstrap_seed):
    rng = np.random.RandomState(int(bootstrap_seed) + 17)
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in episode_rows:
        key = (str(row["task_name"]), str(row["assist_mode"]), str(row["method"]), str(row["env_id"]))
        seed = int(row["seed"])
        for out_name, src_key in METRIC_SPECS:
            grouped[key][out_name][seed].append(float(row.get(src_key, 0.0)))
    rows = []
    for (task_name, assist_mode, method, env_id), metric_seed_map in sorted(grouped.items()):
        row = {
            "task_name": task_name,
            "assist_mode": assist_mode,
            "method": method,
            "env_id": env_id,
            "is_baseline": int(method in BASELINE_METHODS),
        }
        for out_name, _ in METRIC_SPECS:
            seed_means = [safe_mean(vals) for _, vals in sorted(metric_seed_map[out_name].items())]
            mean, low, high = bootstrap_seed_ci(seed_means, rng, num_bootstrap)
            row[out_name] = float(mean)
            row[f"{out_name}_ci_low"] = float(low)
            row[f"{out_name}_ci_high"] = float(high)
        rows.append(row)
    return rows


def build_memory_scaling_rows(episode_rows, num_bootstrap, bootstrap_seed):
    valid_rows = [row for row in episode_rows if int(row.get("graph_nodes", 0)) > 0]
    if not valid_rows:
        return []
    rng = np.random.RandomState(int(bootstrap_seed) + 29)
    graph_nodes = np.asarray([int(row["graph_nodes"]) for row in valid_rows], dtype=np.int32)
    quantiles = np.quantile(graph_nodes, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.unique(quantiles.astype(np.int32))
    if bins.size < 2:
        bins = np.asarray([int(graph_nodes.min()), int(graph_nodes.max()) + 1], dtype=np.int32)
    rows = []
    for idx in range(len(bins) - 1):
        left = int(bins[idx])
        right = int(bins[idx + 1])
        if idx == len(bins) - 2:
            bucket = [row for row in valid_rows if left <= int(row["graph_nodes"]) <= right]
        else:
            bucket = [row for row in valid_rows if left <= int(row["graph_nodes"]) < right]
        if not bucket:
            continue
        seed_groups = defaultdict(list)
        for row in bucket:
            seed_groups[int(row["seed"])].append(row)
        metric_rows = {}
        for metric_key in [
            "success",
            "query_nonempty_rate",
            "query_satisfaction_rate",
            "semantic_target_materialization_rate",
            "post_retrieval_completion",
        ]:
            seed_means = [safe_mean([float(item.get(metric_key, 0.0)) for item in seed_bucket]) for seed_bucket in seed_groups.values()]
            mean, low, high = bootstrap_seed_ci(seed_means, rng=rng, num_bootstrap=num_bootstrap)
            metric_rows[metric_key] = (mean, low, high)
        rows.append(
            {
                "bin_label": f"{left}-{right}",
                "min_graph_nodes": left,
                "max_graph_nodes": right,
                "num_episodes": int(len(bucket)),
                "num_seeds": int(len(seed_groups)),
                "success_rate": float(metric_rows["success"][0]),
                "success_rate_ci_low": float(metric_rows["success"][1]),
                "success_rate_ci_high": float(metric_rows["success"][2]),
                "query_nonempty_rate": float(metric_rows["query_nonempty_rate"][0]),
                "query_nonempty_rate_ci_low": float(metric_rows["query_nonempty_rate"][1]),
                "query_nonempty_rate_ci_high": float(metric_rows["query_nonempty_rate"][2]),
                "query_satisfaction_rate": float(metric_rows["query_satisfaction_rate"][0]),
                "query_satisfaction_rate_ci_low": float(metric_rows["query_satisfaction_rate"][1]),
                "query_satisfaction_rate_ci_high": float(metric_rows["query_satisfaction_rate"][2]),
                "semantic_target_materialization_rate": float(metric_rows["semantic_target_materialization_rate"][0]),
                "semantic_target_materialization_rate_ci_low": float(metric_rows["semantic_target_materialization_rate"][1]),
                "semantic_target_materialization_rate_ci_high": float(metric_rows["semantic_target_materialization_rate"][2]),
                "post_retrieval_completion_rate": float(metric_rows["post_retrieval_completion"][0]),
                "post_retrieval_completion_rate_ci_low": float(metric_rows["post_retrieval_completion"][1]),
                "post_retrieval_completion_rate_ci_high": float(metric_rows["post_retrieval_completion"][2]),
            }
        )
    return rows


def build_stage_consistency_rows(episode_rows, article_rows):
    rows = []
    for task_name in ordered_task_names({row["task_name"] for row in article_rows}):
        semantic_rows = [row for row in article_rows if row["task_name"] == task_name and not int(row["is_baseline"])]
        best_semantic = best_row(semantic_rows)
        if best_semantic is None:
            continue
        match_rows = [
            row
            for row in episode_rows
            if str(row["task_name"]) == str(task_name)
            and str(row["assist_mode"]) == str(best_semantic["assist_mode"])
            and str(row["method"]) == str(best_semantic["method"])
        ]
        if not match_rows:
            continue
        q_vals = []
        r_vals = []
        m_vals = []
        c_vals = []
        success_vals = []
        for row in match_rows:
            q_event, r_event, m_event, c_event = episode_stage_event_flags(row)
            q_vals.append(q_event)
            r_vals.append(r_event)
            m_vals.append(m_event)
            c_vals.append(c_event)
            success_vals.append(int(row.get("success", 0)))
        q_total = max(1, int(sum(q_vals)))
        r_total = max(1, int(sum(r_vals)))
        m_total = max(1, int(sum(m_vals)))
        p_q = float(np.mean(q_vals))
        p_r_given_q = float(sum(r_vals) / q_total)
        p_m_given_r = float(sum(m_vals) / r_total)
        p_c_given_m = float(sum(c_vals) / m_total)
        stage_product = float(p_q * p_r_given_q * p_m_given_r * p_c_given_m)
        semantic_completion_rate = float(np.mean(c_vals))
        empirical_success_rate = float(np.mean(success_vals))
        rows.append(
            {
                "task_name": task_name,
                "assist_mode": str(best_semantic["assist_mode"]),
                "method": str(best_semantic["method"]),
                "num_episodes": int(len(match_rows)),
                "empirical_success_rate": empirical_success_rate,
                "semantic_path_completion_rate": semantic_completion_rate,
                "p_q": p_q,
                "p_r_given_q": p_r_given_q,
                "p_m_given_r": p_m_given_r,
                "p_c_given_m": p_c_given_m,
                "stage_product": stage_product,
                "gap_to_semantic_completion": float(stage_product - semantic_completion_rate),
                "gap_to_overall_success": float(stage_product - empirical_success_rate),
            }
        )
    return rows


def plot_baseline_comparison(rows, out_path):
    if not rows:
        return
    rows = sorted(rows, key=lambda row: TASK_NAMES.index(str(row["task_name"])) if str(row["task_name"]) in TASK_NAMES else 999)
    tasks = [row["task_name"] for row in rows]
    x = np.arange(len(tasks))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    semantic = [float(row["success_semantic"]) for row in rows]
    graph = [float(row["success_graph"]) for row in rows]
    random = [float(row["success_random"]) for row in rows]
    sptm = [float(row["success_sptm"]) for row in rows]
    learned = [float(row["success_learned"]) for row in rows]
    bars_random = ax.bar(x - 2.0 * width, random, width=width, label="random", color="#c7c7c7")
    bars_graph = ax.bar(x - 1.0 * width, graph, width=width, label="graph-only", color="#9ecae1")
    bars_sptm = ax.bar(x, sptm, width=width, label="SPTM-like", color="#6baed6")
    bars_learned = ax.bar(x + 1.0 * width, learned, width=width, label="learned Q-table", color="#bdbdbd")
    bars_semantic = ax.bar(x + 2.0 * width, semantic, width=width, label="best semantic", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Baseline comparison")
    ax.legend()
    for bar_group in [bars_random, bars_graph, bars_sptm, bars_learned, bars_semantic]:
        for bar in bar_group:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_stage_baseline_grid(rows, out_path):
    if not rows:
        return
    tasks = [row["task_name"] for row in rows]
    metrics = [
        ("query", "Query formed"),
        ("retrieval", "Retrieval satisfied"),
        ("materialized", "Target materialized"),
        ("post", "Completion after retrieval"),
    ]
    x = np.arange(len(tasks))
    width = 0.2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (prefix, title) in zip(axes.ravel(), metrics):
        vals_random = [float(row[f"{prefix}_random"]) for row in rows]
        vals_graph = [float(row[f"{prefix}_graph"]) for row in rows]
        vals_sptm = [float(row[f"{prefix}_sptm"]) for row in rows]
        vals_semantic = [float(row[f"{prefix}_semantic"]) for row in rows]
        ax.bar(x - 1.5 * width, vals_random, width=width, label="random")
        ax.bar(x - 0.5 * width, vals_graph, width=width, label="graph-only")
        ax.bar(x + 0.5 * width, vals_sptm, width=width, label="SPTM-like")
        ax.bar(x + 1.5 * width, vals_semantic, width=width, label="best semantic")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=10, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_memory_scaling(rows, out_path):
    if not rows:
        return
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for metric, label in [
        ("query_satisfaction_rate", "Retrieval satisfied"),
        ("semantic_target_materialization_rate", "Target materialized"),
        ("post_retrieval_completion_rate", "Completion after retrieval"),
    ]:
        means = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        lows = np.asarray([float(row[f"{metric}_ci_low"]) for row in rows], dtype=np.float64)
        highs = np.asarray([float(row[f"{metric}_ci_high"]) for row in rows], dtype=np.float64)
        yerr = np.vstack([means - lows, highs - means])
        ax.errorbar(x, means, yerr=yerr, marker="o", linewidth=2, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([row["bin_label"] for row in rows], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("graph_nodes bin")
    ax.set_ylabel("rate")
    ax.set_title("Online memory-growth scaling")
    for idx, row in enumerate(rows):
        ax.text(idx, 1.02, f"n={int(row['num_episodes'])}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_layout(rows, out_path):
    focus_rows = [row for row in rows if row["task_name"] == "goal_region" and not int(row["is_baseline"])]
    if not focus_rows:
        return
    best_by_env = {}
    for env_id in sorted({row["env_id"] for row in focus_rows}):
        match = best_row([row for row in focus_rows if row["env_id"] == env_id])
        if match is not None:
            best_by_env[env_id] = match
    if not best_by_env:
        return
    env_ids = list(best_by_env.keys())
    x = np.arange(len(env_ids))
    width = 0.18
    metrics = [
        ("success_rate", "Success"),
        ("query_satisfaction_rate", "Retrieval"),
        ("semantic_target_materialization_rate", "Materialized"),
        ("post_retrieval_completion_rate", "Post"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for idx, (metric, label) in enumerate(metrics):
        vals = [float(best_by_env[env_id][metric]) for env_id in env_ids]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(env_ids, rotation=10, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Cross-layout transfer for goal-region")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_stage_consistency(rows, out_path):
    if not rows:
        return
    rows = sorted(rows, key=lambda row: TASK_NAMES.index(str(row["task_name"])) if str(row["task_name"]) in TASK_NAMES else 999)
    tasks = [row["task_name"] for row in rows]
    x = np.arange(len(tasks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    overall = [float(row["empirical_success_rate"]) for row in rows]
    semantic = [float(row["semantic_path_completion_rate"]) for row in rows]
    product = [float(row["stage_product"]) for row in rows]
    ax.bar(x - width, overall, width=width, label="overall success")
    ax.bar(x, semantic, width=width, label="semantic-path completion")
    ax.bar(x + width, product, width=width, label="product of stage estimators")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Stage-factor consistency audit")
    ax.legend(fontsize=8)
    for offset, vals in [(-width, overall), (0.0, semantic), (width, product)]:
        for xv, yv in zip(x + offset, vals):
            ax.text(xv, yv + 0.02, f"{yv:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_assist_dumbbell(rows, out_path):
    if not rows:
        return
    metrics = [
        ("success_rate", "Success"),
        ("query_satisfaction_rate", "Retrieval"),
        ("semantic_target_materialization_rate", "Materialized"),
        ("post_retrieval_completion_rate", "Post"),
    ]
    task_names = [str(row["task_name"]) for row in rows]
    y = np.arange(len(task_names))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (prefix, title) in zip(axes.ravel(), metrics):
        vals_on = [float(row[f"{prefix}_on"]) for row in rows]
        vals_off = [float(row[f"{prefix}_off"]) for row in rows]
        deltas = [float(row[f"{prefix}_delta_on_minus_off"]) for row in rows]
        for idx in range(len(task_names)):
            ax.plot(
                [vals_off[idx], vals_on[idx]],
                [y[idx], y[idx]],
                color="#9aa0a6",
                linewidth=2.0,
                zorder=1,
            )
            mid = (vals_off[idx] + vals_on[idx]) / 2.0
            ax.text(mid, y[idx] + 0.14, f"{deltas[idx]:+0.02f}", ha="center", va="bottom", fontsize=8)
        ax.scatter(vals_off, y, color="#1f77b4", label="off", s=40, zorder=2)
        ax.scatter(vals_on, y, color="#d62728", label="on", s=40, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(task_names)
        ax.set_xlim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_stage_heatmap(rows, out_path):
    if not rows:
        return
    rows = sorted(rows, key=lambda row: TASK_NAMES.index(str(row["task_name"])) if str(row["task_name"]) in TASK_NAMES else 999)
    task_names = [str(row["task_name"]) for row in rows]
    stage_labels = ["Query", "Retrieval", "Materialized", "Post"]
    rates = np.asarray(
        [
            [
                float(row["query_formed_rate"]),
                float(row["retrieval_satisfied_rate"]),
                float(row["target_materialized_rate"]),
                float(row["completion_after_retrieval_rate"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    weakness = np.zeros_like(rates)
    weakness[:, 0] = 1.0 - rates[:, 0]
    for idx in range(1, rates.shape[1]):
        weakness[:, idx] = np.clip(rates[:, idx - 1] - rates[:, idx], 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(weakness, cmap="YlOrRd", vmin=0.0, vmax=max(0.2, float(np.max(weakness))), aspect="auto")
    ax.set_xticks(np.arange(len(stage_labels)))
    ax.set_xticklabels(stage_labels)
    ax.set_yticks(np.arange(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_title("Task-by-stage bottleneck map")
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            ax.text(j, i, f"{rates[i, j]:.2f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("drop from previous stage")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_funnel_panel(rows, out_path):
    if not rows:
        return
    rows = sorted(rows, key=lambda row: TASK_NAMES.index(str(row["task_name"])) if str(row["task_name"]) in TASK_NAMES else 999)
    stage_labels = ["Q", "R", "M", "C"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), squeeze=False)
    x = np.arange(len(stage_labels))
    for ax, row in zip(axes.ravel(), rows):
        vals = [
            float(row["query_formed_rate"]),
            float(row["retrieval_satisfied_rate"]),
            float(row["target_materialized_rate"]),
            float(row["completion_after_retrieval_rate"]),
        ]
        ax.plot(x, vals, color="#444444", linewidth=2.0, zorder=1)
        for idx, (xv, yv) in enumerate(zip(x, vals)):
            ax.scatter([xv], [yv], color=colors[idx], s=55, zorder=2)
            ax.text(xv, yv + 0.04, f"{yv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(str(row["task_name"]))
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.article_dir, "analysis")
    ensure_dir(out_dir)

    episode_rows = collect_episode_rows(args.article_dir)
    article_rows, grouped_seed_metrics = build_article_rows(
        episode_rows=episode_rows,
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    task_summary_rows = build_task_summary(article_rows)
    assist_rows = build_assist_significance(article_rows, grouped_seed_metrics)
    baseline_rows = build_baseline_comparison(article_rows)
    cross_layout_rows = build_cross_layout_rows(
        episode_rows=episode_rows,
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    scaling_rows = build_memory_scaling_rows(
        episode_rows=episode_rows,
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    stage_consistency_rows = build_stage_consistency_rows(episode_rows, article_rows)

    outputs = {
        "article_overview_with_ci.json": article_rows,
        "article_task_family_summary_with_ci.json": task_summary_rows,
        "article_assist_significance.json": assist_rows,
        "article_baseline_comparison.json": baseline_rows,
        "article_cross_layout.json": cross_layout_rows,
        "article_memory_scaling.json": scaling_rows,
        "article_stage_consistency.json": stage_consistency_rows,
    }
    for filename, rows in outputs.items():
        with open(os.path.join(out_dir, filename), "w") as file_obj:
            json.dump(rows, file_obj, indent=2)
        if rows:
            write_csv(os.path.join(out_dir, filename.replace(".json", ".csv")), rows, list(rows[0].keys()))

    plot_baseline_comparison(baseline_rows, os.path.join(out_dir, "article_baseline_comparison.png"))
    plot_stage_baseline_grid(baseline_rows, os.path.join(out_dir, "article_stage_baseline_grid.png"))
    plot_memory_scaling(scaling_rows, os.path.join(out_dir, "article_memory_scaling.png"))
    plot_cross_layout(cross_layout_rows, os.path.join(out_dir, "article_cross_layout.png"))
    plot_stage_consistency(stage_consistency_rows, os.path.join(out_dir, "article_stage_consistency.png"))
    plot_assist_dumbbell(assist_rows, os.path.join(out_dir, "article_assist_dumbbell.png"))
    plot_stage_heatmap(task_summary_rows, os.path.join(out_dir, "article_stage_heatmap.png"))
    plot_funnel_panel(task_summary_rows, os.path.join(out_dir, "article_funnel_panel.png"))


if __name__ == "__main__":
    main()
