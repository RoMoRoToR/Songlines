import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.songline_minigrid import ensure_dir


SYNTHETIC_TASKS = [
    {"task_name": "easy", "p_q": 0.90, "p_r_given_q": 0.85, "p_m_given_r": 0.90, "p_c_given_m": 0.88},
    {"task_name": "transfer_gap", "p_q": 0.40, "p_r_given_q": 0.55, "p_m_given_r": 0.75, "p_c_given_m": 0.80},
    {"task_name": "control_limited", "p_q": 0.85, "p_r_given_q": 0.80, "p_m_given_r": 0.85, "p_c_given_m": 0.35},
    {"task_name": "retrieval_limited", "p_q": 0.80, "p_r_given_q": 0.25, "p_m_given_r": 0.90, "p_c_given_m": 0.85},
]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def simulate_task(task, num_episodes, rng):
    q = rng.binomial(1, float(task["p_q"]), size=num_episodes)
    r = q * rng.binomial(1, float(task["p_r_given_q"]), size=num_episodes)
    m = r * rng.binomial(1, float(task["p_m_given_r"]), size=num_episodes)
    c = m * rng.binomial(1, float(task["p_c_given_m"]), size=num_episodes)
    q_count = max(1, int(q.sum()))
    r_count = max(1, int(r.sum()))
    m_count = max(1, int(m.sum()))
    est = {
        "task_name": str(task["task_name"]),
        "num_episodes": int(num_episodes),
        "true_p_q": float(task["p_q"]),
        "true_p_r_given_q": float(task["p_r_given_q"]),
        "true_p_m_given_r": float(task["p_m_given_r"]),
        "true_p_c_given_m": float(task["p_c_given_m"]),
        "true_stage_product": float(
            task["p_q"] * task["p_r_given_q"] * task["p_m_given_r"] * task["p_c_given_m"]
        ),
        "est_p_q": float(np.mean(q)),
        "est_p_r_given_q": float(r.sum() / q_count),
        "est_p_m_given_r": float(m.sum() / r_count),
        "est_p_c_given_m": float(c.sum() / m_count),
        "empirical_success_rate": float(np.mean(c)),
    }
    est["est_stage_product"] = float(
        est["est_p_q"] * est["est_p_r_given_q"] * est["est_p_m_given_r"] * est["est_p_c_given_m"]
    )
    est["calibration_gap"] = float(est["est_stage_product"] - est["empirical_success_rate"])
    return est


def aggregate_simulations(sample_sizes, num_replicates, base_seed):
    rows = []
    rng = np.random.RandomState(int(base_seed))
    for sample_size in sample_sizes:
        for task in SYNTHETIC_TASKS:
            reps = [simulate_task(task, num_episodes=sample_size, rng=np.random.RandomState(rng.randint(0, 10**9))) for _ in range(num_replicates)]
            row = {
                "task_name": str(task["task_name"]),
                "num_episodes": int(sample_size),
                "num_replicates": int(num_replicates),
            }
            for key in [
                "true_p_q",
                "true_p_r_given_q",
                "true_p_m_given_r",
                "true_p_c_given_m",
                "true_stage_product",
                "est_p_q",
                "est_p_r_given_q",
                "est_p_m_given_r",
                "est_p_c_given_m",
                "est_stage_product",
                "empirical_success_rate",
                "calibration_gap",
            ]:
                row[key] = float(np.mean([rep[key] for rep in reps]))
            rows.append(row)
    return rows


def plot_synthetic_validation(rows, out_path):
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    tasks = sorted({row["task_name"] for row in rows})
    for task_name in tasks:
        task_rows = sorted(
            [row for row in rows if row["task_name"] == task_name],
            key=lambda row: int(row["num_episodes"]),
        )
        xs = [int(row["num_episodes"]) for row in task_rows]
        axes[0].plot(xs, [float(row["empirical_success_rate"]) for row in task_rows], marker="o", label=f"{task_name}: empirical")
        axes[0].plot(xs, [float(row["est_stage_product"]) for row in task_rows], marker="x", linestyle="--", label=f"{task_name}: product")
        axes[1].plot(xs, [abs(float(row["calibration_gap"])) for row in task_rows], marker="o", label=task_name)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Synthetic stage-product validation")
    axes[0].set_xlabel("episodes")
    axes[0].set_ylabel("rate")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xscale("log")
    axes[1].set_title("Absolute calibration gap")
    axes[1].set_xlabel("episodes")
    axes[1].set_ylabel("|product - empirical success|")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--num_replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    rows = aggregate_simulations(
        sample_sizes=args.sample_sizes,
        num_replicates=args.num_replicates,
        base_seed=args.seed,
    )
    json_path = os.path.join(args.out_dir, "synthetic_stage_validation.json")
    with open(json_path, "w") as file_obj:
        json.dump(rows, file_obj, indent=2)
    if rows:
        write_csv(
            os.path.join(args.out_dir, "synthetic_stage_validation.csv"),
            rows,
            list(rows[0].keys()),
        )
    plot_synthetic_validation(rows, os.path.join(args.out_dir, "synthetic_stage_validation.png"))


if __name__ == "__main__":
    main()
