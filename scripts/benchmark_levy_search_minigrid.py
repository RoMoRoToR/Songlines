import argparse
import json
import os
from types import SimpleNamespace

from scripts.benchmark_multiagent_levy_minigrid import run_benchmark
from scripts.songline_minigrid import ensure_dir


DEFAULT_ENV_IDS = [
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-FourRooms-v0",
]

DEFAULT_TASK_MODES = [
    "default",
    "water_search_v1",
    "rest_search_v1",
]


def write_report(results, args, out_path: str):
    aggregate_rows = results["aggregate_rows"]
    grouped = {}
    for row in aggregate_rows:
        key = (row["env_id"], row["task_mode"])
        grouped.setdefault(key, []).append(row)

    lines = []
    lines.append("# Levy-like Search Report")
    lines.append("")
    lines.append("## Claim")
    lines.append("This experiment is intended to show a Levy-like search regime separately from multi-agent coordination.")
    lines.append("The qualitative pattern is local scanning most of the time, combined with occasional longer relocations.")
    lines.append("That is the same high-level pattern often discussed in the context of animal foraging and sparse resource search.")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- env_ids: {', '.join(args.env_ids)}")
    lines.append(f"- task_modes: {', '.join(args.task_modes)}")
    lines.append(f"- seeds: {', '.join(str(v) for v in args.seeds)}")
    lines.append(f"- episodes_per_seed: {args.episodes}")
    lines.append(f"- max_steps: {args.max_steps}")
    lines.append(f"- levy_alpha: {args.levy_alpha}")
    lines.append("")
    lines.append("## Random vs Levy")
    for key, rows in sorted(grouped.items()):
        env_id, task_mode = key
        random_row = next((row for row in rows if row["policy"] == "random"), None)
        levy_row = next((row for row in rows if row["policy"] == "levy"), None)
        lines.append(f"### {env_id} / {task_mode}")
        if random_row is not None and levy_row is not None:
            lines.append(
                f"- success: {random_row['success_mean']:.3f} -> {levy_row['success_mean']:.3f}"
            )
            lines.append(
                f"- discovery latency: {random_row['discovery_latency_mean']:.2f} -> {levy_row['discovery_latency_mean']:.2f}"
            )
            lines.append(
                f"- coverage rate: {random_row['coverage_rate_mean']:.3f} -> {levy_row['coverage_rate_mean']:.3f}"
            )
            lines.append(
                f"- mean Levy segment length: {levy_row['mean_levy_run_length_mean']:.2f}"
            )
        lines.append("")

    with open(out_path, "w") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENV_IDS)
    parser.add_argument("--task_modes", nargs="+", default=DEFAULT_TASK_MODES, choices=DEFAULT_TASK_MODES)
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
    parser.add_argument("--out_dir", type=str, default="tmp/levy_search_final")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    bench_args = SimpleNamespace(
        env_ids=list(args.env_ids),
        task_modes=list(args.task_modes),
        policies=["random", "levy"],
        team_configs=["single"],
        seeds=list(args.seeds),
        episodes=args.episodes,
        max_steps=args.max_steps,
        water_success_radius=args.water_success_radius,
        rest_success_radius=args.rest_success_radius,
        success_radius=args.success_radius,
        discovery_radius=args.discovery_radius,
        levy_alpha=args.levy_alpha,
        levy_min_run=args.levy_min_run,
        levy_max_run=args.levy_max_run,
        levy_horizon=args.levy_horizon,
        long_run_threshold=args.long_run_threshold,
        out_dir=args.out_dir,
    )
    results = run_benchmark(bench_args)
    write_report(
        results=results,
        args=args,
        out_path=os.path.join(args.out_dir, "levy_search_report.md"),
    )
    with open(os.path.join(args.out_dir, "levy_search_metadata.json"), "w") as file_obj:
        json.dump(
            {
                "experiment_type": "levy_search",
                "claim": "Levy-like search approximates local scanning with occasional long relocations.",
                "env_ids": list(args.env_ids),
                "task_modes": list(args.task_modes),
                "seeds": list(args.seeds),
            },
            file_obj,
            indent=2,
        )


if __name__ == "__main__":
    main()
