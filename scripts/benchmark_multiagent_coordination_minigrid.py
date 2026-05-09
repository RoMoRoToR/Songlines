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
        key = (row["env_id"], row["task_mode"], row["policy"])
        grouped.setdefault(key, []).append(row)

    lines = []
    lines.append("# Multi-Agent Coordination Report")
    lines.append("")
    lines.append("## Claim")
    lines.append("This experiment is intended to isolate the transition from one agent to two agents, with and without communication.")
    lines.append("The main signal is whether communication improves team completion after one agent discovers the target.")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- env_ids: {', '.join(args.env_ids)}")
    lines.append(f"- task_modes: {', '.join(args.task_modes)}")
    lines.append(f"- policies: {', '.join(args.policies)}")
    lines.append(f"- seeds: {', '.join(str(v) for v in args.seeds)}")
    lines.append(f"- episodes_per_seed: {args.episodes}")
    lines.append(f"- max_steps: {args.max_steps}")
    lines.append("")
    lines.append("## Single vs two agents")
    for key, rows in sorted(grouped.items()):
        env_id, task_mode, policy = key
        single = next((row for row in rows if row["team_config"] == "single"), None)
        no_comm = next((row for row in rows if row["team_config"] == "two_no_comm"), None)
        comm = next((row for row in rows if row["team_config"] == "two_comm"), None)
        lines.append(f"### {env_id} / {task_mode} / {policy}")
        if single is not None:
            lines.append(
                f"- single-agent discovery success: {single['success_mean']:.3f}, latency {single['discovery_latency_mean']:.2f}"
            )
        if no_comm is not None:
            lines.append(
                f"- two agents without communication: discovery {no_comm['success_mean']:.3f}, team completion {no_comm['team_success_mean']:.3f}, completion latency {no_comm['team_completion_latency_mean']:.2f}"
            )
        if comm is not None:
            lines.append(
                f"- two agents with communication: discovery {comm['success_mean']:.3f}, team completion {comm['team_success_mean']:.3f}, completion latency {comm['team_completion_latency_mean']:.2f}"
            )
        lines.append("")

    with open(out_path, "w") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENV_IDS)
    parser.add_argument("--task_modes", nargs="+", default=DEFAULT_TASK_MODES, choices=DEFAULT_TASK_MODES)
    parser.add_argument("--policies", nargs="+", default=["random", "levy"], choices=["random", "levy"])
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
    parser.add_argument("--out_dir", type=str, default="tmp/multiagent_coordination_final")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    bench_args = SimpleNamespace(
        env_ids=list(args.env_ids),
        task_modes=list(args.task_modes),
        policies=list(args.policies),
        team_configs=["single", "two_no_comm", "two_comm"],
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
        out_path=os.path.join(args.out_dir, "multiagent_coordination_report.md"),
    )
    with open(os.path.join(args.out_dir, "multiagent_coordination_metadata.json"), "w") as file_obj:
        json.dump(
            {
                "experiment_type": "multiagent_coordination",
                "claim": "Communication should help the second agent complete the task after discovery.",
                "env_ids": list(args.env_ids),
                "task_modes": list(args.task_modes),
                "policies": list(args.policies),
                "seeds": list(args.seeds),
            },
            file_obj,
            indent=2,
        )


if __name__ == "__main__":
    main()
