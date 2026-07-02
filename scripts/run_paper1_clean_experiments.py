"""Run the clean experiment suite for Paper 1 (Q/R/M/C diagnostics).

The suite separates quick smoke checks from paper-facing full runs.
It does not invent new metrics; it orchestrates existing runners and
stores a manifest with success/failure status for each block.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Step:
    name: str
    cmd: List[str]
    needs_llm: bool = False
    full_only: bool = False


def _base_env() -> Dict[str, str]:
    env = dict(os.environ)
    repo = os.getcwd()
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", "/tmp/songlines_mpl")
    return env


def _step_result(name: str, cmd: List[str], code: int, elapsed: float, log_path: str) -> Dict:
    return {
        "name": name,
        "cmd": cmd,
        "returncode": int(code),
        "elapsed_s": round(elapsed, 2),
        "log_path": log_path,
        "ok": code == 0,
    }


def build_steps(args: argparse.Namespace) -> List[Step]:
    py = args.python
    out = args.out_dir
    steps: List[Step] = []

    steps.append(Step(
        "factorization_sanity",
        [
            py, "scripts/validate_qrmc_factorization.py",
            "--out_dir", os.path.join(out, "factorization_sanity"),
            "--sample_sizes", "50", "100", "250" if args.mode == "smoke" else "1000",
            "--num_replicates", "20" if args.mode == "smoke" else "200",
        ],
    ))

    if args.mode == "smoke":
        steps.append(Step(
            "single_agent_minigrid_smoke",
            [
                py, "scripts/compare_semnav_minigrid.py",
                "--env_ids", "MiniGrid-Empty-Random-6x6-v0", "MiniGrid-LavaGapS7-v0",
                "--methods", "random", "songline_graph_path", "milestone_semantic_handoff_v1",
                "--num_seeds", "1",
                "--episodes", "2",
                "--max_steps", "80",
                "--token_source", "scene_semantic",
                "--out_dir", os.path.join(out, "single_agent_minigrid_smoke"),
            ],
        ))
    else:
        steps.append(Step(
            "single_agent_article_full",
            [
                py, "scripts/benchmark_symbolic_memory_article.py",
                "--num_seeds", "10",
                "--episodes", "8",
                "--max_steps", "120",
                "--out_dir", os.path.join(out, "single_agent_article_full"),
            ],
        ))

    steps.append(Step(
        "oracle_interventions_" + args.mode,
        [
            py, "scripts/benchmark_oracle_stage_interventions.py",
            "--num_seeds", "1" if args.mode == "smoke" else "10",
            "--episodes", "2" if args.mode == "smoke" else "8",
            "--max_steps", "80" if args.mode == "smoke" else "120",
            "--num_bootstrap", "100" if args.mode == "smoke" else "4000",
            "--out_dir", os.path.join(out, "oracle_interventions_" + args.mode),
        ],
    ))

    steps.append(Step(
        "multiagent_cadence_" + args.mode,
        [
            py, "experiments/big_experiment/exp_cadence_phase.py",
            "--mode", "smoke" if args.mode == "smoke" else "full",
            "--workers", str(args.workers),
            "--out_dir", os.path.join(out, "multiagent_cadence_" + args.mode),
            "--progress_every", "5000",
        ],
    ))
    steps.append(Step(
        "multiagent_qrmc_analysis_" + args.mode,
        [
            py, "experiments/big_experiment/analyze_qrmc.py",
            "--runs_csv", os.path.join(out, "multiagent_cadence_" + args.mode, "runs.csv"),
            "--out_dir", os.path.join(out, "multiagent_cadence_" + args.mode),
        ],
    ))

    steps.append(Step(
        "commnet_qrmc_eval_" + args.mode,
        [
            py, "experiments/commnet_baseline/eval_with_qrmc.py",
            "--policy_path", "experiments/commnet_ppo_baseline/ppo_policy.pt",
            "--n_episodes", "10" if args.mode == "smoke" else "100",
            "--out_dir", os.path.join(out, "commnet_qrmc_eval_" + args.mode),
        ],
    ))

    if args.mode == "full":
        steps.append(Step(
            "semantic_noise_robustness_full",
            [
                py, "scripts/benchmark_semantic_noise_robustness.py",
                "--out_dir", os.path.join(out, "semantic_noise_robustness_full"),
            ],
        ))
        steps.append(Step(
            "babyai_portability_full",
            [
                py, "scripts/compare_semnav_babyai.py",
                "--out_dir", os.path.join(out, "babyai_portability_full"),
            ],
        ))
        steps.append(Step(
            "minigrid_multiagent_portability_full",
            [
                py, "experiments/minigrid_multiagent_wrapper/run_portability_sweep.py",
                "--out_dir", os.path.join(out, "minigrid_multiagent_portability_full"),
            ],
        ))

    if not args.skip_llm:
        steps.append(Step(
            "llm_textnav_model_sweep_" + args.mode,
            [
                py, "-m", "experiments.llm_collective.run_model_sweep",
                "--models", *args.llm_models,
                "--episodes", str(args.llm_episodes if args.llm_episodes is not None else (1 if args.mode == "smoke" else 10)),
                "--step_limit", str(args.llm_step_limit if args.llm_step_limit is not None else (10 if args.mode == "smoke" else 25)),
                "--out_dir", os.path.join(out, "llm_textnav_model_sweep_" + args.mode),
            ],
            needs_llm=True,
        ))

    return steps


def run_step(step: Step, args: argparse.Namespace, env: Dict[str, str]) -> Dict:
    log_dir = os.path.join(args.out_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{step.name}.log")
    print(f"\n[{step.name}] {' '.join(step.cmd)}")
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(step.cmd) + "\n\n")
        proc = subprocess.run(
            step.cmd,
            cwd=os.getcwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - t0
    print(f"[{step.name}] returncode={proc.returncode} elapsed={elapsed:.1f}s log={log_path}")
    return _step_result(step.name, step.cmd, proc.returncode, elapsed, log_path)


def write_results(out_dir: str, results: List[Dict]) -> None:
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "RESULTS.md"), "w") as f:
        f.write("# Paper 1 clean experiment suite\n\n")
        f.write("| Step | Status | Seconds | Log |\n")
        f.write("|---|---:|---:|---|\n")
        for r in results:
            status = "PASS" if r["ok"] else "FAIL"
            f.write(f"| {r['name']} | {status} | {r['elapsed_s']} | `{r['log_path']}` |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--out_dir", default="tmp/paper1_clean_experiments")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--llm_models", nargs="+", default=["qwen3:4b", "llama3.1:latest"])
    parser.add_argument("--llm_episodes", type=int, default=None)
    parser.add_argument("--llm_step_limit", type=int, default=None)
    parser.add_argument("--stop_on_failure", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = _base_env()
    results: List[Dict] = []
    for step in build_steps(args):
        result = run_step(step, args, env)
        results.append(result)
        write_results(args.out_dir, results)
        if args.stop_on_failure and not result["ok"]:
            break

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\nFAILED: {', '.join(r['name'] for r in failed)}")
        sys.exit(1)
    print(f"\nAll steps passed. Manifest: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
