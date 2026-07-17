"""Run Q/R/M/C TextNav experiments across local LLM backends.

This is the Paper 1 LLM block: it keeps the substrate and Q/R/M/C event
semantics fixed, varying only the local Ollama model. It intentionally
does not include a mock backend; if Ollama is unavailable the experiment
should fail visibly rather than silently becoming non-LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.llm_collective.llm_backend import OllamaBackend
from experiments.llm_collective.qrmc_llm_runner import run_one_episode


def _safe_div(num: float, den: float) -> float:
    return float("nan") if den == 0 else float(num) / float(den)


def _fmt_float(x: float) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    return round(float(x), 4)


def _write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_model(model: str, args: argparse.Namespace) -> Dict[str, Any]:
    model_slug = model.replace("/", "_").replace(":", "_")
    model_out_dir = os.path.join(args.out_dir, model_slug)
    os.makedirs(model_out_dir, exist_ok=True)
    runs_path = os.path.join(model_out_dir, "runs.csv")
    cache_dir = os.path.join(model_out_dir, ".cache_llm")
    if getattr(args, "backend", "ollama") == "hf":
        from experiments.llm_collective.hf_backend import HFBackend
        backend = HFBackend(
            model=model,
            cache_dir=cache_dir,
            temperature=args.temperature,
        )
    else:
        backend = OllamaBackend(
            model=model,
            cache_dir=cache_dir,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )

    rows: List[Dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        result = run_one_episode(
            seed=seed,
            step_limit=args.step_limit,
            verbose=args.verbose,
            backend=backend,
        )
        row = result.to_summary_dict()
        row["model"] = model
        rows.append(row)
        print(
            f"[{model}] seed={seed} succ={int(result.succeeded)} "
            f"Q/R/M/C={int(result.q_star)}/{int(result.r_star)}/"
            f"{int(result.m_star)}/{int(result.c_star)} ticks={result.n_ticks}",
            flush=True,
        )
        _write_rows_csv(runs_path, rows)

    q = sum(int(r["q_star"]) for r in rows)
    r = sum(int(r["r_star"]) for r in rows)
    m = sum(int(r["m_star"]) for r in rows)
    c = sum(int(r["c_star"]) for r in rows)
    n = len(rows)
    p_r_q = _safe_div(r, q)
    p_m_r = _safe_div(m, r)
    p_c_m = _safe_div(c, m)
    q_rate = _safe_div(q, n)
    c_rate = _safe_div(c, n)
    product = q_rate * (0.0 if math.isnan(p_r_q) else p_r_q) * (0.0 if math.isnan(p_m_r) else p_m_r) * (0.0 if math.isnan(p_c_m) else p_c_m)

    summary = {
        "model": model,
        "episodes": n,
        "step_limit": args.step_limit,
        "success_rate": _safe_div(sum(int(r["succeeded"]) for r in rows), n),
        "q_star_rate": q_rate,
        "r_star_rate": _safe_div(r, n),
        "m_star_rate": _safe_div(m, n),
        "c_star_rate": c_rate,
        "p_R_given_Q": p_r_q,
        "p_M_given_R": p_m_r,
        "p_C_given_M": p_c_m,
        "product": product,
        "factorization_abs_gap": abs(product - c_rate),
        "nested_ok": q >= r >= m >= c,
        "backend": backend.summary(),
        "runs_csv": runs_path,
    }
    with open(os.path.join(model_out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_markdown(out_dir: str, summaries: List[Dict[str, Any]]) -> None:
    with open(os.path.join(out_dir, "RESULTS.md"), "w") as f:
        f.write("# LLM Q/R/M/C model sweep\n\n")
        f.write("Same TextNav substrate, same Q/R/M/C event definitions, varying only the local Ollama model.\n\n")
        f.write("| Model | Raw prompt | Episodes | Success | Q* | R* | M* | C* | P(R|Q) | P(M|R) | P(C|M) | Product gap | Nested |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for s in summaries:
            raw_prompt = bool(s.get("backend", {}).get("raw_prompt", False))
            f.write(
                f"| {s['model']} | {'yes' if raw_prompt else 'no'} "
                f"| {s['episodes']} | {_fmt_float(s['success_rate'])} "
                f"| {_fmt_float(s['q_star_rate'])} | {_fmt_float(s['r_star_rate'])} "
                f"| {_fmt_float(s['m_star_rate'])} | {_fmt_float(s['c_star_rate'])} "
                f"| {_fmt_float(s['p_R_given_Q'])} | {_fmt_float(s['p_M_given_R'])} "
                f"| {_fmt_float(s['p_C_given_M'])} | {_fmt_float(s['factorization_abs_gap'])} "
                f"| {'yes' if s['nested_ok'] else 'no'} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3:4b", "llama3.1:latest"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--step_limit", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument("--backend", choices=["ollama", "hf"], default="ollama")
    parser.add_argument("--out_dir", type=str, default="tmp/paper1_llm_model_sweep")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = [run_model(model, args) for model in args.models]
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    write_markdown(args.out_dir, summaries)
    print(f"Saved summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
