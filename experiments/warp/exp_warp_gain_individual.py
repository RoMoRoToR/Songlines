"""W1 addendum — per-agent counterfactual (individual warp attribution).

The main W1 mask is system-level: foreign evidence is zeroed for every
agent, measuring the value of the sharing CHANNEL.  This addendum masks
one agent at a time (the rest keep sharing), giving the individual
attribution of warp gain:

    WG_i = t_i(mask agent i) − t_i(full),   censored at step_limit

on the (5,3)|random scarcity cell at peer-K2 (the arm with the largest
warp effects), 20 seeds.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_gain_individual.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.big_experiment.env_factory import build_env
from experiments.big_experiment.memory_factory import build_memory
from experiments.warp.warp_runner import run_warp_episode

N, M, LAYOUT, ARCH, K, HAZ, STEP = 5, 3, "random", "peer", 2, 0.05, 120
SEEDS = range(20)
OUT = "tmp/warp/w1_gain/w1_individual.json"


def run(seed: int, mask_agents) -> Dict[str, Any]:
    built = build_env(n_agents=N, n_waters=M, layout=LAYOUT,
                      hazard_density=HAZ, seed=seed, step_limit=STEP)
    memory = build_memory(ARCH, built.agent_ids, f"ind_{seed}",
                          broadcast_every_k=K)
    _, log = run_warp_episode(
        built.env, built.agent_ids, built.water_positions, memory,
        step_limit=STEP, variant_tag=ARCH, mask_agents=mask_agents)
    return {aid: (t if t is not None else STEP)
            for aid, t in log.first_success_tick.items()}


def main() -> None:
    agent_ids = [f"agent-{chr(ord('A') + i)}" for i in range(N)]
    per_agent_wg: Dict[str, List[float]] = {aid: [] for aid in agent_ids}
    others_wg: List[float] = []  # spillover onto non-masked agents

    for seed in SEEDS:
        full = run(seed, None)
        for aid in agent_ids:
            masked = run(seed, [aid])
            per_agent_wg[aid].append(masked[aid] - full[aid])
            others_wg.extend(masked[o] - full[o]
                             for o in agent_ids if o != aid)

    def ci(vals):
        rng = np.random.default_rng(4)
        arr = np.array(vals, dtype=float)
        boots = [np.mean(rng.choice(arr, len(arr))) for _ in range(4000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return round(float(np.mean(arr)), 2), round(float(lo), 2), \
            round(float(hi), 2)

    all_wg = [v for vals in per_agent_wg.values() for v in vals]
    summary = {
        "cell": f"N{N}M{M}|{LAYOUT}|{ARCH}-k{K}",
        "n_seeds": len(list(SEEDS)),
        "per_agent_wg_ticks": {aid: ci(vals)
                               for aid, vals in per_agent_wg.items()},
        "pooled_individual_wg": ci(all_wg),
        "spillover_on_others": ci(others_wg),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Per-agent counterfactual, {summary['cell']}, "
          f"{summary['n_seeds']} seeds")
    print(f"  pooled individual WG_i (>0 → warp helps agent i): "
          f"{summary['pooled_individual_wg']}")
    print(f"  spillover on non-masked agents:                   "
          f"{summary['spillover_on_others']}")
    for aid, v in summary["per_agent_wg_ticks"].items():
        print(f"    {aid}: {v}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
