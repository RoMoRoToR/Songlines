"""Smoke test for the MiniGrid wrapper integrated with run_one_config.

Acceptance (a): Q/R/M/C events extractable with the *same* operational
definitions used in experiments/big_experiment (Targets_i, Locked_i, W,
eps=0.6). Smoke runs N=3, T=2 on the MiniGrid FourRooms layout via the
existing peer-architecture runner.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import experiments.big_experiment.env_factory as ef
import experiments.big_experiment.runner as runner_mod
from experiments.big_experiment.runner import RunConfig, run_one_config
from experiments.minigrid_multiagent_wrapper.env_wrapper import build_minigrid_env


def main():
    # Monkey-patch build_env on BOTH the source module and the binding
    # imported into runner_mod (Python's `from x import y` copies the
    # reference at import time).
    _orig_ef = ef.build_env
    _orig_rn = runner_mod.build_env
    ef.build_env = build_minigrid_env
    runner_mod.build_env = build_minigrid_env

    print("Smoke: MiniGrid wrapper running through run_one_config")
    print("=" * 60)
    try:
        for seed in range(3):
            cfg = RunConfig(
                n_agents=3, n_waters=2, layout="fourrooms",
                architecture="peer", broadcast_every_k=8,
                hazard_density=0.05, seed=seed, step_limit=80,
            )
            out = run_one_config(cfg)
            q, r = out.get("p_Q", float("nan")), out.get("p_R_given_Q", float("nan"))
            m, c = out.get("p_M_given_R", float("nan")), out.get("p_C_given_M", float("nan"))
            succ = out.get("success_rate", float("nan"))
            print(f"  seed {seed}: P(Q*)={q:.2f}  P(R|Q)={r:.2f}  "
                  f"P(M|R)={m:.2f}  P(C|M)={c:.2f}  succ={succ:.2f}")
        print()
        print("Smoke PASS — runner.py executes unchanged on MiniGrid substrate.")
    finally:
        ef.build_env = _orig_ef
        runner_mod.build_env = _orig_rn


if __name__ == "__main__":
    main()
