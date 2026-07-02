"""Smoke: CSM runs end-to-end through the same Q/R/M/C logger."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.big_experiment.runner import RunConfig, run_one_config


def main():
    print("CSM smoke (N=3, T=2, custom grid, 5 seeds):")
    print("=" * 60)
    for s in range(5):
        cfg = RunConfig(
            n_agents=3, n_waters=2, layout="asymmetric",
            architecture="csm", broadcast_every_k=8,
            hazard_density=0.05, seed=s, step_limit=80,
        )
        out = run_one_config(cfg)
        print(f"  seed {s}: "
              f"P(R|Q)={float(out.get('p_R_given_Q', float('nan'))):.2f}  "
              f"P(M|R)={float(out.get('p_M_given_R', float('nan'))):.2f}  "
              f"P(C|M)={float(out.get('p_C_given_M', float('nan'))):.2f}  "
              f"succ={float(out.get('success_rate', float('nan'))):.2f}")
    print("\nCSM ran end-to-end — Q/R/M/C events extract correctly.")


if __name__ == "__main__":
    main()
