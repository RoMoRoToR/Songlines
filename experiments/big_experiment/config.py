"""Sweep configuration — enumerates all valid (N, M, layout, arch, K, hazard, seed)."""

from __future__ import annotations

from typing import List

from experiments.big_experiment.runner import RunConfig


# ─────────────────────── default full sweep ───────────────────────

N_AGENTS_FULL = [3, 5, 8]
M_BY_N_FULL = {
    3: [2, 3],
    5: [2, 3, 5],
    8: [2, 3, 5, 8],
}
LAYOUTS_FULL = ["symmetric", "asymmetric", "random"]
ARCHITECTURES_FULL = ["independent", "shared", "centralized", "peer"]
PEER_KS_FULL = [1, 2, 4, 8, 16]
HAZARD_DENSITIES_FULL = [0.0, 0.05, 0.10]
SEEDS_FULL = list(range(20))
STEP_LIMIT_FULL = 120


# ─────────────────────── smoke sweep ───────────────────────

N_AGENTS_SMOKE = [3]
M_BY_N_SMOKE = {3: [2, 3]}
LAYOUTS_SMOKE = ["asymmetric"]
ARCHITECTURES_SMOKE = ["independent", "centralized", "peer"]
PEER_KS_SMOKE = [1, 4]
HAZARD_DENSITIES_SMOKE = [0.0]
SEEDS_SMOKE = list(range(3))
STEP_LIMIT_SMOKE = 60


def expand_configs(
    n_agents_list, m_by_n, layouts, architectures, peer_ks,
    hazard_densities, seeds, step_limit,
) -> List[RunConfig]:
    """Enumerate all valid configs given the sweep axes."""
    configs: List[RunConfig] = []
    for n in n_agents_list:
        for m in m_by_n[n]:
            if m > n:
                continue
            for layout in layouts:
                for arch in architectures:
                    if arch == "peer":
                        ks = peer_ks
                    else:
                        ks = [-1]  # sentinel: K does not apply
                    for k in ks:
                        for h in hazard_densities:
                            for s in seeds:
                                configs.append(RunConfig(
                                    n_agents=n, n_waters=m, layout=layout,
                                    architecture=arch, broadcast_every_k=k,
                                    hazard_density=h, seed=s,
                                    step_limit=step_limit,
                                ))
    return configs


def smoke_configs() -> List[RunConfig]:
    return expand_configs(
        N_AGENTS_SMOKE, M_BY_N_SMOKE, LAYOUTS_SMOKE,
        ARCHITECTURES_SMOKE, PEER_KS_SMOKE,
        HAZARD_DENSITIES_SMOKE, SEEDS_SMOKE, STEP_LIMIT_SMOKE,
    )


def full_configs() -> List[RunConfig]:
    return expand_configs(
        N_AGENTS_FULL, M_BY_N_FULL, LAYOUTS_FULL,
        ARCHITECTURES_FULL, PEER_KS_FULL,
        HAZARD_DENSITIES_FULL, SEEDS_FULL, STEP_LIMIT_FULL,
    )
