"""Config-driven episode wrapper for the coordination arms (Package A).

Wraps ``experiments.warp.warp_runner.run_warp_episode`` — the exact
runner used for the Warp Drive benchmark — with an arm-selected
coordination protocol from ``coord_protocols``.  Nothing in the warp
runner, planner or memory layers is modified; the protocol object is the
only moving part, and it is constructed here so it can hold an ``env``
reference for reading each agent's OWN position (own knowledge; foreign
state arrives only via broadcast announcements).

Adds to the metric dict:
  * ``arm`` and full config echo,
  * ``t_cens`` — mean censored completion time over agents
    (first_success_tick, or step_limit for agents that never complete),
  * duplicate-commitment and message accounting from the protocol.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Tuple

from experiments.big_experiment.env_factory import build_env
from experiments.big_experiment.memory_factory import build_memory
from experiments.big_experiment.runner import RunConfig
from experiments.warp.warp_runner import run_warp_episode

from experiments.coordination.coord_protocols import make_protocol


def t_censored(log, agent_ids, step_limit: int) -> float:
    """Mean completion time with failures censored at step_limit."""
    return statistics.mean(
        (log.first_success_tick.get(aid) if log.first_success_tick.get(aid)
         is not None else step_limit)
        for aid in agent_ids)


def run_coord_episode(cfg: RunConfig, arm: str,
                      *, keep_log: bool = False
                      ) -> Tuple[Dict[str, Any], Any]:
    """Run one benchmark episode with the given coordination arm."""
    built = build_env(
        n_agents=cfg.n_agents, n_waters=cfg.n_waters,
        layout=cfg.layout, hazard_density=cfg.hazard_density,
        seed=cfg.seed, step_limit=cfg.step_limit,
    )
    env_id = f"coord_{arm}_{cfg.as_tag()}"
    memory = build_memory(
        "peer", built.agent_ids, env_id,
        broadcast_every_k=cfg.broadcast_every_k,
    )
    proto = make_protocol(arm, built.agent_ids, cfg.broadcast_every_k,
                          env=built.env, seed=cfg.seed)
    metrics, log = run_warp_episode(
        built.env, built.agent_ids, built.water_positions, memory,
        step_limit=cfg.step_limit, variant_tag="peer", warp_drive=proto,
    )
    metrics.update({
        "arm": arm,
        "n_agents": cfg.n_agents, "n_waters": cfg.n_waters,
        "layout": cfg.layout, "broadcast_every_k": cfg.broadcast_every_k,
        "hazard_density": cfg.hazard_density, "seed": cfg.seed,
        "step_limit": cfg.step_limit,
        "scarcity": cfg.n_agents / cfg.n_waters,
        "t_cens": t_censored(log, built.agent_ids, cfg.step_limit),
        # successes with no lock ever held on the reached cell — explains
        # p_C_given_M > 1 under heavy query filtering
        "n_success_no_lock": sum(log.success_without_lock.values()),
        "tag": cfg.as_tag(),
    })
    return metrics, (log if keep_log else None)
