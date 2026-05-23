"""Universal runner — runs one config, returns metric dict."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from experiments.big_experiment.env_factory import build_env
from experiments.big_experiment.memory_factory import WATER_TAG, build_memory
from experiments.big_experiment.planner import PlannerState, plan_action


@dataclass
class RunConfig:
    n_agents: int
    n_waters: int
    layout: str            # symmetric / asymmetric / random
    architecture: str      # independent / shared / centralized / peer
    broadcast_every_k: int # only meaningful for peer; -1 for non-peer
    hazard_density: float
    seed: int
    step_limit: int = 120

    def as_tag(self) -> str:
        k = f"k{self.broadcast_every_k}" if self.architecture == "peer" else "-"
        return (f"N={self.n_agents}_M={self.n_waters}_L={self.layout}"
                f"_A={self.architecture}_{k}_H={self.hazard_density}_s={self.seed}")


def run_one_config(cfg: RunConfig) -> Dict[str, Any]:
    built = build_env(
        n_agents=cfg.n_agents, n_waters=cfg.n_waters,
        layout=cfg.layout, hazard_density=cfg.hazard_density,
        seed=cfg.seed, step_limit=cfg.step_limit,
    )
    env = built.env
    agent_ids = built.agent_ids
    env_id = f"big_{cfg.as_tag()}"

    memory = build_memory(
        cfg.architecture, agent_ids, env_id,
        broadcast_every_k=cfg.broadcast_every_k if cfg.architecture == "peer" else 4,
    )

    planners = {aid: PlannerState(aid) for aid in agent_ids}
    first_success_tick: Dict[str, Optional[int]] = {aid: None for aid in agent_ids}
    trail: Dict[str, set] = {aid: set() for aid in agent_ids}

    for tick in range(cfg.step_limit):
        # observe + tick memory
        for aid in agent_ids:
            obs = env._observation(aid)
            memory.observe(aid, obs.get("cells", []), tick)
        memory.tick(tick)

        # record success tick
        for aid in agent_ids:
            if env.agents[aid].success and first_success_tick[aid] is None:
                first_success_tick[aid] = tick

        # plan + step
        actions: Dict[str, int] = {}
        per_agent_targets: Dict[str, List] = {}
        for aid in agent_ids:
            targets = memory.query(aid)
            per_agent_targets[aid] = targets
            actions[aid] = plan_action(
                planners[aid], env, targets, tick, cfg.architecture,
            )

        # capture positions before stepping for trail
        for aid in agent_ids:
            ag = env.agents[aid]
            trail[aid].add((ag.x, ag.y))

        result = env.step(actions)

        if result.all_succeeded:
            # one more success-tick pass for the just-finished step
            for aid in agent_ids:
                if env.agents[aid].success and first_success_tick[aid] is None:
                    first_success_tick[aid] = tick + 1
            break

    # final metrics
    n_succeeded = sum(1 for v in first_success_tick.values() if v is not None)
    succ_ticks = [v for v in first_success_tick.values() if v is not None]
    mean_t_succ = statistics.mean(succ_ticks) if succ_ticks else float("nan")
    p95_t_succ = (sorted(succ_ticks)[int(0.95 * len(succ_ticks))]
                  if succ_ticks else float("nan"))
    total_trail = sum(len(v) for v in trail.values())
    n_hazard_hits = sum(env.agents[aid].n_hazard_hits for aid in agent_ids)

    return {
        # config echo
        "n_agents": cfg.n_agents, "n_waters": cfg.n_waters,
        "layout": cfg.layout, "architecture": cfg.architecture,
        "broadcast_every_k": cfg.broadcast_every_k,
        "hazard_density": cfg.hazard_density, "seed": cfg.seed,
        "step_limit": cfg.step_limit,
        # outcomes
        "n_succeeded": n_succeeded,
        "success_rate": n_succeeded / cfg.n_agents,
        "mean_t_succ": mean_t_succ,
        "p95_t_succ": p95_t_succ,
        "total_trail": total_trail,
        "n_hazard_hits": n_hazard_hits,
        # auxiliary
        "scarcity": cfg.n_agents / cfg.n_waters,
        "tag": cfg.as_tag(),
    }
