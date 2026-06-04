"""Universal runner — runs one config, returns metric dict.

Now instruments Q/R/M/C stage events per agent per tick following the
single-agent factorisation of Work A, extended to multi-agent:

  Q (query):         memory_query for this agent returned non-empty
  R (retrieval):     at least one returned candidate is a real water cell
  M (materialisation): planner committed to a real water (locked_target valid)
  C (completion):    agent reached water by end of episode

Per-episode flags Q*, R*, M*, C* are the OR across ticks (true iff any tick).
Per-tick rates are emitted as ratios (denominator = step_limit or last tick).

For aggregation across seeds we emit BOTH:
  - Episode flags (Q_star, R_star, M_star, C_star) — boolean per agent → mean
  - Per-tick counts (n_Q, n_R, n_M) — useful for conditional rates
"""

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


def _xy_close(a, b, tol: float = 0.6) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def run_one_config(cfg: RunConfig) -> Dict[str, Any]:
    built = build_env(
        n_agents=cfg.n_agents, n_waters=cfg.n_waters,
        layout=cfg.layout, hazard_density=cfg.hazard_density,
        seed=cfg.seed, step_limit=cfg.step_limit,
    )
    env = built.env
    agent_ids = built.agent_ids
    water_positions = built.water_positions
    env_id = f"big_{cfg.as_tag()}"

    memory = build_memory(
        cfg.architecture, agent_ids, env_id,
        broadcast_every_k=cfg.broadcast_every_k if cfg.architecture == "peer" else 4,
    )

    planners = {aid: PlannerState(aid) for aid in agent_ids}
    first_success_tick: Dict[str, Optional[int]] = {aid: None for aid in agent_ids}
    trail: Dict[str, set] = {aid: set() for aid in agent_ids}

    # ── Q/R/M/C accumulators per agent ────────────────────────────
    # Episode flags
    Q_star = {aid: False for aid in agent_ids}
    R_star = {aid: False for aid in agent_ids}
    M_star = {aid: False for aid in agent_ids}
    # Per-tick counts (denominator: ticks_played)
    n_Q = {aid: 0 for aid in agent_ids}
    n_R = {aid: 0 for aid in agent_ids}
    n_M = {aid: 0 for aid in agent_ids}
    ticks_played = 0

    for tick in range(cfg.step_limit):
        ticks_played = tick + 1

        # observe + tick memory
        for aid in agent_ids:
            obs = env._observation(aid)
            memory.observe(aid, obs.get("cells", []), tick)
        memory.tick(tick)

        # record success tick
        for aid in agent_ids:
            if env.agents[aid].success and first_success_tick[aid] is None:
                first_success_tick[aid] = tick

        # plan + capture stage events
        actions: Dict[str, int] = {}
        for aid in agent_ids:
            targets = memory.query(aid)

            # ── Q: query non-empty
            q_held = len(targets) > 0
            if q_held:
                n_Q[aid] += 1
                Q_star[aid] = True

            # ── R: any returned candidate is a real water cell
            r_held = False
            if q_held:
                for t in targets:
                    for w in water_positions:
                        if _xy_close((float(t[0]), float(t[1])), w):
                            r_held = True
                            break
                    if r_held:
                        break
            if r_held:
                n_R[aid] += 1
                R_star[aid] = True

            # ── plan action (this also updates planner.locked_target)
            actions[aid] = plan_action(
                planners[aid], env, targets, tick, cfg.architecture,
            )

            # ── M: planner committed to a real water cell
            lt = planners[aid].locked_target
            m_held = False
            if lt is not None:
                for w in water_positions:
                    if _xy_close((float(lt[0]), float(lt[1])), w):
                        m_held = True
                        break
            if m_held:
                n_M[aid] += 1
                M_star[aid] = True

        # capture positions before stepping for trail
        for aid in agent_ids:
            ag = env.agents[aid]
            trail[aid].add((ag.x, ag.y))

        result = env.step(actions)

        if result.all_succeeded:
            for aid in agent_ids:
                if env.agents[aid].success and first_success_tick[aid] is None:
                    first_success_tick[aid] = tick + 1
            break

    # ── final metrics ──────────────────────────────────────────────
    n_succeeded = sum(1 for v in first_success_tick.values() if v is not None)
    succ_ticks = [v for v in first_success_tick.values() if v is not None]
    mean_t_succ = statistics.mean(succ_ticks) if succ_ticks else float("nan")
    p95_t_succ = (sorted(succ_ticks)[int(0.95 * len(succ_ticks))]
                  if succ_ticks else float("nan"))
    total_trail = sum(len(v) for v in trail.values())
    n_hazard_hits = sum(env.agents[aid].n_hazard_hits for aid in agent_ids)

    # C* per agent: success at end of episode
    C_star = {aid: (first_success_tick[aid] is not None) for aid in agent_ids}

    # Aggregate stage rates across agents
    n_ag = len(agent_ids)
    q_star_rate = sum(Q_star.values()) / n_ag
    r_star_rate = sum(R_star.values()) / n_ag
    m_star_rate = sum(M_star.values()) / n_ag
    c_star_rate = sum(C_star.values()) / n_ag

    # Conditional episode-level rates (NaN if denominator 0)
    def _safe_div(num, den):
        return (num / den) if den > 0 else float("nan")

    qstar_count = sum(Q_star.values())
    rstar_count = sum(R_star.values())
    mstar_count = sum(M_star.values())
    cstar_count = sum(C_star.values())

    p_R_given_Q = _safe_div(rstar_count, qstar_count)
    p_M_given_R = _safe_div(mstar_count, rstar_count)
    p_C_given_M = _safe_div(cstar_count, mstar_count)

    # Per-tick aggregate rates (averaged across agents)
    if ticks_played > 0:
        q_tick_rate = statistics.mean(n_Q[aid] / ticks_played for aid in agent_ids)
        r_tick_rate = statistics.mean(n_R[aid] / ticks_played for aid in agent_ids)
        m_tick_rate = statistics.mean(n_M[aid] / ticks_played for aid in agent_ids)
    else:
        q_tick_rate = r_tick_rate = m_tick_rate = float("nan")

    return {
        # config echo
        "n_agents": cfg.n_agents, "n_waters": cfg.n_waters,
        "layout": cfg.layout, "architecture": cfg.architecture,
        "broadcast_every_k": cfg.broadcast_every_k,
        "hazard_density": cfg.hazard_density, "seed": cfg.seed,
        "step_limit": cfg.step_limit,
        # primary outcomes
        "n_succeeded": n_succeeded,
        "success_rate": n_succeeded / cfg.n_agents,
        "mean_t_succ": mean_t_succ,
        "p95_t_succ": p95_t_succ,
        "total_trail": total_trail,
        "n_hazard_hits": n_hazard_hits,
        "ticks_played": ticks_played,
        # auxiliary
        "scarcity": cfg.n_agents / cfg.n_waters,
        # Q/R/M/C — episode-level rates (avg across N agents)
        "q_star_rate": q_star_rate,
        "r_star_rate": r_star_rate,
        "m_star_rate": m_star_rate,
        "c_star_rate": c_star_rate,
        # conditional rates (episode-level)
        "p_R_given_Q": p_R_given_Q,
        "p_M_given_R": p_M_given_R,
        "p_C_given_M": p_C_given_M,
        # per-tick rates (continuous diagnostic)
        "q_tick_rate": q_tick_rate,
        "r_tick_rate": r_tick_rate,
        "m_tick_rate": m_tick_rate,
        # tag
        "tag": cfg.as_tag(),
    }
