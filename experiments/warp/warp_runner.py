"""Warp-annotated episode runner.

A fork of ``experiments/big_experiment/runner.py`` that keeps the exact
same environment, memory adapters, planner and Q/R/M/C accounting, and
adds the W* measurement layer on top:

  - every planner lock becomes an ``MStarEvent`` with phi frozen at the
    lock moment (design §3.1),
  - completion is attributed per lock (did the agent reach THIS target
    while holding THIS lock),
  - optional counterfactual masking of foreign evidence (W1),
  - optional Warp Drive protocol hooks (W3).

The planner itself is untouched — the warp layer reads planner state
(``locked_target``) and memory state; it never writes either, except
through the explicit Warp Drive hooks which act via the same
``memory_targets`` interface every architecture already uses.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from experiments.big_experiment.env_factory import build_env
from experiments.big_experiment.memory_factory import build_memory
from experiments.big_experiment.planner import PlannerState, plan_action
from experiments.big_experiment.runner import RunConfig, _xy_close

from experiments.warp.warp_instrumentation import (
    ProvenanceLedger,
    candidate_provenance,
    mask_foreign_targets,
)
from experiments.warp.warp_types import (
    MStarEvent,
    THETA_SOFT,
    THETA_STRICT,
    WarpEpisodeLog,
)


def run_warp_episode(
    env,
    agent_ids: List[str],
    water_positions: List[Tuple[int, int]],
    memory,
    *,
    step_limit: int,
    variant_tag: str,
    mask_foreign: bool = False,
    mask_agents: Optional[List[str]] = None,
    warp_drive=None,
    theta_soft: float = THETA_SOFT,
) -> Tuple[Dict[str, Any], WarpEpisodeLog]:
    """Run one episode with W*-annotation.  Returns (metrics, warp log).

    ``mask_foreign=True`` masks foreign evidence for EVERY agent — the
    system-level counterfactual (value of the sharing channel as a
    whole).  ``mask_agents=[...]`` masks only the listed agents — the
    per-agent counterfactual (individual attribution of warp gain,
    other agents keep sharing normally).
    """

    planners = {aid: PlannerState(aid) for aid in agent_ids}
    ledger = ProvenanceLedger()
    log = WarpEpisodeLog()

    first_success_tick: Dict[str, Optional[int]] = {aid: None for aid in agent_ids}
    open_event: Dict[str, Optional[MStarEvent]] = {aid: None for aid in agent_ids}
    log.success_without_lock = {aid: False for aid in agent_ids}

    # Q/R/M/C accumulators — identical semantics to the base runner.
    Q_star = {aid: False for aid in agent_ids}
    R_star = {aid: False for aid in agent_ids}
    M_star = {aid: False for aid in agent_ids}
    n_Q = {aid: 0 for aid in agent_ids}
    n_R = {aid: 0 for aid in agent_ids}
    n_M = {aid: 0 for aid in agent_ids}
    ticks_played = 0
    trail: Dict[str, set] = {aid: set() for aid in agent_ids}
    n_agents = len(agent_ids)

    def _close_lock(aid: str, tick: int) -> None:
        ev = open_event[aid]
        if ev is not None and ev.dropped_tick is None and not ev.completed:
            ev.dropped_tick = tick
        open_event[aid] = None

    for tick in range(step_limit):
        ticks_played = tick + 1

        # 1. observe + feed the provenance ledger + memory tick
        for aid in agent_ids:
            obs = env._observation(aid)
            cells = obs.get("cells", [])
            ledger.record(aid, cells, tick)
            memory.observe(aid, cells, tick)
        memory.tick(tick)

        for aid in agent_ids:
            if env.agents[aid].success and first_success_tick[aid] is None:
                first_success_tick[aid] = tick

        # 2. Warp Drive: deliver reservations / detect conflicts (W3)
        if warp_drive is not None:
            retractions = warp_drive.on_tick(
                tick,
                {aid: (open_event[aid].target_xy if open_event[aid] else None)
                 for aid in agent_ids},
            )
            for aid, reason in retractions.items():
                ev = open_event[aid]
                if ev is not None:
                    ev.retracted = True
                    ev.retraction_tick = tick
                    ev.rollback_latency = tick - ev.tick
                    _close_lock(aid, tick)
                    # anti-M*: retract the planner lock so it re-queries
                    planners[aid].locked_target = None

        # 3. plan + capture stage events + lock provenance
        actions: Dict[str, int] = {}
        for aid in agent_ids:
            targets = memory.query(aid)
            if mask_foreign or (mask_agents is not None and aid in mask_agents):
                targets = mask_foreign_targets(targets, aid, ledger)
            if warp_drive is not None:
                targets = warp_drive.filter_targets(aid, targets, tick)

            q_held = len(targets) > 0
            if q_held:
                n_Q[aid] += 1
                Q_star[aid] = True

            r_held = False
            for t in targets:
                if any(_xy_close((float(t[0]), float(t[1])), w)
                       for w in water_positions):
                    r_held = True
                    break
            if r_held:
                n_R[aid] += 1
                R_star[aid] = True

            prev_lock = planners[aid].locked_target
            actions[aid] = plan_action(planners[aid], env, targets, tick,
                                       variant_tag)
            lt = planners[aid].locked_target

            if lt is not None:
                if any(_xy_close((float(lt[0]), float(lt[1])), w)
                       for w in water_positions):
                    n_M[aid] += 1
                    M_star[aid] = True

            # ── lock lifecycle → MStarEvent with frozen phi ──────
            cur = open_event[aid]
            if lt is None:
                if cur is not None:
                    _close_lock(aid, tick)
            elif cur is None or not _xy_close(cur.target_xy, lt):
                if cur is not None:
                    _close_lock(aid, tick)
                prov = candidate_provenance(
                    memory, aid, lt, ledger, tick, n_agents=n_agents)
                ag = env.agents[aid]
                self_seen = ledger.self_observed(aid, lt)
                visited = planners[aid].visited
                radius_visited = (
                    min(abs(v[0] - lt[0]) + abs(v[1] - lt[1]) for v in visited)
                    if visited else 0)
                co_locked = sum(
                    1 for other, ev in open_event.items()
                    if other != aid and ev is not None
                    and _xy_close(ev.target_xy, lt))
                ev = MStarEvent(
                    agent_id=aid, tick=tick, target_xy=(lt[0], lt[1]),
                    is_real_water=any(_xy_close(lt, w) for w in water_positions),
                    phi=prov.phi,
                    own_mass=prov.own_mass, foreign_mass=prov.foreign_mass,
                    per_source_mass=prov.per_source_mass,
                    self_ever_observed=self_seen,
                    w_star_strict=(prov.phi >= THETA_STRICT and not self_seen),
                    w_star_soft=(prov.phi >= theta_soft),
                    warp_radius_cells=abs(ag.x - lt[0]) + abs(ag.y - lt[1]),
                    warp_radius_visited=radius_visited,
                    source_snapshot_age=prov.source_snapshot_age,
                    co_recipients=prov.co_recipients,
                    co_locked=co_locked,
                )
                log.events.append(ev)
                open_event[aid] = ev
                if warp_drive is not None:
                    warp_drive.on_lock(aid, (lt[0], lt[1]), tick,
                                       distance=ev.warp_radius_cells)

        for aid in agent_ids:
            ag = env.agents[aid]
            trail[aid].add((ag.x, ag.y))

        result = env.step(actions)

        # 4. completion attribution
        for aid in agent_ids:
            ag = env.agents[aid]
            if ag.success and first_success_tick[aid] is None:
                first_success_tick[aid] = tick + 1
                ev = open_event[aid]
                if ev is not None and _xy_close(ev.target_xy, (ag.x, ag.y)):
                    ev.completed = True
                    ev.completion_tick = tick + 1
                else:
                    log.success_without_lock[aid] = True
                open_event[aid] = None
                if warp_drive is not None:
                    warp_drive.on_success(aid, (ag.x, ag.y), tick)

        if result.all_succeeded:
            break

    log.first_success_tick = dict(first_success_tick)

    # ── metrics (same shape as the base runner) ──────────────────
    n_succeeded = sum(1 for v in first_success_tick.values() if v is not None)
    succ_ticks = [v for v in first_success_tick.values() if v is not None]
    mean_t_succ = statistics.mean(succ_ticks) if succ_ticks else float("nan")
    C_star = {aid: (first_success_tick[aid] is not None) for aid in agent_ids}

    def _safe_div(num, den):
        return (num / den) if den > 0 else float("nan")

    metrics: Dict[str, Any] = {
        "n_succeeded": n_succeeded,
        "success_rate": n_succeeded / n_agents,
        "mean_t_succ": mean_t_succ,
        "ticks_played": ticks_played,
        "q_star_rate": sum(Q_star.values()) / n_agents,
        "r_star_rate": sum(R_star.values()) / n_agents,
        "m_star_rate": sum(M_star.values()) / n_agents,
        "c_star_rate": sum(C_star.values()) / n_agents,
        "p_R_given_Q": _safe_div(sum(R_star.values()), sum(Q_star.values())),
        "p_M_given_R": _safe_div(sum(M_star.values()), sum(R_star.values())),
        "p_C_given_M": _safe_div(sum(C_star.values()), sum(M_star.values())),
        "n_hazard_hits": sum(env.agents[aid].n_hazard_hits for aid in agent_ids),
        "total_trail": sum(len(v) for v in trail.values()),
        "mask_foreign": mask_foreign,
        "warp_drive": warp_drive is not None,
    }
    metrics.update(log.warp_metrics())
    if warp_drive is not None:
        metrics.update(warp_drive.stats())
    return metrics, log


def run_one_config_warp(
    cfg: RunConfig,
    *,
    mask_foreign: bool = False,
    warp_drive_factory=None,
) -> Tuple[Dict[str, Any], WarpEpisodeLog]:
    """Config-driven wrapper — mirrors ``runner.run_one_config``."""
    built = build_env(
        n_agents=cfg.n_agents, n_waters=cfg.n_waters,
        layout=cfg.layout, hazard_density=cfg.hazard_density,
        seed=cfg.seed, step_limit=cfg.step_limit,
    )
    env_id = f"warp_{cfg.as_tag()}"
    memory = build_memory(
        cfg.architecture, built.agent_ids, env_id,
        broadcast_every_k=(cfg.broadcast_every_k
                           if cfg.architecture in ("peer", "csm") else 4),
    )
    warp_drive = (warp_drive_factory(built.agent_ids, cfg.broadcast_every_k)
                  if warp_drive_factory is not None else None)
    metrics, log = run_warp_episode(
        built.env, built.agent_ids, built.water_positions, memory,
        step_limit=cfg.step_limit, variant_tag=cfg.architecture,
        mask_foreign=mask_foreign, warp_drive=warp_drive,
    )
    metrics.update({
        "n_agents": cfg.n_agents, "n_waters": cfg.n_waters,
        "layout": cfg.layout, "architecture": cfg.architecture,
        "broadcast_every_k": cfg.broadcast_every_k,
        "hazard_density": cfg.hazard_density, "seed": cfg.seed,
        "scarcity": cfg.n_agents / cfg.n_waters,
        "tag": cfg.as_tag(),
    })
    return metrics, log
