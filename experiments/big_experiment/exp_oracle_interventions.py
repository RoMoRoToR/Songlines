"""Oracle interventions in peer architecture.

Adapts the single-agent oracle-stage protocol (Work A §6.3) to peer
multi-agent setup.  For each stage R, M, we replace its output with
ground truth and re-run a focused subset of the sweep at varying K.

Three variants:

  ``baseline``      — normal peer memory pipeline
  ``oracle_R``      — memory query always returns the positions of REAL
                       water cells (replaces retrieval errors)
  ``oracle_M``      — planner always materialises toward the nearest
                       real water target regardless of memory contents
                       and regardless of peer occupancy

Hypothesis:
  - At LARGE K (slow broadcast), ``oracle_R`` should give large speedup
    (R-bottleneck dominates).
  - At SMALL K (fast broadcast in scarcity), ``oracle_M`` should give
    large speedup (M-bottleneck dominates: peers race for targets).
  - At K*, neither oracle should help much (both R and M are already
    near-optimal balance).

Usage::

    PYTHONPATH=. .venv/bin/python experiments/big_experiment/exp_oracle_interventions.py \\
        --out_dir tmp/big_experiment_oracle
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.big_experiment.env_factory import build_env
from experiments.big_experiment.memory_factory import WATER_TAG, build_memory
from experiments.big_experiment.planner import PlannerState, plan_action


# ──────────────────────────────────────────────────── oracle adapter


def _make_oracle_query(base_query, water_positions, oracle_mode: str):
    """Return a wrapped query function that may inject ground-truth water positions."""
    if oracle_mode == "baseline":
        return base_query

    if oracle_mode == "oracle_R":
        def wrapped(agent_id):
            real_query_result = base_query(agent_id)
            # Replace returned candidates with ground-truth waters when they
            # are close enough to a real water — otherwise replace with full GT.
            return [tuple(map(float, w)) for w in water_positions]
        return wrapped

    raise ValueError(f"Unknown oracle_mode for query: {oracle_mode}")


def _override_plan_action_for_oracle_M(
    state, env, memory_targets, tick, variant, water_positions,
):
    """Oracle M: always navigate toward nearest real water, ignore occupancy."""
    from multiagent_env import FORWARD, NOOP, TURN_LEFT, TURN_RIGHT, WALL
    from multiagent_env.grid_world import DIR_DELTAS

    ag = env.agents[state.agent_id]
    if ag.success:
        return NOOP

    def dist(xy):
        return abs(xy[0] - ag.x) + abs(xy[1] - ag.y)

    targets = sorted(water_positions, key=dist)
    tx, ty = targets[0]
    state.locked_target = (tx, ty)

    if (tx, ty) == (ag.x, ag.y):
        return NOOP

    dx = tx - ag.x
    dy = ty - ag.y
    if abs(dx) >= abs(dy):
        target_dir = 0 if dx > 0 else 2
    else:
        target_dir = 1 if dy > 0 else 3

    if ag.direction == target_dir:
        ddx, ddy = DIR_DELTAS[target_dir]
        nx, ny = ag.x + ddx, ag.y + ddy
        if env.cell(nx, ny) != WALL and 0 <= nx < env.width and 0 <= ny < env.height:
            return FORWARD
        # blocked — turn right
        return TURN_RIGHT
    diff = (target_dir - ag.direction) % 4
    if diff == 1:
        return TURN_RIGHT
    if diff == 3:
        return TURN_LEFT
    return TURN_LEFT


# ──────────────────────────────────────────────────── runner


@dataclass
class OracleRunConfig:
    n_agents: int
    n_waters: int
    layout: str
    broadcast_every_k: int
    hazard_density: float
    oracle_mode: str   # baseline / oracle_R / oracle_M
    seed: int
    step_limit: int = 120


def _xy_close(a, b, tol: float = 0.6) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def run_oracle_config(cfg: OracleRunConfig) -> Dict[str, Any]:
    built = build_env(
        n_agents=cfg.n_agents, n_waters=cfg.n_waters,
        layout=cfg.layout, hazard_density=cfg.hazard_density,
        seed=cfg.seed, step_limit=cfg.step_limit,
    )
    env = built.env
    agent_ids = built.agent_ids
    water_positions = built.water_positions

    env_id = f"oracle_{cfg.oracle_mode}_K{cfg.broadcast_every_k}_s{cfg.seed}"
    memory = build_memory("peer", agent_ids, env_id,
                          broadcast_every_k=cfg.broadcast_every_k)

    base_query = memory.query
    if cfg.oracle_mode == "oracle_R":
        memory.query = _make_oracle_query(base_query, water_positions, "oracle_R")

    planners = {aid: PlannerState(aid) for aid in agent_ids}
    first_success_tick: Dict[str, Optional[int]] = {aid: None for aid in agent_ids}

    for tick in range(cfg.step_limit):
        for aid in agent_ids:
            obs = env._observation(aid)
            memory.observe(aid, obs.get("cells", []), tick)
        memory.tick(tick)

        for aid in agent_ids:
            if env.agents[aid].success and first_success_tick[aid] is None:
                first_success_tick[aid] = tick

        actions: Dict[str, int] = {}
        for aid in agent_ids:
            if cfg.oracle_mode == "oracle_M":
                # Replace plan_action with oracle version
                actions[aid] = _override_plan_action_for_oracle_M(
                    planners[aid], env, [], tick, "peer", water_positions,
                )
            else:
                targets = memory.query(aid)
                actions[aid] = plan_action(
                    planners[aid], env, targets, tick, "peer",
                )

        result = env.step(actions)

        if result.all_succeeded:
            for aid in agent_ids:
                if env.agents[aid].success and first_success_tick[aid] is None:
                    first_success_tick[aid] = tick + 1
            break

    n_succeeded = sum(1 for v in first_success_tick.values() if v is not None)
    succ_ticks = [v for v in first_success_tick.values() if v is not None]
    mean_t_succ = statistics.mean(succ_ticks) if succ_ticks else float("nan")

    return {
        "n_agents": cfg.n_agents,
        "n_waters": cfg.n_waters,
        "layout": cfg.layout,
        "broadcast_every_k": cfg.broadcast_every_k,
        "hazard_density": cfg.hazard_density,
        "oracle_mode": cfg.oracle_mode,
        "seed": cfg.seed,
        "n_succeeded": n_succeeded,
        "success_rate": n_succeeded / cfg.n_agents,
        "mean_t_succ": mean_t_succ,
        "scarcity": cfg.n_agents / cfg.n_waters,
    }


# ──────────────────────────────────────────────────── sweep


SCARCITY_CASES = [(3, 2), (5, 3), (8, 5)]
LAYOUTS = ["asymmetric", "random"]
KS = [1, 2, 4, 8, 16]
ORACLE_MODES = ["baseline", "oracle_R", "oracle_M"]
HAZARD = 0.05
SEEDS = list(range(15))


def expand_oracle_configs() -> List[OracleRunConfig]:
    configs = []
    for n, m in SCARCITY_CASES:
        for layout in LAYOUTS:
            for k in KS:
                for mode in ORACLE_MODES:
                    for s in SEEDS:
                        configs.append(OracleRunConfig(
                            n_agents=n, n_waters=m, layout=layout,
                            broadcast_every_k=k, hazard_density=HAZARD,
                            oracle_mode=mode, seed=s,
                        ))
    return configs


CSV_FIELDS = [
    "n_agents", "n_waters", "layout", "broadcast_every_k",
    "hazard_density", "oracle_mode", "seed",
    "n_succeeded", "success_rate", "mean_t_succ", "scarcity",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/big_experiment_oracle")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    configs = expand_oracle_configs()
    n_total = len(configs)
    print(f"Oracle sweep: {n_total} configs, workers={args.workers}")
    csv_path = os.path.join(args.out_dir, "oracle_runs.csv")
    t0 = time.time()
    n_done = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_oracle_config, c): c for c in configs}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                    writer.writerow({k: row.get(k) for k in CSV_FIELDS})
                except Exception as e:
                    print(f"  FAILED: {e}", file=sys.stderr)
                n_done += 1
                if n_done % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_done}/{n_total}  elapsed={elapsed:.1f}s")
    elapsed = time.time() - t0
    print(f"✓ Done  {n_done}/{n_total}  elapsed={elapsed:.1f}s  CSV→{csv_path}")


if __name__ == "__main__":
    main()
