"""
VMAS continuous-substrate portability test for Empirical Claim 1 [reviewer #1].

Goal: check whether the M<->C bottleneck-shift slopes (P(M*|R*) falls, P(C*|M*)
rises as cadence K grows) survive when the SAME symbolic peer memory + planner
run on a continuous-state substrate (VMAS) instead of a grid. Continuous
positions are discretised for the memory via experiments.vmas_portability.
continuous_bridge; the controller acts in continuous space toward the
materialised waypoint.

STATUS: runnable scaffold. VMAS (>=1.5) must be installed (`pip install vmas`).
The scenario and controller are intentionally minimal; treat the emitted slopes
as a preliminary portability signal, not a headline result, until the sweep is
scaled and the scenario tuned.

Run: PYTHONPATH=. .venv/bin/python experiments/vmas_portability/run_vmas_portability.py --smoke
"""
from __future__ import annotations
import argparse, math, sys, os
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from scipy import stats

from experiments.vmas_portability.continuous_bridge import build_cells, cell_center_xy, to_cell
from experiments.big_experiment.memory_factory import build_memory

CELL = 0.15          # continuous units per grid cell
SENSE_R = 1          # cell sensing radius (small -> retrieval non-trivial, memory freshness matters)
EPS = 0.12           # arrival tolerance (continuous)
WORLD = 1.0


def _make_scenario(n_agents: int, n_waters: int):
    from vmas.simulator.core import World, Agent, Landmark, Sphere
    from vmas.simulator.scenario import BaseScenario

    class WaterSearch(BaseScenario):
        def make_world(self, batch_dim, device, **kw):
            world = World(batch_dim, device, dim_c=0)
            for i in range(n_agents):
                world.add_agent(Agent(name=f"agent_{i}", shape=Sphere(0.03), collide=False))
            for j in range(n_waters):
                world.add_landmark(Landmark(name=f"water_{j}", collide=False, shape=Sphere(0.04)))
            return world

        def reset_world_at(self, env_index=None):
            for i, ag in enumerate(self.world.agents):
                pos = torch.rand((1 if env_index is not None else self.world.batch_dim, 2),
                                 device=self.world.device) * WORLD
                ag.set_pos(pos, batch_index=env_index)
            for lm in self.world.landmarks:
                pos = torch.rand((1 if env_index is not None else self.world.batch_dim, 2),
                                 device=self.world.device) * WORLD
                lm.set_pos(pos, batch_index=env_index)

        def observation(self, agent):
            return agent.state.pos

        def reward(self, agent):
            return torch.zeros(self.world.batch_dim, device=self.world.device)

    return WaterSearch()


def _run_episode(env, memory, planners, water_xys, n_agents, max_steps, K):
    """Returns per-agent (Q, R, M, C, t_succ) using the grid operational defs."""
    agent_ids = list(planners.keys())
    ev = {aid: dict(Q=0, R=0, M=0, C=0, t=None) for aid in agent_ids}
    claimed: set = set()
    for tick in range(max_steps):
        actions = []
        # sense + observe
        for i, aid in enumerate(agent_ids):
            pos = env.agents[i].state.pos[0].tolist()
            near = [w for w in water_xys if abs(w[0]-pos[0]) <= SENSE_R*CELL and abs(w[1]-pos[1]) <= SENSE_R*CELL]
            cells = build_cells(tuple(pos), near, CELL, radius=SENSE_R, water_tag_dist=CELL)
            memory.observe(aid, cells, tick)
        memory.tick(tick)
        # decide
        for i, aid in enumerate(agent_ids):
            pos = env.agents[i].state.pos[0].tolist()
            ev[aid]["Q"] = 1
            targets = memory.query(aid)  # list of grid (x,y)
            act = torch.zeros((1, 2))
            if targets:
                # R*: a returned candidate is within eps of a real water
                cand_xy = [cell_center_xy(tuple(t), CELL) for t in targets]
                if any(min(math.dist(c, w) for w in water_xys) <= CELL for c in cand_xy):
                    ev[aid]["R"] = 1
                # M*: lock the nearest candidate whose nearest real water is NOT
                # already claimed by a successful peer. Matches the grid planner,
                # which skips occupied targets -> materialisation is occupancy-
                # sensitive, so fast broadcast (shared, quickly-claimed targets)
                # depresses P(M*|R*), the predicted M-slope of Empirical Claim 1.

                def nearest_water_idx(c):
                    return min(range(len(water_xys)), key=lambda i: math.dist(c, water_xys[i]))

                unclaimed = [c for c in cand_xy if nearest_water_idx(c) not in claimed]
                if unclaimed:
                    lock = min(unclaimed, key=lambda c: math.dist(c, tuple(pos)))
                    ev[aid]["M"] = 1
                    dx, dy = lock[0]-pos[0], lock[1]-pos[1]
                    n = math.hypot(dx, dy) or 1.0
                    act = torch.tensor([[dx/n, dy/n]]) * 0.5
            actions.append(act)
        env.step(actions)
        # completion check (respect scarcity via claiming)
        for i, aid in enumerate(agent_ids):
            if ev[aid]["t"] is not None:
                continue
            pos = env.agents[i].state.pos[0].tolist()
            for wi, w in enumerate(water_xys):
                if wi not in claimed and math.dist(pos, w) <= EPS:
                    claimed.add(wi); ev[aid]["C"] = 1; ev[aid]["t"] = tick + 1
                    break
        if all(ev[a]["t"] is not None for a in agent_ids):
            break
    return ev


def sweep(Ks, n_agents, n_waters, seeds, max_steps):
    from vmas import make_env
    from experiments.big_experiment.planner import PlannerState
    rows = []
    for K in Ks:
        for seed in seeds:
            torch.manual_seed(seed)
            scen = _make_scenario(n_agents, n_waters)
            env = make_env(scenario=scen, num_envs=1, continuous_actions=True,
                           max_steps=max_steps, seed=seed)
            env.reset()
            water_xys = [lm.state.pos[0].tolist() for lm in env.world.landmarks]
            agent_ids = [f"a{i}" for i in range(n_agents)]
            memory = build_memory("peer", agent_ids, f"vmas_K{K}_s{seed}", broadcast_every_k=K)
            planners = {aid: PlannerState(aid) for aid in agent_ids}
            ev = _run_episode(env, memory, planners, water_xys, n_agents, max_steps, K)
            for aid, e in ev.items():
                rows.append(dict(K=K, seed=seed, **{k: e[k] for k in "QRMC"},
                                 t=e["t"] if e["t"] is not None else max_steps))
    return rows


def summarize(rows):
    import collections
    byK = collections.defaultdict(list)
    for r in rows:
        byK[r["K"]].append(r)
    print(f"{'K':>4} | {'P(M|R)':>7} {'P(C|M)':>7} {'mean_t':>7}")
    Ks, pmr, pcm = [], [], []
    for K in sorted(byK):
        rs = byK[K]
        nR = sum(r["R"] for r in rs); nM = sum(r["M"] for r in rs); nC = sum(r["C"] for r in rs)
        p_m_r = nM / nR if nR else float("nan")
        p_c_m = nC / nM if nM else float("nan")
        mt = sum(r["t"] for r in rs) / len(rs)
        print(f"{K:>4} | {p_m_r:>7.3f} {p_c_m:>7.3f} {mt:>7.2f}")
        Ks.append(K); pmr.append(p_m_r); pcm.append(p_c_m)
    if len(set(Ks)) > 2:
        print(f"\nSpearman P(M|R) vs K = {stats.spearmanr(Ks,pmr).correlation:+.3f} (Claim 1 predicts < 0)")
        print(f"Spearman P(C|M) vs K = {stats.spearmanr(Ks,pcm).correlation:+.3f} (Claim 1 predicts > 0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to check it executes")
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--n_waters", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=80)
    a = ap.parse_args()
    if a.smoke:
        a.Ks, a.seeds, a.max_steps = [1, 16], 2, 40
    rows = sweep(a.Ks, a.n_agents, a.n_waters, list(range(a.seeds)), a.max_steps)
    summarize(rows)
