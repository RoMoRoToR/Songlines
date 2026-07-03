"""W5a — warp portability to a continuous substrate (VMAS).

The warp layer depends only on the memory adapter, not on the world:
phi is computed from the merge's own weights.  This experiment ports
the W*-annotation to the VMAS water-search scenario through the same
continuous->grid bridge used by the Claim-1 portability test, and
checks three REGISTERED predictions (written to disk before episodes):

  P1 (strata):     P(C*|W*) < P(C*|M*, own) under scarcity, disjoint
                   bootstrap CIs, on continuous dynamics.
  P2 (share):      warp share P(W*|M*) decreases as cadence K grows.
  P3 (metric law): the witness-traveler breakpoint on the initial-age
                   axis equals floor(age_max(trust) - t_sight(D)),
                   where t_sight is measured once per distance on the
                   a0=0 calibration episode (folding speed, drag and
                   discretisation into one constant), and age_max is
                   the CSM closed form of the grid paper — unchanged.

Part A: peer memory, K in {1,4,16}, (N,M) in {(4,2),(8,3)}, 20 seeds.
Part B: CSM witness-traveler in continuous space, trust in {1.0, 0.6},
        travel distance D in {0.45, 0.75} world units.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_vmas.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import torch

from experiments.big_experiment.memory_factory import build_memory
from experiments.vmas_portability.continuous_bridge import (
    build_cells, cell_center_xy,
)
from experiments.vmas_portability.run_vmas_portability import (
    CELL, EPS, SENSE_R, WORLD, _make_scenario,
)
from experiments.warp.exp_warp_age_law import (
    CSMWitnessMemory, predicted_age_max,
)
from experiments.warp.warp_instrumentation import (
    ProvenanceLedger, candidate_provenance,
)

OUT_DIR = "tmp/warp/w5_vmas"
KS = [1, 4, 16]
NM_CELLS = [(4, 2), (8, 3)]
MAX_STEPS = 80
TRUSTS_B = [1.0, 0.6]
DISTANCES_B = [0.45, 0.75]
AGES_B = list(range(0, 25, 2))


def _spread_positions(n: int, rng: np.random.Generator) -> List[List[float]]:
    return [[float(rng.uniform(0.05, WORLD - 0.05)),
             float(rng.uniform(0.05, WORLD - 0.05))] for _ in range(n)]


def _make_env(n_agents: int, n_waters: int, seed: int, max_steps: int):
    from vmas import make_env
    torch.manual_seed(seed)
    scen = _make_scenario(n_agents, n_waters)
    env = make_env(scenario=scen, num_envs=1, continuous_actions=True,
                   max_steps=max_steps, seed=seed)
    env.reset()
    return env


# ─────────────────────────────────── Part A: strata by K


def run_episode_a(env, memory, agent_ids, water_xys,
                  max_steps: int) -> List[Dict[str, Any]]:
    """One warp-annotated episode; returns closed M*-event dicts."""
    ledger = ProvenanceLedger()
    claimed: set = set()
    succeeded: Dict[str, bool] = {aid: False for aid in agent_ids}
    open_ev: Dict[str, Optional[Dict]] = {aid: None for aid in agent_ids}
    events: List[Dict[str, Any]] = []

    def close(aid):
        if open_ev[aid] is not None:
            events.append(open_ev[aid])
            open_ev[aid] = None

    for tick in range(max_steps):
        for i, aid in enumerate(agent_ids):
            pos = env.agents[i].state.pos[0].tolist()
            near = [w for w in water_xys
                    if abs(w[0] - pos[0]) <= SENSE_R * CELL
                    and abs(w[1] - pos[1]) <= SENSE_R * CELL]
            cells = build_cells(tuple(pos), near, CELL, radius=SENSE_R,
                                water_tag_dist=CELL)
            ledger.record(aid, cells, tick)
            memory.observe(aid, cells, tick)
        memory.tick(tick)

        actions = []
        for i, aid in enumerate(agent_ids):
            pos = env.agents[i].state.pos[0].tolist()
            act = torch.zeros((1, 2))
            if not succeeded[aid]:
                targets = memory.query(aid)
                cand = [(tuple(t), cell_center_xy(tuple(t), CELL))
                        for t in targets]

                def nw(cxy):
                    return min(range(len(water_xys)),
                               key=lambda j: math.dist(cxy, water_xys[j]))

                unclaimed = [(g, c) for g, c in cand if nw(c) not in claimed]
                if unclaimed:
                    grid, cxy = min(unclaimed,
                                    key=lambda gc: math.dist(gc[1], pos))
                    cur = open_ev[aid]
                    if cur is None or tuple(cur["target_cell"]) != grid:
                        close(aid)
                        prov = candidate_provenance(
                            memory, aid, grid, ledger, tick,
                            n_agents=len(agent_ids))
                        open_ev[aid] = {
                            "agent": aid, "tick": tick,
                            "target_cell": list(grid),
                            "phi": prov.phi,
                            "w_star_soft": prov.phi >= 0.8,
                            "w_star_strict": (prov.phi >= 0.999 and
                                              not ledger.self_observed(aid, grid)),
                            "completed": False,
                        }
                    dx, dy = cxy[0] - pos[0], cxy[1] - pos[1]
                    n = math.hypot(dx, dy) or 1.0
                    act = torch.tensor([[dx / n, dy / n]]) * 0.5
                else:
                    close(aid)
                    vel = env.agents[i].state.vel[0]
                    act = (-vel).unsqueeze(0).clamp(-0.5, 0.5)
            actions.append(act)
        env.step(actions)

        for i, aid in enumerate(agent_ids):
            if succeeded[aid]:
                continue
            pos = env.agents[i].state.pos[0].tolist()
            for wi, w in enumerate(water_xys):
                if wi not in claimed and math.dist(pos, w) <= EPS:
                    claimed.add(wi)
                    succeeded[aid] = True
                    ev = open_ev[aid]
                    if ev is not None and math.dist(
                            cell_center_xy(tuple(ev["target_cell"]), CELL),
                            w) <= 2 * CELL:
                        ev["completed"] = True
                    close(aid)
                    break
        if all(succeeded.values()):
            break
    for aid in agent_ids:
        close(aid)
    return events


def part_a(seeds: int) -> Dict[str, Any]:
    per_k: Dict[int, List[List[Dict]]] = {k: [] for k in KS}
    for (n, m) in NM_CELLS:
        for k in KS:
            for seed in range(seeds):
                env = _make_env(n, m, seed, MAX_STEPS)
                rng = np.random.default_rng(10_000 + seed)
                for i, ag in enumerate(env.world.agents):
                    ag.set_pos(torch.tensor(
                        [_spread_positions(1, rng)[0]]), batch_index=0)
                water_xys = [lm.state.pos[0].tolist()
                             for lm in env.world.landmarks]
                agent_ids = [f"a{i}" for i in range(n)]
                memory = build_memory("peer", agent_ids,
                                      f"w5_{n}_{m}_{k}_{seed}",
                                      broadcast_every_k=k)
                events = run_episode_a(env, memory, agent_ids, water_xys,
                                       MAX_STEPS)
                per_k[k].append(events)
            print(f"  part A: N{n}M{m} K={k} done")

    def boot(pairs, seed=0, n_boot=4000):
        pairs = [p for p in pairs if p]
        if not pairs:
            return (float("nan"),) * 3
        rng = np.random.default_rng(seed)
        point = float(np.mean([v for p in pairs for v in p]))
        st = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(pairs), len(pairs))
            pooled = [v for i in idx for v in pairs[i]]
            if pooled:
                st.append(np.mean(pooled))
        lo, hi = np.percentile(st, [2.5, 97.5])
        return point, float(lo), float(hi)

    out: Dict[str, Any] = {}
    for k in KS:
        eps_k = per_k[k]
        out[str(k)] = {
            "p_C_given_W": boot([[int(e["completed"]) for e in ep
                                  if e["w_star_soft"]] for ep in eps_k],
                                seed=k),
            "p_C_given_own": boot([[int(e["completed"]) for e in ep
                                    if not e["w_star_soft"]] for ep in eps_k],
                                  seed=50 + k),
            "warp_share": boot([[int(e["w_star_soft"]) for e in ep]
                                for ep in eps_k], seed=100 + k),
            "n_events": sum(len(ep) for ep in eps_k),
        }
    return out


# ─────────────────────────────────── Part B: metric distance law


def run_episode_b(trust: float, a0: int, dist: float,
                  max_steps: int = 70) -> Dict[str, Any]:
    env = _make_env(1, 1, seed=0, max_steps=max_steps)
    start = [0.1, 0.5]
    water = [0.1 + dist, 0.5]
    env.world.agents[0].set_pos(torch.tensor([start]), batch_index=0)
    env.world.landmarks[0].set_pos(torch.tensor([water]), batch_index=0)

    from experiments.vmas_portability.continuous_bridge import to_cell
    water_cell = to_cell(tuple(water), CELL)
    memory = CSMWitnessMemory(initial_age=a0, trust=trust,
                              water_xy=water_cell, broadcast_every_k=8)

    self_seen_tick = None
    succ_tick = None
    locked_ever = False
    ag = env.world.agents[0]
    for tick in range(max_steps):
        pos = ag.state.pos[0].tolist()
        near = [tuple(water)] if (abs(water[0] - pos[0]) <= SENSE_R * CELL
                                  and abs(water[1] - pos[1]) <= SENSE_R * CELL) \
            else []
        cells = build_cells(tuple(pos), near, CELL, radius=SENSE_R,
                            water_tag_dist=CELL)
        memory.observe("traveler", cells, tick)
        memory.tick(tick)
        if near and self_seen_tick is None:
            self_seen_tick = tick

        targets = memory.query("traveler")
        if targets:
            locked_ever = True
            cxy = min((cell_center_xy(tuple(t), CELL) for t in targets),
                      key=lambda c: math.dist(c, pos))
            dx, dy = cxy[0] - pos[0], cxy[1] - pos[1]
            n = math.hypot(dx, dy) or 1.0
            act = torch.tensor([[dx / n, dy / n]]) * 0.5
        else:
            # no lock -> brake.  Matches the grid semantics (no target, no
            # movement); without this, inertia lets the agent coast past
            # the gate closure and the discrete feasibility model breaks.
            vel = ag.state.vel[0]
            act = (-vel).unsqueeze(0).clamp(-0.5, 0.5)
        env.step([act])
        if math.dist(ag.state.pos[0].tolist(), water) <= EPS:
            succ_tick = tick + 1
            break
    return {"trust": trust, "a0": a0, "distance": dist,
            "locked": locked_ever, "t_sight": self_seen_tick,
            "t_succ": succ_tick, "completed": succ_tick is not None}


def part_b() -> Dict[str, Any]:
    # calibration first: a0 = 0 per distance folds speed/drag into t_sight
    calib: Dict[float, Dict] = {}
    for d in DISTANCES_B:
        calib[d] = run_episode_b(1.0, 0, d)
    predictions = {}
    for trust in TRUSTS_B:
        gate = predicted_age_max(trust)
        for d in DISTANCES_B:
            ts = calib[d]["t_sight"]
            if ts is None:
                # even at a0 = 0 the gate closes before the traveler can
                # sight the water: D exceeds the metric warp radius ->
                # predicted NEVER at any age.
                bp = -1
            else:
                bp = math.floor(gate - ts)
                if bp < 0:
                    bp = -1
            predictions[f"trust={trust}|D={d}"] = {
                "age_max": round(gate, 2), "t_sight_calibrated": ts,
                "predicted_breakpoint": bp,
            }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "w5_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    print("  part B predictions registered:",
          {k: v["predicted_breakpoint"] for k, v in predictions.items()})

    results = {}
    for trust in TRUSTS_B:
        for d in DISTANCES_B:
            succ_ages = [a0 for a0 in AGES_B
                         if run_episode_b(trust, a0, d)["completed"]]
            emp = max(succ_ages) if succ_ages else -1
            pred = predictions[f"trust={trust}|D={d}"]["predicted_breakpoint"]
            results[f"trust={trust}|D={d}"] = {
                "empirical_breakpoint": emp, "predicted_breakpoint": pred,
                "within_grid_step": (pred is not None
                                     and abs(emp - pred) <= 2),
                "monotone": succ_ages == [a for a in AGES_B if a <= emp],
            }
            print(f"  part B: trust={trust} D={d} emp={emp} pred={pred}")
    return {"calibration": {str(k): v for k, v in calib.items()},
            "predictions": predictions, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    seeds = 3 if a.smoke else a.seeds
    os.makedirs(OUT_DIR, exist_ok=True)

    # register Part A qualitative predictions before running
    reg = {
        "P1": "P(C*|W*) < P(C*|M*,own), disjoint bootstrap CIs (pooled)",
        "P2": "warp share P(W*|M*) decreases with K over {1,4,16}",
        "P3": "see w5_predictions.json (metric law, calibrated t_sight)",
    }
    with open(os.path.join(OUT_DIR, "w5_registered.json"), "w") as f:
        json.dump(reg, f, indent=2)

    print("Part A: strata by K on VMAS …")
    strata = part_a(seeds)
    for k in KS:
        s = strata[str(k)]
        print(f"  K={k:<3} share={s['warp_share'][0]:.3f} "
              f"P(C|W)={s['p_C_given_W'][0]:.3f} "
              f"[{s['p_C_given_W'][1]:.3f},{s['p_C_given_W'][2]:.3f}] "
              f"P(C|own)={s['p_C_given_own'][0]:.3f} "
              f"[{s['p_C_given_own'][1]:.3f},{s['p_C_given_own'][2]:.3f}] "
              f"n={s['n_events']}")

    print("Part B: metric distance law (witness-traveler, continuous) …")
    law = part_b()

    # verdicts
    pooled_w = [strata[str(k)]["p_C_given_W"] for k in KS]
    pooled_o = [strata[str(k)]["p_C_given_own"] for k in KS]
    p1 = all(w[2] < o[1] for w, o in zip(pooled_w, pooled_o)
             if w[2] == w[2] and o[1] == o[1])
    shares = [strata[str(k)]["warp_share"][0] for k in KS]
    p2 = shares[0] > shares[-1]
    p3 = all(v["within_grid_step"] for v in law["results"].values())

    verdict = {"P1_strata_disjoint": p1, "P2_share_decreases": p2,
               "P3_metric_law": p3}
    with open(os.path.join(OUT_DIR, "w5_results.json"), "w") as f:
        json.dump({"strata": strata, "law": law, "verdict": verdict},
                  f, indent=2)
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/w5_results.json")


if __name__ == "__main__":
    main()
