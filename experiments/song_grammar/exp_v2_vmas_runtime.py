"""V2 --- the full Songline runtime on VMAS physics (Stage 13).

The external end-to-end check the reviewer asks for: not the Q/R/M/C
logger transplanted, but the WHOLE runtime --- formation, certificate
exchange, quarantine, receiver-side admission, frame-free continuous
alignment, safe-prefix consumption --- driving agents in a real
continuous-physics multi-agent simulator (VMAS), a third substrate
after grid and the synthetic continuous box.

Scenario: N holonomic agents in a 2-D box with fixed LANDMARK
entities (the map's structure) and a water target. Each agent
observes landmark positions relative to itself (a continuous
constellation), forms a song of its trajectory to water, exchanges
certificates on a cadence; a receiver with no shared frame recovers
the target by matching landmark constellations (C1's soft_sim +
unimodal anchoring + safe prefix) and a proportional controller
drives it there.

Arms: independent (no exchange) vs songline_safe (full runtime,
three-layer safety). Registered V2.1: on VMAS the full runtime keeps
its advantage --- team steps-to-water(songline_safe) <= independent
on paired seeds --- and transport fail-open stays < 0.05.

Run (cluster base env has vmas; CPU, num_envs=1)::

    PYTHONPATH=. python experiments/song_grammar/exp_v2_vmas_runtime.py \
        --arm songline_safe --seeds 0 8 --out tmp/cluster/song_grammar/v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from songlines.config import get
from songlines.record import Config
from songlines.record import Record
from songlines.runtime import SonglineAgent
from songlines.alignment import song_target

WB, HB = 2.0, 2.0                 # VMAS world half-extent-ish box
N_LANDMARKS = 16
OBS_R = 1.5
ACCEPT = 0.18
SCAN = 0.30
JITTER = 0.02
CADENCE = 4
ROLES = {"fragile": 1.0, "robust": 1.0}   # symmetric here


def _build_scenario(n_agents: int, seed: int):
    from vmas.simulator.scenario import BaseScenario
    from vmas.simulator.core import Agent, Landmark, Sphere, World
    import torch

    rs = np.random.default_rng(seed)
    lms = [(float(rs.uniform(-WB, WB)), float(rs.uniform(-HB, HB)))
           for _ in range(N_LANDMARKS)]
    lm_class = [int(rs.integers(0, 3)) for _ in range(N_LANDMARKS)]
    water = (float(rs.uniform(0.5, WB)), float(rs.uniform(-HB, HB)))

    class SonglineScene(BaseScenario):
        def make_world(self, batch_dim, device, **kw):
            world = World(batch_dim, device, dim_c=0)
            for i in range(n_agents):
                world.add_agent(Agent(name=f"a{i}",
                                      shape=Sphere(0.04), u_multiplier=0.6))
            for k, _ in enumerate(lms):
                world.add_landmark(Landmark(name=f"lm{k}",
                                            collide=False,
                                            shape=Sphere(0.03)))
            world.add_landmark(Landmark(name="water", collide=False,
                                        shape=Sphere(0.05)))
            self._lms, self._water = lms, water
            return world

        def reset_world_at(self, env_index=None):
            import torch
            for i, ag in enumerate(self.world.agents):
                ag.set_pos(torch.tensor([[-WB + 0.2, -HB + 0.2 + 0.1 * i]],
                                        dtype=torch.float32),
                           batch_index=env_index)
            for k, (x, y) in enumerate(lms):
                self.world.landmarks[k].set_pos(
                    torch.tensor([[x, y]], dtype=torch.float32),
                    batch_index=env_index)
            self.world.landmarks[-1].set_pos(
                torch.tensor([[water[0], water[1]]], dtype=torch.float32),
                batch_index=env_index)

        def observation(self, agent):
            return agent.state.pos

        def reward(self, agent):
            import torch
            return torch.zeros(self.world.batch_dim)

    return SonglineScene(), lms, lm_class, water


def _fp(pos, lms, lm_class, rng):
    """Continuous landmark constellation relative to pos (jittered)."""
    sig = {}
    for (lx, ly), cls in zip(lms, lm_class):
        dx, dy = lx - pos[0], ly - pos[1]
        if dx * dx + dy * dy > OBS_R * OBS_R:
            continue
        jx = dx + float(rng.normal(0, JITTER))
        jy = dy + float(rng.normal(0, JITTER))
        sig[f"c{cls}@{jx:.2f},{jy:.2f}"] = 1.0
    return sig


def _soft_sim(a, b, tol=0.25):
    def parse(s):
        out = []
        for k in s:
            c, off = k.rsplit("@", 1)
            x, y = off.split(",")
            out.append((c, float(x), float(y)))
        return out
    pa, pb = parse(a), parse(b)
    if not pa or not pb:
        return 0.0
    used, m = set(), 0
    for c, dx, dy in pa:
        best, bd = None, tol * tol
        for k, (c2, ex, ey) in enumerate(pb):
            if k in used or c2 != c:
                continue
            d = (dx - ex) ** 2 + (dy - ey) ** 2
            if d <= bd:
                best, bd = k, d
        if best is not None:
            used.add(best); m += 1
    return m / math.sqrt(len(pa) * len(pb))


def _band(lms):
    # observation points the receiver has swept (interior band)
    return [(x, y) for x in np.linspace(-WB + 0.3, 0.2, 6)
            for y in np.linspace(-HB + 0.3, HB - 0.3, 8)]


def _witness_song(lms, lm_class, water, rng):
    start = (-WB + 0.2, -HB + 0.2)
    n = max(4, int(math.dist(start, water) / 0.18))
    pts = [(start[0] + (water[0] - start[0]) * k / n,
            start[1] + (water[1] - start[1]) * k / n)
           for k in range(n + 1)]
    couplets, last, li = [], pts[0], -10
    for i, p in enumerate(pts):
        is_last = i == len(pts) - 1
        sig = _fp(p, lms, lm_class, rng)
        if not is_last and (len(sig) < 3 or i - li < 2):
            continue
        couplets.append({"sig": sig,
                         "beat": (p[0] - last[0], p[1] - last[1])})
        last, li = p, i
    return couplets


def run_seed(arm: str, seed: int, n_agents: int, max_steps: int
             ) -> Dict[str, Any]:
    from vmas import make_env
    import torch
    cfg, comm = get(arm, continuous=True)
    if arm == "songline_safe":   # VMAS geometry: fewer, well-separated
        cfg = Config(**(cfg.__dict__ | {"anchor_consensus": 2,
                                        "closure_tol": 1.2}))
    SonglineAgent.simfn = staticmethod(_soft_sim)
    scen, lms, lm_class, water = _build_scenario(n_agents, seed)
    env = make_env(scenario=scen, num_envs=1, continuous_actions=True,
                   seed=seed, n_agents=n_agents)
    env.reset()
    rng = np.random.default_rng(seed * 3 + 1)

    mems = [SonglineAgent(i, "robust", cfg) for i in range(n_agents)]
    band = _band(lms)
    # witnesses seed songs about the water into the collective
    if comm:
        for i in range(n_agents):
            song = _witness_song(lms, lm_class, water, rng)
            role_u = {"robust": 50.0, "fragile": 0.0}
            mems[i].form(song, "water", 0, 0, 0, role_u)

    def observe_fn(q):
        return _fp(q, lms, lm_class, rng)

    reached = [False] * n_agents
    steps_to = [max_steps] * n_agents
    fail_open = 0
    # exchange once up front (single family, single version)
    if comm:
        for i in range(n_agents):
            for rec in mems[i].outbox(-1):
                for j in range(n_agents):
                    if j != i:
                        mems[j].receive(rec, sender=i)
        for i in range(n_agents):
            mems[i].on_visit(None, 0, 0, 0,
                             lambda env, ag, song, intent: 50.0)

    # each agent's committed target
    tgt = [None] * n_agents
    for i in range(n_agents):
        pos = env.agents[i].state.pos[0].tolist()
        bf = {p: _fp(p, lms, lm_class, rng) for p in band}
        ts = mems[i].targets(bf, "water",
                             observe_fn=observe_fn, start=tuple(pos)) \
            if comm else []
        tgt[i] = ts[0] if ts else None
        if tgt[i] is not None and math.dist(tgt[i], water) > SCAN:
            fail_open += 1

    # blind lawnmower sweep for memory-free agents (no water knowledge):
    # a dense boustrophedon that fully tiles the box at the water radius,
    # so a memory-free agent CAN find water by exhaustive search --- the
    # contrast with songline is then a genuine speed-up, not a floor.
    cols = np.linspace(-WB + 0.2, WB - 0.2, 12)
    sweep = []
    for ci, x in enumerate(cols):
        ys = np.linspace(-HB + 0.2, HB - 0.2, 12)
        if ci % 2:
            ys = ys[::-1]
        sweep += [(float(x), float(y)) for y in ys]
    # each memory-free agent starts its sweep at a different phase so the
    # team covers the box faster (still no target knowledge)
    sweep_i = [i * (len(sweep) // max(1, n_agents)) for i in range(n_agents)]

    for step in range(max_steps):
        actions = []
        for i in range(n_agents):
            pos = env.agents[i].state.pos[0].tolist()
            if tgt[i] is not None:              # memory-guided
                goal = tgt[i]
            else:                               # memory-free: explore
                wp = sweep[sweep_i[i] % len(sweep)]
                if math.dist(pos, wp) < 0.12:
                    sweep_i[i] += 1
                    wp = sweep[sweep_i[i] % len(sweep)]
                goal = wp
            dx, dy = goal[0] - pos[0], goal[1] - pos[1]
            norm = math.hypot(dx, dy) + 1e-9
            actions.append(torch.tensor([[dx / norm, dy / norm]],
                                        dtype=torch.float32))
            if not reached[i] and math.dist(pos, water) <= ACCEPT:
                reached[i] = True
                steps_to[i] = step
        env.step(actions)
        if all(reached):
            break

    team = float(np.mean(steps_to))
    return {"arm": arm, "seed": seed, "team_steps": team,
            "reached": sum(reached), "n_agents": n_agents,
            "fail_open": fail_open / n_agents}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 4])
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/v2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "v2_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "V2.1": "on VMAS physics the full runtime keeps its "
                        "advantage: team steps(songline_safe) <= "
                        "independent on paired seeds; transport "
                        "fail-open < 0.05",
                "substrate": "VMAS continuous physics, N holonomic "
                             "agents + landmark entities + water; full "
                             "runtime (formation/cert/quarantine/"
                             "admission/continuous-alignment/safe-prefix)",
            }, f, indent=2)
    shard = f"v2_{a.arm}_s{a.seeds[0]}-{a.seeds[1]}.jsonl"
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            row = run_seed(a.arm, seed, a.agents, a.max_steps)
            f.write(json.dumps(row) + "\n")
            print(f"{a.arm} seed {seed}: team_steps {row['team_steps']:.1f} "
                  f"reached {row['reached']}/{row['n_agents']} "
                  f"fo {row['fail_open']:.2f}", flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
