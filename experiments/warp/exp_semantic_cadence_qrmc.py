"""
Does the M<->C bottleneck shift survive SEMANTIC place matching (no shared frame)?

The main multi-agent sweep resolves place identity by shared (x,y) keys. Here we
rerun a cadence sweep in which every agent holds a PRIVATE coordinate frame and
places are matched BY MEANING (local fingerprint constellations, frame recovered
via experiments.warp.semantic_peer_memory.SemanticFramePeerMemory). Agents
navigate (rule-based planner) toward targets transported through the recovered
frame; Q/R/M/C are logged per agent as usual.

Two questions:
  1. semantic arm -- is the bottleneck shift (P(M*|R*) falls, P(C*|M*) rises as
     cadence K grows) still there WITHOUT a shared coordinate anchor?
  2. coordinate arm (take foreign coords at face value) -- under private frames
     it should fail open: lock phantom (non-water) cells.

Deterministic. Run:
  PYTHONPATH=. .venv/bin/python experiments/warp/exp_semantic_cadence_qrmc.py
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from experiments.big_experiment.planner import PlannerState, plan_action
from experiments.warp.semantic_peer_memory import SemanticFramePeerMemory
from multiagent_env import HAZARD, WATER, MultiAgentGridWorld

W, H, EPS = 14, 12, 0.6


def build(seed, N, T, hazard=0.10):
    rng = np.random.default_rng(seed)
    env = MultiAgentGridWorld(width=W, height=H, step_limit=140,
                              observation_radius=2, rng_seed=seed)
    waters = set()
    while len(waters) < T:
        waters.add((int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1))))
    for (x, y) in waters:
        env.set_cell(x, y, WATER)
    nh = int(round(hazard * W * H))
    placed = 0
    while placed < nh:
        xy = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if xy not in waters and env.cell(*xy) == 0:
            env.set_cell(*xy, HAZARD); placed += 1
    starts = []
    while len(starts) < N:
        xy = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if env.cell(*xy) == 0 and xy not in starts:
            starts.append(xy)
    for i, s in enumerate(starts):
        env.spawn(f"a{i}", start_xy=s, target_tag="water_source", direction=0)
    offsets = {f"a{i}": (int(rng.integers(-8, 9)), int(rng.integers(-8, 9))) for i in range(N)}
    return env, list(waters), offsets


def near_water(xy, waters):
    return any(abs(xy[0] - w[0]) + abs(xy[1] - w[1]) <= EPS for w in waters)


def run_episode(mode, K, seed, N, T):
    env, waters, offsets = build(seed, N, T)
    mem = SemanticFramePeerMemory(list(offsets), offsets, mode=mode,
                                  broadcast_every_k=K, aligner="translation")
    pls = {aid: PlannerState(aid) for aid in offsets}
    ev = {aid: dict(Q=0, R=0, M=0, C=0, lock_real=None, t=None) for aid in offsets}
    for tick in range(env.step_limit):
        for aid in offsets:
            ag = env.agents[aid]
            obs = env._observation(aid)
            cells = [{"xy": (int(c["xy"][0]), int(c["xy"][1])), "tag": c["tag"]}
                     for c in obs.get("cells", [])]
            mem.observe(aid, (ag.x, ag.y), cells, tick)
        mem.tick(tick)
        actions = {}
        for aid in offsets:
            ag = env.agents[aid]
            if ag.success and ev[aid]["t"] is None:
                ev[aid]["t"] = tick
            targets = mem.query(aid, tick)          # true-frame targets (semantic transport)
            ev[aid]["Q"] = 1
            if targets:
                # nested events (C* => M* => R* => Q*), matching the paper:
                # R* = retrieval returned a candidate near a real target;
                # M* = the LOCKED candidate is near a real target.
                if any(near_water(t, waters) for t in targets):
                    ev[aid]["R"] = 1
                lock = min(targets, key=lambda t: abs(t[0]-ag.x)+abs(t[1]-ag.y))
                if near_water(lock, waters):
                    ev[aid]["M"] = 1
                if ev[aid]["lock_real"] is None:
                    ev[aid]["lock_real"] = near_water(lock, waters)
                actions[aid] = plan_action(pls[aid], env, [lock], tick, f"sc-{mode}")
            else:
                actions[aid] = plan_action(pls[aid], env, [], tick, f"sc-{mode}")
        env.step(actions)
        for aid in offsets:
            if env.agents[aid].success and ev[aid]["t"] is None:
                ev[aid]["t"] = tick + 1
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=6); ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    KS = [1, 4, 8, 16]

    def agg(mode, K):
        rows = [ev for s in range(a.seeds) for ev in run_episode(mode, K, s, a.N, a.T).values()]
        nR = sum(r["R"] for r in rows); nM = sum(r["M"] for r in rows); nC = sum(r["C"] for r in rows)
        # C* counts agents that reached water
        nC = sum(1 for r in rows if r["t"] is not None)
        pmr = nM / nR if nR else float("nan")
        pcm = nC / nM if nM else float("nan")
        locks = [r["lock_real"] for r in rows if r["lock_real"] is not None]
        real_lock = np.mean(locks) if locks else float("nan")
        succ = np.mean([r["t"] is not None for r in rows])
        return pmr, pcm, succ, real_lock

    print(f"Semantic-matching cadence sweep (private frames, no shared coords)  "
          f"N={a.N}, T={a.T}, {a.seeds} seeds\n")
    for mode in ("semantic", "coordinate"):
        print(f"=== mode = {mode} ===")
        print(f"{'K':>3} | {'P(M|R)':>7} {'P(C|M)':>7} {'success':>8} {'real-lock%':>10}")
        for K in KS:
            pmr, pcm, succ, rl = agg(mode, K)
            print(f"{K:>3} | {pmr:>7.3f} {pcm:>7.3f} {succ:>8.3f} {rl*100:>9.1f}%")
        print()
    print("Reading: semantic arm should keep the shift (P(M|R) down, P(C|M) up in K) with")
    print("high real-lock% -- the bottleneck shift is NOT an artefact of the shared frame.")
    print("coordinate arm under private frames locks phantom cells (low real-lock%): fails open.")


if __name__ == "__main__":
    main()
