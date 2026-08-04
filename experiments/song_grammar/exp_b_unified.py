"""B-unified --- the single equal-budget benchmark (Stages 10+11).

Runs, on identical paired worlds with identical memory/wire budgets:
  independent, songline_full  (the method, via SonglineAgent)
  decision_centric, execution_path, graph_memory, learned_formation
                              (direct baselines, experiments/.../baselines.py)

Same two-intent drifting world, private frames, N agents, 300
episodes as I1. Headline columns: team cost (mean of roles), first-
lock success, fail-open, mean memory bits, wire bits. Registered
hypothesis BU.1: full Songlines <= the best direct baseline on team
cost at equal budget on unseen (test 100+) worlds.

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_b_unified.py \
        --policy songline_full --seeds 100 112 --episodes 300 \
        --agents 6 --out tmp/cluster/song_grammar/bench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.song_grammar.exp_s0_song_smoke import BAND, TRAVELER_START, fp_at
from experiments.song_grammar.exp_i1_integration import (
    CADENCE, CONTENTION, INTENTS, World, build_song_cfg, make_fp, walk)
from experiments.song_grammar.baselines import BASELINES
from experiments.song_grammar.runtime import Config, SonglineAgent, record_bits
from experiments.song_grammar.u7_common import ROLES, dijkstra

GridXY = Tuple[int, int]
SONGLINE = {"independent", "songline_full"}


class SonglineArm:
    """Adapter giving SonglineAgent the baseline observe/target API."""
    def __init__(self, aid, role_name, cfg, comm):
        self.ag = SonglineAgent(aid, role_name, cfg)
        self.comm = comm

    def observe(self, env, intent, song, fam, ver, t, role_name,
                utility_fn):
        self.ag.now = t
        self.ag.on_visit(env, fam, ver, t, utility_fn)
        role_u = {rn: utility_fn(env, self.ag, song, intent)
                  if rn == role_name else 0.0 for rn in ROLES}
        self.ag.form(song, intent, fam, ver, t, role_u)

    def targets(self, band_fps, intent, role_name):
        return self.ag.targets(band_fps, intent)

    def outbox(self, since):
        return [{"song": r.song, "intent": r.intent, "fam": r.family,
                 "role_u": r.role_u, "origin": r.origin, "uid": r.uid,
                 "t": r.t, "version": r.version, "_rec": r}
                for r in self.ag.outbox(since)]

    def receive(self, rec, sender):
        self.ag.receive(rec["_rec"], sender=sender)

    def memory_bits(self):
        return self.ag.memory_bits()

    def wire_bits(self, rec):
        return record_bits(rec["_rec"], self.ag.cfg)


def make_policy(name, aid, role_name):
    if name in SONGLINE:
        cfg = Config()
        return SonglineArm(aid, role_name, cfg, name != "independent")
    cls = BASELINES[name]
    p = cls(aid, role_name)
    return p


def run_cell(policy: str, seed: int, n_agents: int, n_episodes: int
             ) -> Dict[str, Any]:
    comm = policy != "independent"
    roles = ["fragile" if i % 2 == 0 else "robust"
             for i in range(n_agents)]
    agents = [make_policy(policy, i, roles[i]) for i in range(n_agents)]
    world = World(seed)
    stats = {"cost": {r: [] for r in ROLES}, "succ": 0, "n": 0,
             "phantom": 0, "commits": 0, "dup": 0, "wire": 0}

    def utility_fn(env, agent, song, intent):
        from songlines.alignment import song_target
        bf = {xy: fp_at(env, xy) for xy in BAND}
        # baselines expose .targets; SonglineAgent handed directly
        role = ROLES[getattr(agent, "role_name", "robust")]
        kind = INTENTS[intent]
        cand = song_target(song, bf, 0.999)
        if hasattr(agent, "cfg"):          # SonglineAgent (2-arg)
            base = agent.targets(bf, intent)
        else:                              # baseline policy (3-arg)
            base = agent.targets(bf, intent,
                                 getattr(agent, "role_name", "robust"))
        with_m = walk(env, base + ([cand] if cand else []), role, kind)["cost"]
        without = walk(env, base, role, kind)["cost"]
        return without - with_m

    for t in range(n_episodes):
        assignments = [world.assign() for _ in range(n_agents)]
        ep_targets: Dict[GridXY, int] = {}
        for i, (fam, env, tg, ver) in enumerate(assignments):
            intent = "water" if (t + i) % 3 else "rest"
            bf = {xy: fp_at(env, xy) for xy in BAND}
            tgts = agents[i].targets(bf, intent, roles[i])
            r = walk(env, tgts, ROLES[roles[i]], INTENTS[intent])
            cost = r["cost"]
            if tgts:
                stats["commits"] += 1
                first = tgts[0]
                if ep_targets.get(first, i) != i:
                    stats["dup"] += 1; cost += CONTENTION
                else:
                    ep_targets[first] = i
            stats["cost"][roles[i]].append(cost)
            stats["succ"] += int(r["success_first"])
            stats["phantom"] += int(r["phantom"])
            stats["n"] += 1
            path, _ = dijkstra(env, TRAVELER_START, tg[intent],
                               ROLES[roles[i]])
            if path is None:
                continue
            song = build_song_cfg(env, path, fp_at, Config())
            agents[i].observe(env, intent, song, fam, ver, t,
                              roles[i], utility_fn)
        if comm and t % CADENCE == CADENCE - 1:
            for i, ag in enumerate(agents):
                for rec in ag.outbox(t - CADENCE):
                    stats["wire"] += (ag.wire_bits(rec)
                                      if hasattr(ag, "wire_bits")
                                      else record_bits_dict(rec))
                    for j, other in enumerate(agents):
                        if j != i:
                            other.receive(rec, i)

    return {"policy": policy, "seed": seed,
            "group_cost": {r: float(np.mean(v))
                           for r, v in stats["cost"].items()},
            "team_cost": float(np.mean(
                [np.mean(v) for v in stats["cost"].values()])),
            "success_first": stats["succ"] / stats["n"],
            "fail_open": stats["phantom"] / stats["n"],
            "duplicate_rate": stats["dup"] / max(1, stats["commits"]),
            "wire_bits": stats["wire"],
            "memory_bits": float(np.mean([a.memory_bits()
                                          for a in agents]))}


def record_bits_dict(rec):
    from songlines.record import bits_of_song
    return bits_of_song(rec["song"]) + 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 102])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/bench")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "bench_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "BU.1": "songline_full <= best direct baseline on team "
                        "cost at equal budget, unseen test seeds 100+",
                "policies": ["independent", "songline_full",
                             "decision_centric", "execution_path",
                             "graph_memory", "learned_formation"],
            }, f, indent=2)
    shard = f"bench_{a.policy}_a{a.agents}_e{a.episodes}_s{a.seeds[0]}-{a.seeds[1]}.jsonl"
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            row = run_cell(a.policy, seed, a.agents, a.episodes)
            f.write(json.dumps(row) + "\n")
            print(f"{a.policy} seed {seed}: team {row['team_cost']:.1f} "
                  f"succ {row['success_first']:.2f} "
                  f"fo {row['fail_open']:.3f} "
                  f"mem {row['memory_bits']:.0f}", flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
