"""S1 — the online social memory process.

Everything before this was a controlled team-memory benchmark: one
shared store, consumers evaluated after the stream. S1 makes the
process SOCIAL and simultaneous: N agents of mixed embodiments, each
with its OWN memory, act concurrently in a world whose families
evolve globally (waters move for everyone); on a cadence they
broadcast certificates; receivers recompute admissibility for their
own role; reservations deconflict same-target commitments.

Per episode: every agent is assigned a family in its CURRENT global
state (repeat / fresh appearance / conflict-moved water), walks it
(witness run, role-aware), forms a candidate song and feeds its own
UCSM controller (marginal utility replayed on its own memory).
Every K episodes: broadcast wave --- each agent ships its schemas AS
CERTIFICATES (song + per-role utility profile + support + failures);
receivers consolidate through the two-axis matrix with the
certificate's own-role utility as testimony.  Collisions: when
several agents committed the same target in the same episode, all but
the reservation holder pay a contention cost.

Arms:
  independent  no communication (the no-comm baseline of the series)
  raw_share    broadcast raw songs, receivers append everything
  cert         UCSM certificates + role gating
  cert_resv    certificates + reservations (first commit holds; later
               agents fall through to their next target)

Registered predictions:
  S1.1 (communication pays, structure pays more): cert group cost <
       independent's on both roles; cert <= raw_share.
  S1.2 (role gating protects the fragile): fragile-role group cost
       under cert <= raw_share's fragile cost - 5%.
  S1.3 (reservations deconflict): duplicate-target rate under
       cert_resv <= 0.5x cert's, at group cost within 3%.
  S1.4 (certificates are cheap): broadcast bits under cert <= 0.35x
       raw_share's.

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_s1_social.py \
        --seeds 0 3 --episodes 300 --agents 6 --out tmp/cluster/song_grammar/s1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.song_grammar.exp_s0_song_smoke import BAND, arm_song, fp_at
from experiments.song_grammar.exp_u7_seven_arms import walk_targets
from experiments.song_grammar.u7_common import (
    ROLES, bits_of_song, dijkstra, family_world, marginal_utility,
    valid_world, witness_song)
from experiments.song_grammar.ucsm import Schema, nearest

GridXY = Tuple[int, int]
U_THR, SHARE_THR, D_THR = 5.0, 0.4, 3
CONTENTION = 25.0          # role-neutral cost of a duplicated commit
CADENCE = 5


# ── per-agent UCSM memory (as in U7's stale arm, per agent) ────────

class AgentMemory:
    def __init__(self, role_name: str):
        self.role_name = role_name
        self.items: List[Dict[str, Any]] = []
        self.now = 0
        self.received: set = set()   # dedup keys of foreign items

    def consider(self, song, role_u: Dict[str, float], t: int,
                 fam: int) -> None:
        self.now = t
        u = role_u[self.role_name]
        idx, ana = nearest(song, [Schema(it["song"], cert=None)
                                  for it in self.items])
        simple = ana is not None and ana["share"] >= SHARE_THR
        conflict = simple and ana["D"] >= D_THR
        if u >= U_THR:
            if conflict:
                self.items.append({"song": song, "t": t,
                                   "role_u": role_u,
                                   "kind": "exception", "family": fam,
                                   "parent": idx, "support": 1})
            elif simple:
                it = self.items[idx]
                it["song"], it["t"] = song, t
                it["support"] = it.get("support", 1) + 1
                it["role_u"] = {r: max(it["role_u"].get(r, 0.0),
                                       role_u.get(r, 0.0))
                                for r in ROLES}
            else:
                self.items.append({"song": song, "t": t,
                                   "role_u": role_u, "kind": "schema",
                                   "family": fam, "support": 1})
        elif simple:
            self.items[idx]["support"] = \
                self.items[idx].get("support", 1) + 1

    def targets(self, band_fps, gate_roles: bool) -> List[GridXY]:
        import math
        flipped = {it.get("parent") for it in self.items
                   if it.get("kind") == "exception"}
        def score(i, it):
            s = it.get("support", 1) * math.exp(
                -0.02 * (self.now - it["t"]))
            if i in flipped:
                s *= 0.25
            return s
        usable = [(i, it) for i, it in enumerate(self.items)
                  if not gate_roles
                  or it["role_u"].get(self.role_name, 0.0) >= 0.0]
        usable.sort(key=lambda p: -score(*p))
        out, seen = [], set()
        for _, it in usable:
            res = arm_song(it["song"], band_fps)
            tt = res["transported"]
            if tt is not None:
                tt = (int(tt[0]), int(tt[1]))
                if tt not in seen:
                    seen.add(tt)
                    out.append(tt)
        return out


# ── the social world: families evolve globally ─────────────────────

class SocialWorld:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.families: List[int] = []
        self.state: Dict[int, Tuple[int, int]] = {}  # fam -> (variant, widx)
        self.fam_counter = seed * 100_000

    def assign(self, t: int) -> Tuple[int, Any, GridXY, str]:
        for _ in range(30):
            if not self.families or self.rng.random() < 0.25:
                self.fam_counter += 1
                fam, kind = self.fam_counter, "new"
                variant, widx = 0, 0
            else:
                fam = int(self.rng.choice(self.families))
                kind = self.rng.choice(
                    ["repeat", "appearance", "conflict"],
                    p=[0.35, 0.40, 0.25])
                variant, widx = self.state[fam]
                if kind == "appearance":
                    variant = int(self.rng.integers(1, 40))
                elif kind == "conflict":
                    widx = (widx + 1) % 3
            env, water = family_world(fam, variant, widx)
            if valid_world(env, water):
                if kind == "new":
                    self.families.append(fam)
                self.state[fam] = (variant, widx)
                return fam, env, water, kind
        raise RuntimeError("no valid assignment")


# ── one seeded social run for one arm ──────────────────────────────

def run_arm(arm: str, seed: int, n_agents: int, n_episodes: int
            ) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 999)
    roles = ["fragile" if i % 2 == 0 else "robust"
             for i in range(n_agents)]
    mems = [AgentMemory(roles[i]) for i in range(n_agents)]
    world = SocialWorld(seed)
    comm_bits = 0
    group_cost = {r: [] for r in ROLES}
    duplicates, commits = 0, 0

    for t in range(n_episodes):
        # each agent visits its assigned world; consumption first
        # (does my memory take me to water HERE?), then formation
        assignments = [world.assign(t) for _ in range(n_agents)]
        episode_targets: Dict[GridXY, int] = {}
        for i, (fam, env, water, kind) in enumerate(assignments):
            band_fps = {xy: fp_at(env, xy) for xy in BAND}
            targets = mems[i].targets(
                band_fps, gate_roles=arm in ("cert", "cert_resv"))
            if arm == "cert_resv" and targets:
                held = [tt for tt in targets
                        if episode_targets.get(tt, i) != i
                        and episode_targets.get(tt) is not None]
                targets = [tt for tt in targets
                           if episode_targets.get(tt) is None
                           or episode_targets.get(tt) == i]
            r = walk_targets(env, targets, ROLES[roles[i]])
            cost = r["cost"]
            if targets:
                commits += 1
                first = targets[0]
                if first in episode_targets \
                        and episode_targets[first] != i:
                    duplicates += 1
                    if arm != "cert_resv":
                        cost += CONTENTION
                else:
                    episode_targets[first] = i
            group_cost[roles[i]].append(cost)
            # formation from this visit (witness = the agent itself)
            song = witness_song(env, water, ROLES[roles[i]])
            if song is None:
                continue
            role_u = {rn: marginal_utility(
                env, [it["song"] for it in mems[i].items], song,
                ROLES[rn]) for rn in ROLES}
            n_before = len(mems[i].items)
            mems[i].consider(song, role_u, t, fam)
            if len(mems[i].items) > n_before:
                mems[i].items[-1]["origin"] = i
                mems[i].items[-1]["uid"] = (i, t)

        # broadcast wave: ONLY own-formed items travel (the series'
        # privacy contract: foreign evidence is never re-broadcast),
        # and receivers deduplicate by item uid
        if arm != "independent" and t % CADENCE == CADENCE - 1:
            for i in range(n_agents):
                payload = [it for it in mems[i].items
                           if it.get("origin", i) == i
                           and it.get("t", -1) > t - CADENCE - 1]
                for it in payload:
                    uid = it.get("uid", (i, it.get("t", t)))
                    comm_bits += bits_of_song(it["song"]) + (
                        0 if arm == "raw_share" else 64)  # cert fields
                    for j in range(n_agents):
                        if j == i or uid in mems[j].received:
                            continue
                        mems[j].received.add(uid)
                        if arm == "raw_share":
                            mems[j].items.append(
                                {**it, "origin": i,
                                 "role_u": {r: 0.0 for r in ROLES}})
                        else:
                            nb = len(mems[j].items)
                            mems[j].consider(
                                it["song"], it["role_u"], t,
                                it.get("family", -1))
                            if len(mems[j].items) > nb:
                                # consolidated foreign knowledge is
                                # not re-broadcast (no gossip relay)
                                mems[j].items[-1]["origin"] = i
                                mems[j].items[-1]["uid"] = uid

    return {"arm": arm, "seed": seed,
            "group_cost": {r: float(np.mean(v))
                           for r, v in group_cost.items()},
            "duplicate_rate": duplicates / max(1, commits),
            "comm_bits": comm_bits,
            "mem_items": float(np.mean([len(m.items) for m in mems]))}


ARMS = ["independent", "raw_share", "cert", "cert_resv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/s1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "s1_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "S1.1": "cert group cost < independent both roles; "
                        "cert <= raw_share",
                "S1.2": "fragile cost: cert <= 0.95 * raw_share",
                "S1.3": "duplicate rate: cert_resv <= 0.5 * cert, "
                        "cost within 3%",
                "S1.4": "comm bits: cert <= 0.35 * raw_share",
                "constants": {"CADENCE": CADENCE,
                              "CONTENTION": CONTENTION},
            }, f, indent=2)
    shard = f"s1_e{a.episodes}_s{a.seeds[0]}-{a.seeds[1]}.jsonl"
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            for arm in ARMS:
                row = run_arm(arm, seed, a.agents, a.episodes)
                f.write(json.dumps(row) + "\n")
                print(f"seed {seed} {arm}: "
                      f"frag {row['group_cost']['fragile']:.0f} "
                      f"rob {row['group_cost']['robust']:.0f} "
                      f"dup {row['duplicate_rate']:.3f} "
                      f"bits {row['comm_bits']}", flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
