"""S2 — the world-clock staleness contract for social formation.

S1's honest negative localised the mechanism precisely: receivers
accumulate foreign schemas whose WORLD referents drift stale faster
than role gating or recency can filter --- broadcast time is not
world time. S2 supplies the missing contract:

  * every family carries a STATE VERSION, bumped whenever its world
    actually changes (the water moves); appearance re-texturing does
    not bump it --- the version tracks the referent, not the look;
  * certificates carry (family, version); receivers maintain
    known_version[f] from their own visits AND from certificate
    metadata (version news travels even when the schema itself is
    rejected --- cheap gossip of world time);
  * admissibility is gated on world clocks at BOTH ends: a schema
    older than the receiver's known version of its family is
    inadmissible for consumption and is not consolidated on receipt.

This is the series' age_max law restated on object time: evidence
ages against the evolution of its referent, not against the mail
stamp.

Arms (identical assignment streams per seed --- paired):
  independent   no communication (S1 baseline)
  cert          S1's certificate arm (message-time only; the control)
  cert_wc       certificates + world-clock gate
  cert_wc_resv  + reservations

Registered predictions:
  S2.1 (the reversal reverses): cert_wc group cost < independent on
       BOTH roles at 300 episodes --- communication becomes
       net-positive at the horizon where S1 showed it net-harmful.
  S2.2 (mechanism, not noise): cert_wc < cert by >= 10% on both roles.
  S2.3 (reservations compose): duplicates == 0 under cert_wc_resv at
       <= 3% cost overhead.
  S2.4 (hygiene): cert_wc mean memory size <= 0.7x cert's --- stale
       foreign schemas are refused, not accumulated.

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_s2_worldclock.py \
        --seeds 0 2 --episodes 300 --agents 6 --out tmp/cluster/song_grammar/s2
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
from experiments.song_grammar.exp_s1_social import (
    CADENCE, CONTENTION, AgentMemory, SocialWorld)
from experiments.song_grammar.exp_u7_seven_arms import walk_targets
from experiments.song_grammar.u7_common import (
    ROLES, bits_of_song, marginal_utility, witness_song)

GridXY = Tuple[int, int]


class SocialWorldWC(SocialWorld):
    """SocialWorld with a per-family state version: bumped when the
    world CHANGES (conflict), not when it is merely re-textured."""

    def __init__(self, seed: int):
        super().__init__(seed)
        self.version: Dict[int, int] = {}

    def assign(self, t: int):
        fam, env, water, kind = super().assign(t)
        if kind == "new":
            self.version[fam] = 0
        elif kind == "conflict":
            self.version[fam] = self.version.get(fam, 0) + 1
        return fam, env, water, kind, self.version.get(fam, 0)


class AgentMemoryWC(AgentMemory):
    def __init__(self, role_name: str, world_clock: bool):
        super().__init__(role_name)
        self.world_clock = world_clock
        self.known_version: Dict[int, int] = {}

    def note_version(self, fam: int, ver: int) -> None:
        if ver > self.known_version.get(fam, -1):
            self.known_version[fam] = ver

    def admissible(self, it: Dict[str, Any]) -> bool:
        if not self.world_clock:
            return True
        fam, ver = it.get("family", -1), it.get("version", 0)
        return ver >= self.known_version.get(fam, -1)

    def consider_wc(self, song, role_u, t: int, fam: int,
                    ver: int) -> None:
        self.note_version(fam, ver)
        if self.world_clock and ver < self.known_version.get(fam, -1):
            return                       # stale on arrival: refused
        self.consider(song, role_u, t, fam)
        # both APPEND and MERGE must refresh the version stamp: a
        # schema updated with current evidence is current (v1 bug:
        # merged schemas kept the old version and got gated off)
        for it in self.items:
            if it.get("t") == t and it.get("version", -1) < ver:
                it["version"] = ver

    def targets(self, band_fps, gate_roles: bool):
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
                  if self.admissible(it)
                  and (not gate_roles
                       or it["role_u"].get(self.role_name, 0.0)
                       >= 0.0)]
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


def run_arm(arm: str, seed: int, n_agents: int, n_episodes: int
            ) -> Dict[str, Any]:
    wc = arm in ("cert_wc", "cert_wc_resv")
    resv = arm == "cert_wc_resv"
    comm = arm != "independent"
    roles = ["fragile" if i % 2 == 0 else "robust"
             for i in range(n_agents)]
    mems = [AgentMemoryWC(roles[i], world_clock=wc)
            for i in range(n_agents)]
    world = SocialWorldWC(seed)
    comm_bits = 0
    group_cost = {r: [] for r in ROLES}
    duplicates, commits = 0, 0

    for t in range(n_episodes):
        assignments = [world.assign(t) for _ in range(n_agents)]
        episode_targets: Dict[GridXY, int] = {}
        for i, (fam, env, water, kind, ver) in enumerate(assignments):
            mems[i].note_version(fam, ver)
            band_fps = {xy: fp_at(env, xy) for xy in BAND}
            targets = mems[i].targets(band_fps, gate_roles=comm)
            if resv and targets:
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
                    if not resv:
                        cost += CONTENTION
                else:
                    episode_targets[first] = i
            group_cost[roles[i]].append(cost)
            song = witness_song(env, water, ROLES[roles[i]])
            if song is None:
                continue
            role_u = {rn: marginal_utility(
                env, [it["song"] for it in mems[i].items
                      if mems[i].admissible(it)], song, ROLES[rn])
                for rn in ROLES}
            nb = len(mems[i].items)
            mems[i].consider_wc(song, role_u, t, fam, ver)
            if len(mems[i].items) > nb:
                mems[i].items[-1]["origin"] = i
                mems[i].items[-1]["uid"] = (i, t)

        if comm and t % CADENCE == CADENCE - 1:
            for i in range(n_agents):
                payload = [it for it in mems[i].items
                           if it.get("origin", i) == i
                           and it.get("t", -1) > t - CADENCE - 1]
                for it in payload:
                    uid = it.get("uid", (i, it.get("t", t)))
                    comm_bits += bits_of_song(it["song"]) + 64 + 16
                    for j in range(n_agents):
                        if j == i or uid in mems[j].received:
                            continue
                        mems[j].received.add(uid)
                        nb = len(mems[j].items)
                        mems[j].consider_wc(
                            it["song"], it["role_u"], t,
                            it.get("family", -1),
                            it.get("version", 0))
                        if len(mems[j].items) > nb:
                            mems[j].items[-1]["origin"] = i
                            mems[j].items[-1]["uid"] = uid

    return {"arm": arm, "seed": seed,
            "group_cost": {r: float(np.mean(v))
                           for r, v in group_cost.items()},
            "duplicate_rate": duplicates / max(1, commits),
            "comm_bits": comm_bits,
            "mem_items": float(np.mean([len(m.items) for m in mems]))}


ARMS = ["independent", "cert", "cert_wc", "cert_wc_resv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/s2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "s2_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "S2.1": "cert_wc < independent, both roles, 300 eps",
                "S2.2": "cert_wc <= 0.90 * cert, both roles",
                "S2.3": "cert_wc_resv duplicates == 0 at <= 3% cost",
                "S2.4": "cert_wc mem items <= 0.7 * cert's",
                "contract": "version = referent evolution (conflict "
                            "bumps; re-texturing does not); "
                            "admissibility gated at consumption AND "
                            "consolidation; version news travels in "
                            "certificate metadata (+16 bits)",
            }, f, indent=2)
    shard = f"s2_e{a.episodes}_s{a.seeds[0]}-{a.seeds[1]}.jsonl"
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            for arm in ARMS:
                row = run_arm(arm, seed, a.agents, a.episodes)
                f.write(json.dumps(row) + "\n")
                print(f"seed {seed} {arm}: "
                      f"frag {row['group_cost']['fragile']:.0f} "
                      f"rob {row['group_cost']['robust']:.0f} "
                      f"dup {row['duplicate_rate']:.3f} "
                      f"items {row['mem_items']:.0f}", flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
