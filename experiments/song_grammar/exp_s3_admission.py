"""S3 — admission control: testimony is not admissible evidence.

S1 and S2 bracketed the social-memory failure from both sides:
neither message-time nor world-time filtering makes exchange
net-positive at horizon, and the residual harm is CROSS-FAMILY
ALIASING FROM FOREIGN VOLUME --- schemas about families the receiver
barely visits still partially match its local constellations and emit
phantoms. S3 attacks admission itself:

  * every foreign certificate goes to QUARANTINE on arrival (version
    news still updates world clocks --- metadata is cheap);
  * when the agent VISITS family f, quarantined certificates about f
    are validated ON THE SPOT: the marginal utility of the foreign
    song is replayed on the actual current world with the agent's own
    role; passers are admitted (and usable in that same visit),
    failers are discarded;
  * knowledge about families the agent never visits never enters
    consumption at all.

This is the receiver-recompute principle of the certificate model
carried to the door: admission by demonstrated utility on one's own
visits, not by the sender's testimony.

Arms (paired assignment streams; all include the S2 world clock):
  independent   no communication
  cert_wc       S2's gated arm (testimony admitted; the control)
  adm_visit     relevance only: admit testimony for families the
                agent has visited; quarantine the rest until first
                visit, then admit on testimony
  adm_util      relevance + demonstrated utility (full admission
                control)
  adm_util_resv + reservations (the composed system)

Registered predictions:
  S3.1 (the reversal finally reverses): adm_util group cost <
       independent on BOTH roles at 300 episodes.
  S3.2 (admission is the mechanism, monotonically): cert_wc >=
       adm_visit >= adm_util on both roles, with adm_util <= 0.90 x
       cert_wc.
  S3.3 (hygiene becomes real): adm_util admitted-memory size <= 0.5 x
       cert_wc's.
  S3.4 (composition): adm_util_resv duplicates == 0 at <= 3% cost.

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_s3_admission.py \
        --seeds 0 2 --episodes 300 --agents 6 --out tmp/cluster/song_grammar/s3
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

from experiments.song_grammar.exp_s0_song_smoke import BAND, fp_at
from experiments.song_grammar.exp_s1_social import CADENCE, CONTENTION
from experiments.song_grammar.exp_s2_worldclock import (
    AgentMemoryWC, SocialWorldWC)
from experiments.song_grammar.exp_u7_seven_arms import walk_targets
from experiments.song_grammar.u7_common import (
    ROLES, bits_of_song, marginal_utility, witness_song)

GridXY = Tuple[int, int]
U_THR = 5.0


class AgentMemoryAC(AgentMemoryWC):
    """World-clock memory + admission control with quarantine."""

    def __init__(self, role_name: str, admission: str,
                 world_clock: bool = True):
        # admission: "none" (testimony), "visit", "util"
        super().__init__(role_name, world_clock=world_clock)
        self.admission = admission
        self.visited: set = set()
        self.quarantine: Dict[int, List[Dict[str, Any]]] = {}

    def receive(self, song, role_u, t: int, fam: int, ver: int,
                origin: int, uid) -> None:
        self.note_version(fam, ver)      # world time always travels
        if self.admission == "none" or (self.admission == "visit"
                                        and fam in self.visited):
            nb = len(self.items)
            self.consider_wc(song, role_u, t, fam, ver)
            if len(self.items) > nb:
                self.items[-1]["origin"] = origin
                self.items[-1]["uid"] = uid
            return
        self.quarantine.setdefault(fam, []).append(
            {"song": song, "role_u": role_u, "t": t, "fam": fam,
             "version": ver, "origin": origin, "uid": uid})

    def on_visit(self, env, fam: int, ver: int, t: int) -> None:
        """Validate quarantined certificates about this family ON THE
        SPOT: replay their marginal utility on the actual world."""
        self.visited.add(fam)
        self.note_version(fam, ver)
        pending = self.quarantine.pop(fam, [])
        for q in pending:
            if self.world_clock and q["version"] < \
                    self.known_version.get(fam, -1):
                continue                          # stale: discard
            if self.admission == "util":
                admitted_songs = [it["song"] for it in self.items
                                  if self.admissible(it)]
                u = marginal_utility(env, admitted_songs, q["song"],
                                     ROLES[self.role_name])
                if u < U_THR:
                    continue                      # useless HERE: out
                role_u = dict(q["role_u"])
                role_u[self.role_name] = u        # measured, not told
            else:
                role_u = q["role_u"]
            nb = len(self.items)
            self.consider_wc(q["song"], role_u, q["t"], fam,
                             q["version"])
            if len(self.items) > nb:
                self.items[-1]["origin"] = q["origin"]
                self.items[-1]["uid"] = q["uid"]

    def quarantine_size(self) -> int:
        return sum(len(v) for v in self.quarantine.values())


def run_arm(arm: str, seed: int, n_agents: int, n_episodes: int
            ) -> Dict[str, Any]:
    comm = arm != "independent"
    resv = arm == "adm_util_resv"
    admission = {"cert_wc": "none", "adm_visit": "visit",
                 "adm_util": "util", "adm_util_resv": "util",
                 "independent": "none"}[arm]
    roles = ["fragile" if i % 2 == 0 else "robust"
             for i in range(n_agents)]
    # the independent baseline must stay as strong as in S1/S2: no
    # world-clock gate on one's own memory (a stale own phantom is
    # cheaper than blind search -- gating own memory only weakens it)
    mems = [AgentMemoryAC(roles[i], admission, world_clock=comm)
            for i in range(n_agents)]
    world = SocialWorldWC(seed)
    comm_bits = 0
    group_cost = {r: [] for r in ROLES}
    duplicates, commits = 0, 0

    for t in range(n_episodes):
        assignments = [world.assign(t) for _ in range(n_agents)]
        episode_targets: Dict[GridXY, int] = {}
        for i, (fam, env, water, kind, ver) in enumerate(assignments):
            mems[i].on_visit(env, fam, ver, t)   # validate & admit
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
                        mems[j].receive(
                            it["song"], it["role_u"], t,
                            it.get("family", -1),
                            it.get("version", 0), i, uid)

    return {"arm": arm, "seed": seed,
            "group_cost": {r: float(np.mean(v))
                           for r, v in group_cost.items()},
            "duplicate_rate": duplicates / max(1, commits),
            "comm_bits": comm_bits,
            "mem_items": float(np.mean([len(m.items) for m in mems])),
            "quarantined": float(np.mean([m.quarantine_size()
                                          for m in mems]))}


ARMS = ["independent", "cert_wc", "adm_visit", "adm_util",
        "adm_util_resv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/s3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "s3_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "S3.1": "adm_util < independent, both roles, 300 eps",
                "S3.2": "monotone cert_wc >= adm_visit >= adm_util "
                        "both roles; adm_util <= 0.90 * cert_wc",
                "S3.3": "adm_util admitted items <= 0.5 * cert_wc's",
                "S3.4": "adm_util_resv duplicates == 0 at <= 3% cost",
                "principle": "admission by demonstrated utility on "
                             "one's own visits; quarantine until "
                             "validated on the spot; version news "
                             "travels regardless",
            }, f, indent=2)
    shard = f"s3_e{a.episodes}_s{a.seeds[0]}-{a.seeds[1]}.jsonl"
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            for arm in ARMS:
                row = run_arm(arm, seed, a.agents, a.episodes)
                f.write(json.dumps(row) + "\n")
                print(f"seed {seed} {arm}: "
                      f"frag {row['group_cost']['fragile']:.0f} "
                      f"rob {row['group_cost']['robust']:.0f} "
                      f"dup {row['duplicate_rate']:.3f} "
                      f"items {row['mem_items']:.0f} "
                      f"quar {row['quarantined']:.0f}", flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
