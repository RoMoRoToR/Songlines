"""CA — clock alternatives: is WORLD time load-bearing in S2's gate?

Reviewer: "world-clock necessary" (S2) must be tested against
logical-clock alternatives --- Lamport, vector, version-only, and
no clock at all.  This driver re-runs the S2 setting (the social
process with globally drifting families, paired assignment streams
per seed via ``SocialWorldWC``) with ONE memory class whose
staleness gate is a pluggable clock policy.  An audit fact stated up
front: S2's "world clock" never consults global time in its gate ---
the gate compares REFERENT VERSIONS (family state versions gossiped
in certificate metadata); global time appears nowhere in
admissibility.  The arms below make that decomposition explicit and
test whether causal/logical order without world state versions
suffices.

Arms (identical assignment streams per seed; all communicating arms
are S1-cert-style: consolidated certificates, role gating, cadence
broadcast, no reservations):
  independent    no communication (paired baseline)
  no_clock       no staleness gate (= S2's "cert" control)
  world          referent versions from the world + version gossip,
                 AND certificates carry global formation time used
                 for foreign-item recency decay (a strict SUPERSET
                 of S2's arm: S2 itself used arrival time)
  version_only   referent versions + gossip, NO time field on the
                 wire; foreign recency decays by local arrival time
                 (this is behaviourally S2's cert_wc verbatim)
  lamport        Lamport counters: tick per visit, max+1 on receive;
                 an item about family f is admissible iff its
                 formation stamp >= the receiver's stamp at its own
                 last visit to f.  No referent-change news exists.
  vector         vector clocks: an item is inadmissible iff its
                 stamp is strictly dominated by the receiver's stamp
                 at its own last visit to f (provably
                 happened-before); concurrent items are admitted.
  age            age-since-last-corroboration: no versions, no
                 logical stamps; every item tracks the local episode
                 of its last corroboration (formation, merge, or
                 support bump; receipt counts as corroboration at
                 arrival); admissible iff now - corrob <= AGE_MAX.
                 AGE_MAX = 30 episodes (6 cadences), frozen a priori,
                 not tuned.

Wire accounting per certificate: song bits + 64 (cert) + per-policy
stamp: world +32 (version 16 + global time 16), version_only +16,
lamport +16, vector +16 x N_agents, age +0, no_clock +0.

Registered predictions (frozen before any run; thresholds refer to
the full 12-seed x 300-episode x 6-agent run; the ~100-episode smoke
is a direction check):
  CA.1 (control reproduces S2): world < no_clock on both roles
       (S2.2 measured ~7-10% on robust).
  CA.2 (world TIME is not the mechanism): version_only within 2% of
       world on both roles --- and version_only must reproduce S2's
       cert_wc numbers on shared seeds/episodes, since it is that
       arm restated.  If confirmed, the S2 claim should be recast:
       "referent versioning is necessary", not "world time".
  CA.3 (causal order is partial): lamport and vector land between
       no_clock and world on both roles --- they can retire items
       superseded by the receiver's OWN visits but cannot carry
       referent-change news about families the receiver has not
       revisited.  If either matches world (<= 2% gap on both
       roles), that is an honest result AGAINST the necessity of
       version gossip and is reported as such.
  CA.4 (age law is partial): age improves over no_clock but does not
       reach world --- bounded staleness discards fresh-but-old
       items and keeps stale-but-recent ones.
  CA.5 (accounting): vector pays O(N) stamp bits; the bits column is
       reported for the necessity-vs-price discussion.

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_clock_alternatives.py \
        --seeds 0 3 --episodes 100 --agents 6 \
        --out tmp/clock_alternatives_smoke
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

from experiments.song_grammar.exp_s0_song_smoke import (
    BAND, arm_song, fp_at)
from experiments.song_grammar.exp_s1_social import (
    CADENCE, CONTENTION, D_THR, SHARE_THR, U_THR, AgentMemory)
from experiments.song_grammar.exp_s2_worldclock import SocialWorldWC
from experiments.song_grammar.exp_u7_seven_arms import walk_targets
from experiments.song_grammar.u7_common import (
    ROLES, bits_of_song, marginal_utility, witness_song)
from experiments.song_grammar.ucsm import Schema, nearest

GridXY = Tuple[int, int]
AGE_MAX = 30            # frozen a priori: 6 cadences, not tuned
POLICIES = ("none", "world", "version_only", "lamport", "vector",
            "age")
STAMP_BITS = {"no_clock": 0, "world": 32, "version_only": 16,
              "lamport": 16, "age": 0}


class ClockMemory(AgentMemory):
    """S1's per-agent memory with a pluggable staleness clock.

    ``consider_stamped`` is S1's ``AgentMemory.consider`` verbatim
    plus stamp bookkeeping on the touched item (S2's post-hoc
    version-refresh pattern, extended to logical stamps and
    corroboration --- the support-bump path corroborates too, which
    the post-hoc t == t loop could not see).
    """

    def __init__(self, role_name: str, policy: str, aid: int,
                 n_agents: int):
        super().__init__(role_name)
        assert policy in POLICIES
        self.policy = policy
        self.aid = aid
        self.n_agents = n_agents
        self.known_version: Dict[int, int] = {}
        self.lamport = 0
        self.vc = [0] * n_agents
        self.last_visit_l: Dict[int, int] = {}
        self.last_visit_vc: Dict[int, List[int]] = {}
        self.rejected_arrival = 0

    # ── clock bookkeeping ──────────────────────────────────────────
    def tick_visit(self, fam: int, ver: int, t: int) -> None:
        self.now = t
        self.lamport += 1
        self.vc[self.aid] += 1
        self.last_visit_l[fam] = self.lamport
        self.last_visit_vc[fam] = list(self.vc)
        if ver > self.known_version.get(fam, -1):
            self.known_version[fam] = ver

    def tick_receive(self, stamp: Dict[str, Any]) -> None:
        if self.policy == "lamport":
            self.lamport = max(self.lamport,
                               stamp.get("L", 0)) + 1
        elif self.policy == "vector":
            self.vc = [max(a, b) for a, b in
                       zip(self.vc, stamp.get("vc",
                                              [0] * self.n_agents))]
            self.vc[self.aid] += 1

    def note_version(self, fam: int, ver: int) -> None:
        # version news travels only under the version policies
        if self.policy in ("world", "version_only") \
                and ver > self.known_version.get(fam, -1):
            self.known_version[fam] = ver

    # ── the pluggable staleness gate ───────────────────────────────
    def admissible(self, it: Dict[str, Any]) -> bool:
        p = self.policy
        fam = it.get("family", -1)
        if p == "none":
            return True
        if p in ("world", "version_only"):
            return it.get("version", 0) >= \
                self.known_version.get(fam, -1)
        if p == "lamport":
            return it.get("L", 0) >= self.last_visit_l.get(fam, -1)
        if p == "vector":
            ref = self.last_visit_vc.get(fam)
            v = it.get("vc")
            if ref is None or v is None:
                return True
            dominated = (all(a <= b for a, b in zip(v, ref))
                         and any(a < b for a, b in zip(v, ref)))
            return not dominated
        if p == "age":
            return (self.now - it.get("corrob", it.get("t", 0))) \
                <= AGE_MAX
        return True

    # ── formation/consolidation with stamps ────────────────────────
    def consider_stamped(self, song, role_u: Dict[str, float],
                         t: int, fam: int,
                         stamp: Dict[str, Any]) -> Optional[int]:
        """S1's two-axis consider + stamp bookkeeping.  Returns the
        touched index (None on DROP).

        Version stamping under the version policies replicates S2's
        ``consider_wc`` VERBATIM, including its post-hoc refresh of
        every item with ``t == t`` (audit note: that loop bleeds
        version stamps across items formed at the same episode ---
        e.g. across families during one broadcast wave; it is kept
        here so that ``version_only`` reproduces S2's cert_wc
        tick-for-tick, and the quirk is reported, not silently
        fixed).  Logical stamps (L, vc) and corroboration are
        per-item --- they have no S2 precedent to replicate."""
        self.now = max(self.now, t)
        u = role_u[self.role_name]
        idx, ana = nearest(song, [Schema(it["song"], cert=None)
                                  for it in self.items])
        simple = ana is not None and ana["share"] >= SHARE_THR
        conflict = simple and ana["D"] >= D_THR
        touched: Optional[int] = None
        if u >= U_THR:
            if conflict:
                self.items.append({"song": song, "t": t,
                                   "role_u": role_u,
                                   "kind": "exception",
                                   "family": fam, "parent": idx,
                                   "support": 1})
                touched = len(self.items) - 1
            elif simple:
                it = self.items[idx]
                it["song"], it["t"] = song, t
                it["support"] = it.get("support", 1) + 1
                it["role_u"] = {r: max(it["role_u"].get(r, 0.0),
                                       role_u.get(r, 0.0))
                                for r in ROLES}
                touched = idx
            else:
                self.items.append({"song": song, "t": t,
                                   "role_u": role_u,
                                   "kind": "schema", "family": fam,
                                   "support": 1})
                touched = len(self.items) - 1
        elif simple:
            self.items[idx]["support"] = \
                self.items[idx].get("support", 1) + 1
            # a repeat IS corroboration, but not new content
            self.items[idx]["corrob"] = stamp.get("corrob", t)
            touched = None if self.policy in ("world",
                                              "version_only") \
                else idx
        if touched is not None:
            it = self.items[touched]
            for k in ("L", "vc", "corrob"):
                if k in stamp:
                    it[k] = stamp[k]
            if self.policy not in ("world", "version_only") \
                    and "version" in stamp:
                old = it.get("version")
                it["version"] = max(old if old is not None else -1,
                                    stamp["version"])
        if self.policy in ("world", "version_only") \
                and "version" in stamp:
            ver = stamp["version"]
            for it in self.items:           # S2 verbatim (see above)
                if it.get("t") == t and it.get("version", -1) < ver:
                    it["version"] = ver
        return touched

    def receive_cert(self, song, role_u, t_arrival: int, fam: int,
                     stamp: Dict[str, Any], origin: int,
                     uid) -> None:
        if uid in self.received:
            return
        self.received.add(uid)
        self.tick_receive(stamp)
        self.note_version(fam, stamp.get("version", 0))
        probe = {"family": fam, "t": t_arrival,
                 "corrob": t_arrival, **{k: stamp[k] for k in
                                         ("version", "L", "vc")
                                         if k in stamp}}
        if not self.admissible(probe):
            self.rejected_arrival += 1
            return                          # stale on arrival
        # world: global formation time on the item; all others:
        # local arrival time (no time travels on the wire)
        t_store = (stamp.get("t_global", t_arrival)
                   if self.policy == "world" else t_arrival)
        local = dict(stamp)
        local["corrob"] = t_arrival
        nb = len(self.items)
        touched = self.consider_stamped(song, role_u, t_store, fam,
                                        local)
        if len(self.items) > nb and touched is not None:
            self.items[touched]["origin"] = origin
            self.items[touched]["uid"] = uid

    # ── consumption (S2's targets with the pluggable gate) ────────
    def targets(self, band_fps, gate_roles: bool):
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


ARMS = ["independent", "no_clock", "world", "version_only",
        "lamport", "vector", "age"]
ARM_POLICY = {"independent": "none", "no_clock": "none",
              "world": "world", "version_only": "version_only",
              "lamport": "lamport", "vector": "vector",
              "age": "age"}


def run_arm(arm: str, seed: int, n_agents: int, n_episodes: int
            ) -> Dict[str, Any]:
    comm = arm != "independent"
    policy = ARM_POLICY[arm]
    roles = ["fragile" if i % 2 == 0 else "robust"
             for i in range(n_agents)]
    mems = [ClockMemory(roles[i], policy, i, n_agents)
            for i in range(n_agents)]
    world = SocialWorldWC(seed)
    comm_bits = 0
    group_cost = {r: [] for r in ROLES}
    duplicates, commits = 0, 0
    stamp_bits = (STAMP_BITS[arm] if arm != "vector"
                  else 16 * n_agents) if comm else 0

    for t in range(n_episodes):
        assignments = [world.assign(t) for _ in range(n_agents)]
        episode_targets: Dict[GridXY, int] = {}
        for i, (fam, env, water, kind, ver) in enumerate(assignments):
            mems[i].tick_visit(fam, ver, t)
            band_fps = {xy: fp_at(env, xy) for xy in BAND}
            targets = mems[i].targets(band_fps, gate_roles=comm)
            r = walk_targets(env, targets, ROLES[roles[i]])
            cost = r["cost"]
            if targets:
                commits += 1
                first = targets[0]
                if first in episode_targets \
                        and episode_targets[first] != i:
                    duplicates += 1
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
            stamp = {"version": ver, "L": mems[i].lamport,
                     "vc": list(mems[i].vc), "corrob": t,
                     "t_global": t}
            # S2 gates admissibility at consolidation too, own
            # evidence included (consider_wc's stale check).  Audit
            # note: with the version-bleed quirk gossiped into
            # known_version, this can refuse the agent's own FRESH
            # witness of a family whose known_version was inflated
            # by a co-timestamped foreign item.  Replicated verbatim
            # for the version policies; a no-op for lamport/vector/
            # age (a just-ticked visit stamp is always admissible).
            probe = {"family": fam, "t": t, **stamp,
                     "corrob": t}
            if not mems[i].admissible(probe):
                continue
            nb = len(mems[i].items)
            touched = mems[i].consider_stamped(song, role_u, t, fam,
                                               stamp)
            if len(mems[i].items) > nb and touched is not None:
                mems[i].items[touched]["origin"] = i
                mems[i].items[touched]["uid"] = (i, t)
                mems[i].items[touched]["t_global"] = t

        if comm and t % CADENCE == CADENCE - 1:
            for i in range(n_agents):
                payload = [it for it in mems[i].items
                           if it.get("origin", i) == i
                           and it.get("t", -1) > t - CADENCE - 1]
                for it in payload:
                    uid = it.get("uid", (i, it.get("t", t)))
                    comm_bits += (bits_of_song(it["song"]) + 64
                                  + stamp_bits)
                    stamp = {k: it[k] for k in
                             ("version", "L", "vc", "t_global")
                             if k in it}
                    for j in range(n_agents):
                        if j == i:
                            continue
                        mems[j].receive_cert(
                            it["song"], it["role_u"], t,
                            it.get("family", -1), stamp, i, uid)

    return {"arm": arm, "seed": seed, "agents": n_agents,
            "episodes": n_episodes,
            "group_cost": {r: float(np.mean(v))
                           for r, v in group_cost.items()},
            "duplicate_rate": duplicates / max(1, commits),
            "comm_bits": comm_bits,
            "mem_items": float(np.mean([len(m.items)
                                        for m in mems])),
            "admissible_items": float(np.mean(
                [sum(1 for it in m.items if m.admissible(it))
                 for m in mems])),
            "rejected_arrival": float(np.mean(
                [m.rejected_arrival for m in mems]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 3])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--arms", type=str, nargs="*", default=None)
    ap.add_argument("--out", type=str,
                    default="tmp/clock_alternatives_smoke")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "ca_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "CA.1": "world < no_clock, both roles",
                "CA.2": "version_only within 2% of world, both "
                        "roles; version_only == S2 cert_wc restated "
                        "-- claim recast to 'referent versioning', "
                        "not 'world time'",
                "CA.3": "lamport, vector between no_clock and world; "
                        "if either within 2% of world -- honest "
                        "result against version-gossip necessity, "
                        "reported as is",
                "CA.4": "age > no_clock improvement-wise, < world",
                "CA.5": "vector pays 16 x N stamp bits",
                "constants": {"AGE_MAX": AGE_MAX,
                              "CADENCE": CADENCE,
                              "CONTENTION": CONTENTION},
                "protocol": "paired assignment streams "
                            "(SocialWorldWC); S1-cert exchange, no "
                            "reservations; audit fact: S2's gate "
                            "never consults global time",
            }, f, indent=2)
    arms = a.arms or ARMS
    shard = (f"ca_e{a.episodes}_a{a.agents}"
             f"_s{a.seeds[0]}-{a.seeds[1]}.jsonl")
    if a.arms:
        shard = shard.replace(".jsonl", f"_{'_'.join(a.arms)}.jsonl")
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            for arm in arms:
                row = run_arm(arm, seed, a.agents, a.episodes)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"seed {seed} {arm}: "
                      f"frag {row['group_cost']['fragile']:.0f} "
                      f"rob {row['group_cost']['robust']:.0f} "
                      f"dup {row['duplicate_rate']:.3f} "
                      f"items {row['mem_items']:.0f} "
                      f"adm {row['admissible_items']:.0f}",
                      flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
