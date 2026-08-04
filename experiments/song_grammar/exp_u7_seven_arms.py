"""U7 — seven memory policies, one episode stream, heterogeneous roles.

Two cooperating consumers with different bodies (fragile: hazards cost
12 steps; robust: hazards cost 1, steps cost 2) receive witness songs
from a mixed stream (new families / exact repeats / appearance variants
/ water-moved conflicts) and are then evaluated on the families' true
worlds PLUS fresh appearance variants never seen in the stream.

Arms (what the shared memory does with each incoming song):
  raw        append everything, consume newest-first (long context)
  vector     bag-of-signatures cosine dedup, consume by similarity
  rir        recency + importance + relevance scoring, consume top-k
  snapshot   full fingerprint snapshots (CSM-style), align & transport
  utility    keep iff marginal counterfactual utility high (any role);
             no structure, newest-first
  analogy    structural LCS dedup with last-write-wins merge; no utility
  ucsm       utility x analogy matrix + exceptions + role-profiled
             utility certificates (consumer skips songs certified
             harmful for its role)

Registered predictions (written before runs; smoke-scale constructions
were debugged at n_episodes=10 before registration, revisions logged):
  U7.1 (dominance): UCSM mean battery cost <= every other arm for BOTH
       roles at every stream length run.
  U7.2 (sublinear memory): UCSM stored bits grow sublinearly with
       episodes (ratio_1000/100 < 4) while raw grows ~linearly (> 7).
  U7.3 (role preservation): on the fragile role UCSM beats every
       role-blind arm by >= 10% mean cost (certificate gating).
  U7.4 (false merges): analogy-only phantoms first on conflict-family
       base worlds more often than UCSM (last-write-wins corruption).

Usage (seed-sharded for SLURM arrays)::

    PYTHONPATH=. python experiments/song_grammar/exp_u7_seven_arms.py \
        --seeds 0 12 --episodes 100 --out tmp/song_grammar/u7
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

from experiments.song_grammar.exp_s0_song_smoke import (
    BAND, TRAVELER_START, arm_song, fp_at)
from experiments.song_grammar.u7_common import (
    BEAT_BITS, KEY_BITS, LEN_BITS, ROLES, Episode, bits_of_snapshot,
    bits_of_song, blind_cost, dijkstra, eval_battery, make_stream,
    marginal_utility)
from experiments.song_grammar.ucsm import analogy, nearest, Schema
from experiments.warp.semantic_identity import align_frames
from multiagent_env import WATER

GridXY = Tuple[int, int]

U_THR, SHARE_THR, D_THR = 5.0, 0.4, 3
RIR_TOPK, VEC_TOPK = 3, 5


# ── generic consumer: walk an ordered target list ──────────────────

def walk_targets(env, targets: List[GridXY], role: Dict[str, float]
                 ) -> Dict[str, Any]:
    pos, total = TRAVELER_START, 0.0
    phantom_first: Optional[bool] = None
    for t in targets:
        path, cost = dijkstra(env, pos, t, role)
        if path is None:
            continue
        total += cost
        pos = t
        hit = env.cell(*t) == WATER
        if phantom_first is None:
            phantom_first = not hit
        if hit:
            return {"cost": total, "phantom_first": phantom_first}
    return {"cost": total + blind_cost(env, pos, role),
            "phantom_first": bool(phantom_first)}


def song_target(song, band_fps) -> Optional[GridXY]:
    res = arm_song(song, band_fps)
    t = res["transported"]
    return (int(t[0]), int(t[1])) if t is not None else None


def sig_bag(song) -> set:
    return set().union(*[set(c["sig"].keys()) for c in song]) \
        if song else set()


def bag_cos(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


# ── arms ───────────────────────────────────────────────────────────

class ArmBase:
    name = "base"

    def __init__(self):
        self.items: List[Dict[str, Any]] = []   # {song, meta}

    def observe(self, ep: Episode, t: int) -> None:
        raise NotImplementedError

    def targets(self, env, band_fps, role_name: str) -> List[GridXY]:
        raise NotImplementedError

    def bits(self) -> int:
        return sum(bits_of_song(it["song"]) for it in self.items)

    def _targets_of(self, songs, band_fps) -> List[GridXY]:
        out, seen = [], set()
        for s in songs:
            t = song_target(s, band_fps)
            if t is not None and t not in seen:
                seen.add(t)
                out.append(t)
        return out


class RawArm(ArmBase):
    name = "raw"

    def observe(self, ep, t):
        self.items.append({"song": ep.song, "t": t})

    def targets(self, env, band_fps, role_name):
        return self._targets_of(
            [it["song"] for it in reversed(self.items)], band_fps)


class VectorArm(ArmBase):
    name = "vector"

    def observe(self, ep, t):
        bag = sig_bag(ep.song)
        best, best_c = None, 0.0
        for it in self.items:
            c = bag_cos(bag, it["bag"])
            if c > best_c:
                best, best_c = it, c
        if best is not None and best_c >= 0.9:
            best["song"], best["bag"] = ep.song, bag   # dedup-replace
        else:
            self.items.append({"song": ep.song, "bag": bag, "t": t})

    def targets(self, env, band_fps, role_name):
        band_bag = set().union(*[set(v.keys())
                                 for v in band_fps.values()]) \
            if band_fps else set()
        ranked = sorted(self.items,
                        key=lambda it: -bag_cos(band_bag, it["bag"]))
        return self._targets_of([it["song"] for it in ranked[:VEC_TOPK]],
                                band_fps)


class RirArm(ArmBase):
    name = "rir"

    def __init__(self):
        super().__init__()
        self.clock = 0

    def observe(self, ep, t):
        self.clock = t
        imp = max(0.0, min(1.0, ep.meta_importance / 50.0)) \
            if hasattr(ep, "meta_importance") else 0.5
        self.items.append({"song": ep.song, "t": t, "imp": imp,
                           "bag": sig_bag(ep.song)})

    def targets(self, env, band_fps, role_name):
        band_bag = set().union(*[set(v.keys())
                                 for v in band_fps.values()]) \
            if band_fps else set()
        def score(it):
            rec = 0.99 ** (self.clock - it["t"])
            rel = bag_cos(band_bag, it["bag"])
            return 0.4 * rec + 0.3 * it["imp"] + 0.3 * rel
        ranked = sorted(self.items, key=lambda it: -score(it))
        return self._targets_of([it["song"] for it in ranked[:RIR_TOPK]],
                                band_fps)


class SnapshotArm(ArmBase):
    name = "snapshot"

    def observe(self, ep, t):
        fps = {(x, y): fp_at(ep.env, (x, y))
               for y in range(0, 12) for x in range(0, 14)
               if ep.env.cell(x, y) != 1}
        self.items.append({"fps": fps, "water": ep.water, "t": t})

    def targets(self, env, band_fps, role_name):
        out, seen = [], set()
        for it in reversed(self.items[-3:]):
            res = align_frames(band_fps, it["fps"])
            if res.offset is None:
                continue
            dx, dy = res.offset
            t = (it["water"][0] + dx, it["water"][1] + dy)
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def bits(self):
        return sum(bits_of_snapshot(it["fps"]) for it in self.items)


class UtilityArm(ArmBase):
    name = "utility"

    def observe(self, ep, t):
        songs = [it["song"] for it in self.items]
        u = max(marginal_utility(ep.env, songs, ep.song, ROLES[r])
                for r in ROLES)
        if u >= U_THR:
            self.items.append({"song": ep.song, "t": t})

    def targets(self, env, band_fps, role_name):
        return self._targets_of(
            [it["song"] for it in reversed(self.items)], band_fps)


class AnalogyArm(ArmBase):
    name = "analogy"

    def observe(self, ep, t):
        schemas = [Schema(it["song"], cert=None) for it in self.items]
        idx, ana = nearest(ep.song, schemas)
        if ana is not None and ana["share"] >= SHARE_THR:
            self.items[idx]["song"] = ep.song      # last-write-wins
        else:
            self.items.append({"song": ep.song, "t": t})

    def targets(self, env, band_fps, role_name):
        return self._targets_of([it["song"] for it in self.items],
                                band_fps)


class UcsmArm(ArmBase):
    name = "ucsm"

    def observe(self, ep, t):
        role_u = {r: marginal_utility(
            ep.env, [it["song"] for it in self.items], ep.song, ROLES[r])
            for r in ROLES}
        u = max(role_u.values())
        schemas = [Schema(it["song"], cert=None) for it in self.items]
        idx, ana = nearest(ep.song, schemas)
        simple = ana is not None and ana["share"] >= SHARE_THR
        conflict = simple and ana["D"] >= D_THR
        if u >= U_THR:
            if conflict:
                self.items.append({"song": ep.song, "t": t,
                                   "role_u": role_u, "kind": "exception",
                                   "family": ep.family, "parent": idx})
            elif simple:
                it = self.items[idx]
                it["song"], it["t"] = ep.song, t
                it["role_u"] = {r: max(it["role_u"][r], role_u[r])
                                for r in ROLES}
                it["support"] = it.get("support", 1) + 1
            else:
                self.items.append({"song": ep.song, "t": t,
                                   "role_u": role_u, "kind": "schema",
                                   "family": ep.family,
                                   "support": 1})
        elif simple:
            self.items[idx]["support"] = \
                self.items[idx].get("support", 1) + 1

    def targets(self, env, band_fps, role_name):
        # certificate gating: skip only what is certified HARMFUL for
        # this role; unknown-for-role (0) stays accessible
        usable = [it for it in self.items
                  if it["role_u"].get(role_name, 0.0) >= 0.0]
        return self._targets_of([it["song"] for it in usable], band_fps)


class UcsmRecentArm(UcsmArm):
    """U7b: same consolidation as UCSM, but consumption is ordered
    newest-first --- a staleness heuristic borrowed from raw's only
    virtue.  Registered U7b.1: this combination matches or beats raw
    on BOTH roles at 100 and 1000 episodes while keeping UCSM's bit
    economy (the phantom cascade at scale is a STALENESS problem, not
    a consolidation problem --- the age_max law's territory)."""
    name = "ucsm_recent"

    def targets(self, env, band_fps, role_name):
        usable = [it for it in self.items
                  if it["role_u"].get(role_name, 0.0) >= 0.0]
        usable = sorted(usable, key=lambda it: -it["t"])
        return self._targets_of([it["song"] for it in usable], band_fps)


class UcsmStaleArm(UcsmArm):
    """U7c: the series' trust x staleness contract wired into
    consumption.  Two mechanisms from the proven laws, applied at the
    schema level: (i) age decay --- consumption ordered by support *
    exp(-ALPHA * age); (ii) the discrete trust-flip --- a schema that
    has accumulated an EXCEPTION is demoted (its evidence conflicted),
    so the counterexample is tried before the discredited parent.
    Registered U7c.1: matches or beats raw long context on BOTH roles
    at 1000 episodes (where every ordering-only policy failed)."""
    name = "ucsm_stale"
    ALPHA = 0.02
    FLIP = 0.25

    def __init__(self):
        super().__init__()
        self.now = 0

    def observe(self, ep, t):
        self.now = t
        super().observe(ep, t)

    def targets(self, env, band_fps, role_name):
        import math
        flipped = {it.get("parent") for it in self.items
                   if it.get("kind") == "exception"}
        def score(i, it):
            s = it.get("support", 1) * math.exp(
                -self.ALPHA * (self.now - it["t"]))
            if i in flipped:
                s *= self.FLIP
            return s
        usable = [(i, it) for i, it in enumerate(self.items)
                  if it["role_u"].get(role_name, 0.0) >= 0.0]
        usable.sort(key=lambda p: -score(*p))
        return self._targets_of([it["song"] for _, it in usable],
                                band_fps)


ARMS = [RawArm, VectorArm, RirArm, SnapshotArm, UtilityArm, AnalogyArm,
        UcsmArm, UcsmRecentArm, UcsmStaleArm]


# ── X1: cross-family codebook (first-order abstraction) ────────────

def bits_codebook(all_songs) -> int:
    """Shared signature codebook across stored songs: recurring
    constellations (motifs that repeat across families) are stored
    once and referenced by index --- the first-order 'songs ->
    cross-family schemas' step, lossless by construction.
    Registered X1.1: the codebook bits growth ratio (1000/100
    episodes) is smaller than the plain codec's."""
    import math as _m
    from collections import Counter
    sig_counts = Counter(frozenset(c["sig"].keys())
                         for song in all_songs for c in song)
    entries = {s for s, n in sig_counts.items() if n >= 2}
    dict_cost = sum(LEN_BITS + KEY_BITS * len(s) for s in entries)
    idx_bits = max(1, _m.ceil(_m.log2(max(2, len(entries)))))
    total = dict_cost
    for song in all_songs:
        for c in song:
            key = frozenset(c["sig"].keys())
            total += (idx_bits if key in entries
                      else LEN_BITS + KEY_BITS * len(key)) + BEAT_BITS
    return total


# ── one seeded run ─────────────────────────────────────────────────

def run_seed(seed: int, n_episodes: int) -> Dict[str, Any]:
    stream = make_stream(seed, n_episodes)
    arms = [cls() for cls in ARMS]
    for t, ep in enumerate(stream):
        for arm in arms:
            arm.observe(ep, t)
    battery = eval_battery(stream, seed)
    conflict_fams = {ep.family for ep in stream if ep.kind == "conflict"}

    out: Dict[str, Any] = {"seed": seed, "episodes": len(stream),
                           "battery": len(battery), "arms": {}}
    for arm in arms:
        per_role: Dict[str, Any] = {}
        for role_name, role in ROLES.items():
            costs, phantoms, cphantoms = [], 0, 0
            for ep in battery:
                band_fps = {xy: fp_at(ep.env, xy) for xy in BAND}
                targets = arm.targets(ep.env, band_fps, role_name)
                r = walk_targets(ep.env, targets, role)
                costs.append(r["cost"])
                phantoms += int(r["phantom_first"])
                if ep.family in conflict_fams:
                    cphantoms += int(r["phantom_first"])
            per_role[role_name] = {
                "mean_cost": float(np.mean(costs)),
                "phantom_rate": phantoms / len(battery),
                "conflict_phantoms": cphantoms}
        entry = {"roles": per_role, "n_items": len(arm.items),
                 "bits": arm.bits()}
        if arm.name.startswith("ucsm"):
            entry["bits_codebook"] = bits_codebook(
                [it["song"] for it in arm.items])
        out["arms"][arm.name] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 4])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/u7")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "u7_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "constants": {"U_THR": U_THR, "SHARE_THR": SHARE_THR,
                              "D_THR": D_THR, "ROLES": ROLES},
                "U7.1": "UCSM mean battery cost <= every arm, both "
                        "roles, every stream length",
                "U7.2": "UCSM bits sublinear (1000/100 < 4), raw > 7",
                "U7.3": "fragile role: UCSM >= 10% cheaper than every "
                        "role-blind arm",
                "U7.4": "analogy-only conflict phantoms > UCSM's",
            }, f, indent=2)

    shard = f"u7_e{a.episodes}_s{a.seeds[0]}-{a.seeds[1]}.jsonl"
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            row = run_seed(seed, a.episodes)
            f.write(json.dumps(row) + "\n")
            arms = row["arms"]
            line = " ".join(
                f"{k}:{v['roles']['fragile']['mean_cost']:.0f}/"
                f"{v['roles']['robust']['mean_cost']:.0f}"
                for k, v in arms.items())
            print(f"seed {seed} ({row['episodes']} eps, "
                  f"{row['battery']} eval): {line}", flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
