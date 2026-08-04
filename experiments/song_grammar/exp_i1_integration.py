"""I1 — the integration benchmark: one runtime, every arm and
ablation as a configuration, standard metrics, full wire cost.

Substrate (reviewer items 2, 4, 12): N agents of mixed embodiments,
private frames (consumption is frame-free landmark matching), TWO
intents per collective (resource search: water; rest place: goal),
globally drifting families (water moves; rest is static), optional
NOISY perception (per-observation false negatives and spurious
features; matching margin loosened accordingly), horizons up to 300
episodes. Test seeds (100+) were never used to tune any threshold;
all constants are frozen from the earlier waves.

Main arms (reviewer item 2):
  independent, raw_history, vector_sim, graph_no_prov, song_plain,
  song_trust, song_wclock, songline_full
Ablations of the full method (one flag each):
  no_landmarks, no_beats, no_provenance, no_worldclock, no_admission,
  no_exceptions, no_reservations, no_immutable, no_utility_gate

Registered hypotheses (frozen before the runs):
  H1 (composition): songline_full is <= every main arm on mean group
     cost for both roles, and every ablation is worse than full on at
     least one of {group cost, fail-open rate, memory bits,
     duplicate rate}.
  H2 (noise): at 10% FN + 10% FP the full method's fail-open rate
     stays < 0.05 and its advantage over independent stays positive
     on both roles.
  H3 (scale): the full-vs-independent advantage is positive at N=4,
     6 and 8.

Metrics per cell: per-role mean cost, first-lock success by intent,
phantom-first (fail-open) rate, refusal rate, hazard contacts,
duplicate commitments, wire bits (full codec incl. certificates,
provenance, timestamps, reservations), memory bits (incl. quarantine
and immutable layer), matching ops (latency proxy).

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_i1_integration.py \
        --config songline_full --seeds 100 112 --episodes 300 \
        --agents 6 --noise 0.0 --out tmp/cluster/song_grammar/i1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.song_grammar.exp_s0_song_smoke import BAND, TRAVELER_START
from experiments.song_grammar.runtime import (
    CERT_BITS, PROV_BITS, RESV_BITS, TIME_BITS, Config, Record,
    SonglineAgent, record_bits, song_target)
from experiments.song_grammar.u7_common import (
    H, W, ROLES, dijkstra, enter_cost, family_world, valid_world)
from experiments.warp.exp_warp_landmark_ablation import cells_around
from experiments.warp.semantic_identity import fingerprint
from multiagent_env import HAZARD, WALL, WATER
from multiagent_env.grid_world import GOAL

GridXY = Tuple[int, int]
CADENCE, CONTENTION = 5, 25.0
UTILITY_OVERRIDE = None   # UE1 hook: maker(fpf) -> utility_fn
INTENTS = {"water": WATER, "rest": GOAL}
INTENT_TAGS = {"water": "water_source", "rest": "goal"}


# ── worlds with two intents ────────────────────────────────────────

def family_world2(fam: int, variant: int, widx: int):
    env, water = family_world(fam, variant, widx)
    rng = np.random.default_rng(fam * 31 + 7)
    for _ in range(40):
        c = (int(rng.integers(1, 5)), int(rng.integers(2, 10)))
        if env.cell(*c) == 0 and c != TRAVELER_START:
            env.set_cell(*c, GOAL)
            return env, {"water": water, "rest": c}
    return None, None


class World:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.families: List[int] = []
        self.state: Dict[int, Tuple[int, int]] = {}
        self.version: Dict[int, int] = {}
        self.fam_counter = seed * 100_000

    def assign(self):
        for _ in range(40):
            if not self.families or self.rng.random() < 0.25:
                self.fam_counter += 1
                fam, kind, (variant, widx) = self.fam_counter, "new", (0, 0)
            else:
                fam = int(self.rng.choice(self.families))
                kind = self.rng.choice(["repeat", "appearance",
                                        "conflict"], p=[0.35, 0.4, 0.25])
                variant, widx = self.state[fam]
                if kind == "appearance":
                    variant = int(self.rng.integers(1, 40))
                elif kind == "conflict":
                    widx = (widx + 1) % 3
            env, tg = family_world2(fam, variant, widx)
            if env is None or not valid_world(env, tg["water"]):
                continue
            if kind == "new":
                self.families.append(fam)
                self.version[fam] = 0
            elif kind == "conflict":
                self.version[fam] += 1
            self.state[fam] = (variant, widx)
            return fam, env, tg, self.version[fam]
        raise RuntimeError("no valid assignment")


# ── noisy perception front end (reviewer item 4) ───────────────────

ALL_TAGS = ["wall", "hazard_edge", "water_source", "goal", "void"]


def make_fp(noise: float, rng: np.random.Generator):
    def fpf(env, xy: GridXY) -> Dict[str, float]:
        sig = fingerprint(xy, cells_around(env, *xy))
        if noise <= 0:
            return sig
        out = {k: v for k, v in sig.items()
               if rng.random() >= noise}            # false negatives
        for _ in range(int(rng.binomial(4, noise))):  # spurious
            tag = ALL_TAGS[int(rng.integers(len(ALL_TAGS)))]
            dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            out[f"{tag}@{dx},{dy}"] = 1.0
        return out
    return fpf


# ── role-aware execution with hazard accounting ────────────────────

def blind_tag(env, start: GridXY, role, cell_kind) -> Tuple[float, int]:
    seen, q, cost, hits = {start}, deque([start]), 0.0, 0
    while q:
        cur = q.popleft()
        cost += enter_cost(env, cur, role)
        if env.cell(*cur) == HAZARD:
            hits += 1
        if env.cell(*cur) == cell_kind:
            return cost, hits
        x, y = cur
        for nxt in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if (0 <= nxt[0] < W and 0 <= nxt[1] < H
                    and nxt not in seen and env.cell(*nxt) != WALL):
                seen.add(nxt)
                q.append(nxt)
    return cost + role["step"] * W * H, hits


def walk(env, targets: List[GridXY], role, cell_kind
         ) -> Dict[str, Any]:
    pos, cost, hits = TRAVELER_START, 0.0, 0
    phantom: Optional[bool] = None
    for tt in targets:
        path, c = dijkstra(env, pos, tt, role)
        if path is None:
            continue
        cost += c
        hits += sum(1 for cell in path[1:]
                    if env.cell(*cell) == HAZARD)
        pos = tt
        hit = env.cell(*tt) == cell_kind
        if phantom is None:
            phantom = not hit
        if hit:
            return {"cost": cost, "phantom": phantom, "hits": hits,
                    "refused": False, "success_first": not phantom}
    bc, bh = blind_tag(env, pos, role, cell_kind)
    return {"cost": cost + bc, "phantom": bool(phantom),
            "hits": hits + bh, "refused": phantom is None,
            "success_first": False}


# ── song building under config + noise ─────────────────────────────

def build_song_cfg(env, path, fpf, cfg: Config):
    couplets, last_xy, last_idx = [], path[0], -10
    for i, xy in enumerate(path):
        is_last = i == len(path) - 1
        sig = fpf(env, xy) if cfg.landmarks else {}
        informative = len(sig) >= 2 and any(
            not k.startswith("void") for k in sig)
        if not is_last and (not informative or i - last_idx < 2):
            continue
        couplets.append({
            "sig": sig,
            "beat": ((xy[0] - last_xy[0], xy[1] - last_xy[1])
                     if cfg.beats else None)})
        last_xy, last_idx = xy, i
    return couplets


# ── configs: arms and ablations ────────────────────────────────────

def configs() -> Dict[str, Tuple[Config, bool]]:
    """name -> (Config, communicating). Canonical registry lives in
    songlines.config.ARMS --- one config per arm, no hand-forks."""
    from songlines.config import ARMS
    return dict(ARMS)


# ── one seeded run of one config ───────────────────────────────────

def run_cell(name: str, seed: int, n_agents: int, n_episodes: int,
             noise: float, wire_cap_kb: float = 0.0,
             mem_cap_kb: float = 0.0, poison: int = 0
             ) -> Dict[str, Any]:
    cfg, comm = configs()[name]
    if noise > 0:
        cfg = Config(**(cfg.__dict__ | {"sim_threshold": 0.75}))
    rng = np.random.default_rng(seed * 7 + 1)
    roles = ["fragile" if i % 2 == 0 else "robust"
             for i in range(n_agents)]
    agents = [SonglineAgent(i, roles[i],
                            cfg if comm or True else cfg)
              for i in range(n_agents)]
    # the independent baseline stays strong: no world-clock gate on
    # one's own memory (S3 lesson)
    if not comm:
        for ag in agents:
            ag.cfg = Config(**(cfg.__dict__ | {"world_clock": False}))
    fpf = make_fp(noise, rng)
    world = World(seed)
    plog: Dict[int, List[Record]] = {i: [] for i in range(n_agents)}
    stats = {"cost": {r: [] for r in ROLES}, "hits": 0, "dup": 0,
             "commits": 0, "phantom": 0, "refused": 0, "succ": 0,
             "wire_bits": 0, "n": 0}

    def utility_fn(env, agent, song, intent):
        band_fps = {xy: fpf(env, xy) for xy in BAND}
        base = agent.targets(band_fps, intent)
        cand = song_target(song, band_fps, agent.cfg.sim_threshold)
        probe = base + ([cand] if cand is not None else [])
        role = ROLES[agent.role_name]
        kind = INTENTS[intent]
        without = walk(env, base, role, kind)["cost"]
        with_m = walk(env, probe, role, kind)["cost"]
        return without - with_m

    if UTILITY_OVERRIDE is not None:
        utility_fn = UTILITY_OVERRIDE(fpf)

    for t in range(n_episodes):
        assignments = [world.assign() for _ in range(n_agents)]
        episode_targets: Dict[GridXY, int] = {}
        for i, (fam, env, tg, ver) in enumerate(assignments):
            ag = agents[i]
            intent = "water" if (t + i) % 3 else "rest"
            ag.now = t
            ag.on_visit(env, fam, ver, t, utility_fn)
            band_fps = {xy: fpf(env, xy) for xy in BAND}
            targets = ag.targets(band_fps, intent)
            if cfg.reservations and comm and targets:
                stats["wire_bits"] += RESV_BITS
                targets = [tt for tt in targets
                           if episode_targets.get(tt) in (None, i)]
            r = walk(env, targets, ROLES[roles[i]], INTENTS[intent])
            cost = r["cost"]
            if targets:
                stats["commits"] += 1
                first = targets[0]
                if episode_targets.get(first, i) != i:
                    stats["dup"] += 1
                    if not (cfg.reservations and comm):
                        cost += CONTENTION
                else:
                    episode_targets[first] = i
            stats["cost"][roles[i]].append(cost)
            stats["hits"] += r["hits"]
            stats["phantom"] += int(r["phantom"])
            stats["refused"] += int(r["refused"])
            stats["succ"] += int(r["success_first"])
            stats["n"] += 1
            # formation
            path, _ = dijkstra(env, TRAVELER_START,
                               tg[intent], ROLES[roles[i]])
            if path is None:
                continue
            song = build_song_cfg(env, path, fpf, ag.cfg)
            role_u = {rn: utility_fn(env, ag, song, intent) if rn ==
                      ag.role_name else 0.0 for rn in ROLES}
            ag.form(song, intent, fam, ver, t, role_u)

        # budget caps (B1): evict oldest when over the memory cap
        if mem_cap_kb > 0:
            for ag in agents:
                while ag.records and ag.memory_bits() > mem_cap_kb * 8000:
                    oldest = min(range(len(ag.records)),
                                 key=lambda k: ag.records[k].t)
                    ag.records.pop(oldest)
                while ag.episodic and ag.memory_bits() > mem_cap_kb * 8000:
                    ag.episodic.pop(0)

        if comm and t % CADENCE == CADENCE - 1:
            for i, ag in enumerate(agents):
                sends = list(ag.outbox(t - CADENCE))
                if i < poison:
                    # adversarial peer (P1): corrupts its own songs
                    # and LAUNDERS mutated foreign records under
                    # their original origin (fake corroboration)
                    prng = np.random.default_rng(seed * 31 + t)
                    corrupted = []
                    for rec in sends:
                        song2 = [dict(c) for c in rec.song]
                        if song2 and song2[-1].get("beat"):
                            bx, by = song2[-1]["beat"]
                            song2[-1]["beat"] = (
                                bx + int(prng.integers(2, 5)),
                                by + int(prng.integers(2, 5)))
                        corrupted.append(Record(
                            song2, rec.intent, rec.family,
                            dict(rec.role_u), rec.origin, rec.uid,
                            rec.t, rec.version))
                    for lrec in plog[i][-3:]:
                        song2 = [dict(c) for c in lrec.song]
                        if song2 and song2[-1].get("beat"):
                            bx, by = song2[-1]["beat"]
                            song2[-1]["beat"] = (bx + 3, by + 3)
                        corrupted.append(Record(
                            song2, lrec.intent, lrec.family,
                            dict(lrec.role_u), lrec.origin,
                            (lrec.origin, lrec.t + 5000),
                            t, lrec.version))
                    sends = corrupted
                for rec in sends:
                    if wire_cap_kb > 0 and stats["wire_bits"] \
                            >= wire_cap_kb * 8000:
                        break
                    stats["wire_bits"] += record_bits(rec, ag.cfg)
                    for j, other in enumerate(agents):
                        if j != i:
                            other.receive(rec, sender=i)
                            if j < poison:
                                plog[j].append(rec)

    return {"config": name, "seed": seed, "noise": noise,
            "agents": n_agents, "episodes": n_episodes,
            "group_cost": {r: float(np.mean(v))
                           for r, v in stats["cost"].items()},
            "success_first": stats["succ"] / stats["n"],
            "fail_open": stats["phantom"] / stats["n"],
            "refusal": stats["refused"] / stats["n"],
            "hazard_hits": stats["hits"] / stats["n"],
            "duplicate_rate": stats["dup"] / max(1, stats["commits"]),
            "wire_bits": stats["wire_bits"],
            "memory_bits": float(np.mean([a.memory_bits()
                                          for a in agents])),
            "match_ops": float(np.mean([a.match_ops
                                        for a in agents]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 102])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--wire-cap-kb", type=float, default=0.0)
    ap.add_argument("--mem-cap-kb", type=float, default=0.0)
    ap.add_argument("--poison", type=int, default=0)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/i1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "i1_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "H1": "full <= every main arm on group cost (both "
                      "roles); every ablation worse than full on >=1 "
                      "of {cost, fail-open, memory bits, duplicates}",
                "H2": "at 10% noise: full fail-open < 0.05 and "
                      "full < independent on both roles",
                "H3": "full < independent at N = 4, 6, 8",
                "protocol": "test seeds 100+ never used for tuning; "
                            "all thresholds frozen from prior waves; "
                            "paired assignment streams per seed; "
                            "sim_threshold 0.75 under noise "
                            "(registered)",
            }, f, indent=2)
    tag = ""
    if a.wire_cap_kb or a.mem_cap_kb:
        tag += f"_cap{a.mem_cap_kb:g}m{a.wire_cap_kb:g}w"
    if a.poison:
        tag += f"_poison{a.poison}"
    shard = (f"i1_{a.config}_n{a.noise}_a{a.agents}_e{a.episodes}"
             f"{tag}_s{a.seeds[0]}-{a.seeds[1]}.jsonl")
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            row = run_cell(a.config, seed, a.agents, a.episodes,
                           a.noise, a.wire_cap_kb, a.mem_cap_kb,
                           a.poison)
            f.write(json.dumps(row) + "\n")
            print(f"{a.config} seed {seed}: "
                  f"frag {row['group_cost']['fragile']:.0f} "
                  f"rob {row['group_cost']['robust']:.0f} "
                  f"succ {row['success_first']:.2f} "
                  f"fo {row['fail_open']:.3f}", flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
