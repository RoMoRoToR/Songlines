"""C1 — the second substrate: continuous space, jittered features,
the SAME runtime.

Everything so far formed memory on a discrete grid with exact tags.
C1 moves formation AND consumption to a continuous 14x12 box: point
features (rocks, hazards) at real-valued positions, feature-position
jitter on every observation, continuous beat vectors, tolerance-based
loop closure, and role costs as travel distance plus hazard-proximity
penalties. The runtime is reused UNCHANGED --- songs are the same
dicts; only the perception front end (binned constellation keys), the
walker and the world are continuous. If the method's advantage is a
grid artifact, it dies here.

Arms: independent, raw_history, songline_full (safety mode: anchor
consensus 2, closure tolerance 0.7).

Registered predictions:
  C1.1 (transfer of the effect): songline_full < independent on both
       roles at 300 episodes, 12 paired test seeds.
  C1.2 (safety bar on the second substrate): transport-level
       fail-open (wrong final commitment) <= 0.01 with consensus.
  C1.3: songline_full <= raw_history on both roles.

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_c1_continuous.py \
        --config songline_full --seeds 100 112 --episodes 300 \
        --out tmp/cluster/song_grammar/c1
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

from experiments.song_grammar.runtime import (
    Config, Record, SonglineAgent, record_bits, song_target)

Pos = Tuple[float, float]
WB, HB = 14.0, 12.0
START: Pos = (2.0, 2.0)
OBS_R = 3.0
BIN = 0.5
JITTER = 0.15
ACCEPT = 0.8
SCAN = 1.5      # terminal local search radius (arrive-and-look);
                # a dead-reckon miss beyond SCAN is a WRONG PLACE ---
                # the safety event; misses within SCAN cost a scan
ROLES_C = {"fragile": {"step": 1.0, "hazard": 12.0},
           "robust": {"step": 2.0, "hazard": 1.0}}
CADENCE, CONTENTION = 5, 25.0
INTENTS = ("water", "rest")


# ── the continuous world ───────────────────────────────────────────

class CWorld:
    """Families: rocks + targets = structure (family seed); hazards =
    texture (variant); conflict resamples the water (version++)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.families: List[int] = []
        self.state: Dict[int, Tuple[int, int]] = {}
        self.version: Dict[int, int] = {}
        self.fam_counter = seed * 100_000

    @staticmethod
    def build(fam: int, variant: int, widx: int):
        rs = np.random.default_rng(fam)
        rocks = [(float(rs.uniform(1, WB - 1)),
                  float(rs.uniform(1, HB - 1))) for _ in range(6)]
        bushes = [(float(rs.uniform(1, WB - 1)),
                   float(rs.uniform(1, HB - 1))) for _ in range(6)]
        waters = [(float(rs.uniform(10, 13)), float(rs.uniform(2, 10)))
                  for _ in range(3)]
        rest = (float(rs.uniform(1, 4)), float(rs.uniform(2, 10)))
        rh = np.random.default_rng(fam * 7919 + variant)
        hazards = [(float(rh.uniform(0.5, WB - 0.5)),
                    float(rh.uniform(0.5, HB - 0.5)))
                   for _ in range(14)]
        feats = ([("rock", p) for p in rocks]
                 + [("bush", p) for p in bushes]
                 + [("hazard", p) for p in hazards])
        return {"feats": feats, "water": waters[widx % 3],
                "rest": rest}

    def assign(self):
        if not self.families or self.rng.random() < 0.25:
            self.fam_counter += 1
            fam, kind = self.fam_counter, "new"
            variant, widx = 0, 0
        else:
            fam = int(self.rng.choice(self.families))
            kind = self.rng.choice(["repeat", "appearance", "conflict"],
                                   p=[0.35, 0.4, 0.25])
            variant, widx = self.state[fam]
            if kind == "appearance":
                variant = int(self.rng.integers(1, 40))
            elif kind == "conflict":
                widx = (widx + 1) % 3
        env = self.build(fam, variant, widx)
        if kind == "new":
            self.families.append(fam)
            self.version[fam] = 0
        elif kind == "conflict":
            self.version[fam] += 1
        self.state[fam] = (variant, widx)
        return fam, env, self.version[fam]


# ── continuous perception (jittered, binned constellations) ───────

def fpf_at(env, pos: Pos, rng: np.random.Generator) -> Dict[str, float]:
    sig: Dict[str, float] = {}
    for cls, (fx, fy) in env["feats"]:
        dx, dy = fx - pos[0], fy - pos[1]
        if dx * dx + dy * dy > OBS_R * OBS_R:
            continue
        jx = dx + float(rng.normal(0, JITTER))
        jy = dy + float(rng.normal(0, JITTER))
        sig[f"{cls}@{jx:.2f},{jy:.2f}"] = 1.0
    return sig


def _parse(sig):
    out = []
    for k in sig:
        cls, off = k.rsplit("@", 1)
        dx, dy = off.split(",")
        out.append((cls, float(dx), float(dy)))
    return out


def soft_sim(a: Dict[str, float], b: Dict[str, float],
             tol: float = 1.0) -> float:
    """Continuous constellation similarity: greedy class-respecting
    nearest matching with a positional tolerance (the honest
    continuous replacement for exact key equality)."""
    pa, pb = _parse(a), _parse(b)
    if not pa or not pb:
        return 0.0
    used, m = set(), 0
    for cls, dx, dy in pa:
        best_k, best_d = None, tol * tol
        for k, (c2, ex, ey) in enumerate(pb):
            if k in used or c2 != cls:
                continue
            d = (dx - ex) ** 2 + (dy - ey) ** 2
            if d <= best_d:
                best_k, best_d = k, d
        if best_k is not None:
            used.add(best_k)
            m += 1
    return m / math.sqrt(len(pa) * len(pb))


def band_points() -> List[Pos]:
    xs = [2 + 0.5 * i for i in range(12)]
    ys = [2 + 0.5 * i for i in range(16)]
    return [(x, y) for x in xs for y in ys]


# ── continuous execution ───────────────────────────────────────────

def seg_hazard(env, a: Pos, b: Pos) -> int:
    hits = 0
    d = math.dist(a, b)
    n = max(1, int(d / 0.5))
    for k in range(1, n + 1):
        px = a[0] + (b[0] - a[0]) * k / n
        py = a[1] + (b[1] - a[1]) * k / n
        for cls, (fx, fy) in env["feats"]:
            if cls == "hazard" and (fx - px) ** 2 + (fy - py) ** 2 \
                    <= 0.64:
                hits += 1
                break
    return hits


def walk_c(env, targets: List[Pos], role, true_target: Pos
           ) -> Dict[str, Any]:
    pos, cost, hits = START, 0.0, 0
    phantom: Optional[bool] = None
    for tt in targets:
        d = math.dist(pos, tt)
        cost += d * role["step"]
        hits += seg_hazard(env, pos, tt)
        pos = tt
        d_true = math.dist(tt, true_target)
        hit = d_true <= ACCEPT
        if not hit and d_true <= SCAN:
            cost += 2 * SCAN * role["step"]      # terminal scan
            hit = True
        if phantom is None:
            phantom = not hit
        if hit:
            cost += hits * role["hazard"]
            return {"cost": cost, "phantom": phantom, "hits": hits,
                    "success_first": not phantom, "refused": False}
    # blind: greedy tour over features until the target is stumbled on
    unvisited = [p for _, p in env["feats"]] + [true_target]
    while unvisited:
        nxt = min(unvisited, key=lambda p: math.dist(pos, p))
        unvisited.remove(nxt)
        cost += math.dist(pos, nxt) * role["step"]
        hits += seg_hazard(env, pos, nxt)
        pos = nxt
        if math.dist(pos, true_target) <= ACCEPT:
            break
    cost += hits * role["hazard"]
    return {"cost": cost, "phantom": bool(phantom), "hits": hits,
            "success_first": False, "refused": phantom is None}


# ── song formation on the continuous substrate ─────────────────────

def build_song_c(env, target: Pos, rng: np.random.Generator,
                 cfg: Config) -> List[Dict[str, Any]]:
    d = math.dist(START, target)
    n = max(2, int(d / 1.0))
    pts = [(START[0] + (target[0] - START[0]) * k / n,
            START[1] + (target[1] - START[1]) * k / n)
           for k in range(n + 1)]
    couplets, last, last_i = [], pts[0], -10
    for i, p in enumerate(pts):
        is_last = i == len(pts) - 1
        sig = fpf_at(env, p, rng) if cfg.landmarks else {}
        if not is_last and (len(sig) < 3 or i - last_i < 2):
            continue
        couplets.append({"sig": sig,
                         "beat": ((p[0] - last[0], p[1] - last[1])
                                  if cfg.beats else None)})
        last, last_i = p, i
    return couplets


# ── configs and the social loop ────────────────────────────────────

def configs() -> Dict[str, Tuple[Config, bool]]:
    """Continuous-substrate arms; canonical in
    songlines.config.CONTINUOUS_ARMS."""
    from songlines.config import CONTINUOUS_ARMS
    return dict(CONTINUOUS_ARMS)


def run_cell(name: str, seed: int, n_agents: int, n_episodes: int
             ) -> Dict[str, Any]:
    cfg, comm = configs()[name]
    if not comm:
        cfg = Config(**(cfg.__dict__ | {"world_clock": False}))
    rng = np.random.default_rng(seed * 11 + 3)
    roles = ["fragile" if i % 2 == 0 else "robust"
             for i in range(n_agents)]
    SonglineAgent.simfn = staticmethod(soft_sim)
    agents = [SonglineAgent(i, roles[i], cfg) for i in range(n_agents)]
    world = CWorld(seed)
    band = band_points()
    stats = {"cost": {r: [] for r in ROLES_C}, "succ": 0, "n": 0,
             "fail_open_final": 0, "commits": 0, "dup": 0,
             "wire_bits": 0, "hits": 0}

    def utility_fn(env, agent, song, intent):
        bf = {p: fpf_at(env, p, rng) for p in band}
        base = agent.targets(bf, intent)
        cand = song_target(song, bf, cfg.sim_threshold,
                           cfg.anchor_consensus, cfg.closure_tol,
                           soft_sim, cfg.unimodal_tol)
        probe = base + ([cand] if cand else [])
        role = ROLES_C[agent.role_name]
        true_t = env[intent]
        return (walk_c(env, base, role, true_t)["cost"]
                - walk_c(env, probe, role, true_t)["cost"])

    for t in range(n_episodes):
        assignments = [world.assign() for _ in range(n_agents)]
        episode_targets: Dict[Tuple[int, int], int] = {}
        for i, (fam, env, ver) in enumerate(assignments):
            ag = agents[i]
            intent = "water" if (t + i) % 3 else "rest"
            ag.now = t
            ag.on_visit(env, fam, ver, t, utility_fn)
            bf = {p: fpf_at(env, p, rng) for p in band}
            targets = ag.targets(bf, intent,
                                 observe_fn=lambda q: fpf_at(env, q, rng),
                                 start=START)
            if cfg.reservations and comm and targets:
                targets = [tt for tt in targets
                           if episode_targets.get(
                               (round(tt[0]), round(tt[1])))
                           in (None, i)]
            true_t = env[intent]
            r = walk_c(env, targets, ROLES_C[roles[i]], true_t)
            cost = r["cost"]
            if targets:
                stats["commits"] += 1
                key = (round(targets[0][0]), round(targets[0][1]))
                if episode_targets.get(key, i) != i:
                    stats["dup"] += 1
                    if not (cfg.reservations and comm):
                        cost += CONTENTION
                else:
                    episode_targets[key] = i
                # transport-level fail-open: committed AND final
                # target wrong (never reached the true one)
                if not r["success_first"] and r["phantom"] \
                        and not r["refused"]:
                    last = targets[-1]
                    if math.dist(last, true_t) > SCAN:
                        stats["fail_open_final"] += 1
            stats["cost"][roles[i]].append(cost)
            stats["succ"] += int(r["success_first"])
            stats["hits"] += r["hits"]
            stats["n"] += 1
            song = build_song_c(env, true_t, rng, cfg)
            role_u = {rn: utility_fn(env, ag, song, intent)
                      if rn == ag.role_name else 0.0
                      for rn in ROLES_C}
            ag.form(song, intent, fam, ver, t, role_u)

        if comm and t % CADENCE == CADENCE - 1:
            for i, ag in enumerate(agents):
                for rec in ag.outbox(t - CADENCE):
                    stats["wire_bits"] += record_bits(rec, ag.cfg)
                    for j, other in enumerate(agents):
                        if j != i:
                            other.receive(rec)

    return {"config": name, "seed": seed,
            "group_cost": {r: float(np.mean(v))
                           for r, v in stats["cost"].items()},
            "success_first": stats["succ"] / stats["n"],
            "fail_open_final": stats["fail_open_final"]
            / max(1, stats["commits"]),
            "hazard_hits": stats["hits"] / stats["n"],
            "duplicate_rate": stats["dup"] / max(1, stats["commits"]),
            "wire_bits": stats["wire_bits"],
            "memory_bits": float(np.mean([a.memory_bits()
                                          for a in agents]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 102])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/c1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "c1_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "C1.1": "songline_full < independent both roles",
                "C1.2": "transport-level fail-open <= 0.01 with "
                        "anchor consensus 2",
                "C1.3": "songline_full <= raw_history both roles",
                "C1.4": "songline_safe (anchor consensus 4, closure "
                        "1.2): transport wrong-place fail-open <= 0.01 "
                        "at social scale, trading reach (calibrated on "
                        "dev families 5001+, never on test seeds)",
                "substrate": "continuous 14x12 box, 3 feature "
                             "classes, jitter N(0,0.15), soft "
                             "constellation matching tol 1.0, band "
                             "0.5, closure 1.2, unimodal clusters "
                             "1.6 + centroid, consensus 2, accept "
                             "0.8, terminal scan 1.5 (fail-open = "
                             "wrong place > 1.5); constants frozen "
                             "on dev families 5001+, never on test "
                             "seeds",
            }, f, indent=2)
    shard = (f"c1_{a.config}_e{a.episodes}"
             f"_s{a.seeds[0]}-{a.seeds[1]}.jsonl")
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            row = run_cell(a.config, seed, a.agents, a.episodes)
            f.write(json.dumps(row) + "\n")
            print(f"{a.config} seed {seed}: "
                  f"frag {row['group_cost']['fragile']:.0f} "
                  f"rob {row['group_cost']['robust']:.0f} "
                  f"succ {row['success_first']:.2f} "
                  f"fo {row['fail_open_final']:.3f}", flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
