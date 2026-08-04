"""Shared substrate for UCSM stages 2-3 and the seven-arm comparison.

Adds to the U1 machinery:
  * heterogeneous ROLES (fast-fragile vs slow-robust) as cost profiles:
    the same route has different utility for different bodies, so a
    memory item must carry a utility PROFILE, not one weight;
  * hazard-aware Dijkstra walking (a robust witness sings routes
    THROUGH hazards, a fragile one around them);
  * world FAMILIES: walls+water = functional structure (persistent),
    hazards = appearance/risk texture (resampled per variant), plus a
    water-moved variant (the conflict case);
  * an episode STREAM generator (deterministic per seed) mixing
    repeats, appearance variants, conflicts and new families.

Everything is CPU-only and deterministic; designed for seed-sharded
cluster runs (jsonl rows per shard, aggregate separately).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.song_grammar.exp_s0_song_smoke import (
    BAND, BAND_WAYPOINT, TRAVELER_START, WITNESS_START, arm_song, fp_at)
from experiments.warp.exp_warp_landmark_ablation import H, W
from multiagent_env import HAZARD, WALL, WATER, MultiAgentGridWorld

GridXY = Tuple[int, int]

ROLES: Dict[str, Dict[str, float]] = {
    "fragile": {"step": 1.0, "hazard": 12.0},   # fast but fragile
    "robust": {"step": 2.0, "hazard": 1.0},     # slow but tough
}

N_WALLS, N_HAZARDS = 13, 17
WATER_CANDIDATES = 3


# ── world families ─────────────────────────────────────────────────

def family_world(fam_seed: int, variant: int = 0, water_idx: int = 0
                 ) -> Tuple[MultiAgentGridWorld, GridXY]:
    """Structure (walls + water slots) from fam_seed; appearance
    (hazard texture) from variant.  water_idx > 0 = the moved-water
    conflict variant of the same structure."""
    rng_s = np.random.default_rng(fam_seed)
    env = MultiAgentGridWorld(width=W, height=H, step_limit=1,
                              observation_radius=2, rng_seed=fam_seed)
    waters = []
    while len(waters) < WATER_CANDIDATES:
        c = (int(rng_s.integers(10, W - 1)), int(rng_s.integers(2, H - 2)))
        if c not in waters:
            waters.append(c)
    placed = 0
    while placed < N_WALLS:
        c = (int(rng_s.integers(0, W)), int(rng_s.integers(0, H)))
        if c not in waters and c != TRAVELER_START and c != WITNESS_START \
                and c != BAND_WAYPOINT and env.cell(*c) == 0:
            env.set_cell(*c, WALL)
            placed += 1
    water = waters[water_idx % WATER_CANDIDATES]
    env.set_cell(*water, WATER)
    rng_h = np.random.default_rng(fam_seed * 7919 + variant)
    placed = 0
    while placed < N_HAZARDS:
        c = (int(rng_h.integers(0, W)), int(rng_h.integers(0, H)))
        if env.cell(*c) == 0 and c != TRAVELER_START:
            env.set_cell(*c, HAZARD)
            placed += 1
    return env, water


# ── role-aware movement ────────────────────────────────────────────

def enter_cost(env, xy: GridXY, role: Dict[str, float]) -> float:
    c = env.cell(*xy)
    return role["step"] + (role["hazard"] if c == HAZARD else 0.0)


def dijkstra(env, start: GridXY, goal: GridXY, role: Dict[str, float]
             ) -> Tuple[Optional[List[GridXY]], float]:
    """Cheapest role-aware walk (steps + hazard penalties on entry)."""
    if env.cell(*start) == WALL or env.cell(*goal) == WALL:
        return None, float("inf")
    dist = {start: 0.0}
    prev: Dict[GridXY, Optional[GridXY]] = {start: None}
    pq = [(0.0, start)]
    while pq:
        d, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1], d
        if d > dist.get(cur, float("inf")):
            continue
        x, y = cur
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nxt[0] < W and 0 <= nxt[1] < H):
                continue
            if env.cell(*nxt) == WALL:
                continue
            nd = d + enter_cost(env, nxt, role)
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(pq, (nd, nxt))
    return None, float("inf")


def blind_cost(env, start: GridXY, role: Dict[str, float]) -> float:
    """Exploration sweep: BFS visitation, each visited cell paid at its
    role-aware entry cost (a fragile explorer bleeds on hazards)."""
    from collections import deque
    seen = {start}
    q = deque([start])
    total = 0.0
    while q:
        cur = q.popleft()
        total += enter_cost(env, cur, role)
        if env.cell(*cur) == WATER:
            return total
        x, y = cur
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (0 <= nxt[0] < W and 0 <= nxt[1] < H
                    and nxt not in seen and env.cell(*nxt) != WALL):
                seen.add(nxt)
                q.append(nxt)
    return total + role["step"] * W * H


# ── songs (parameterised grammar) ──────────────────────────────────

def informative(sig: Dict[str, float]) -> bool:
    return len(sig) >= 2 and any(not k.startswith("void") for k in sig)


def build_song_g(env, path: List[GridXY], gap: int = 2,
                 vmax: int = 0) -> List[Dict[str, Any]]:
    couplets: List[Dict[str, Any]] = []
    last_xy, last_idx = path[0], -10
    for i, xy in enumerate(path):
        is_last = i == len(path) - 1
        sig = fp_at(env, xy)
        if not is_last and (not informative(sig) or i - last_idx < gap):
            continue
        couplets.append({"sig": sig,
                         "beat": (xy[0] - last_xy[0], xy[1] - last_xy[1]),
                         "is_target": is_last})
        last_xy, last_idx = xy, i
    if vmax and len(couplets) > vmax:       # bounded-rationality budget
        keep = sorted(set(
            [0, len(couplets) - 1]
            + list(np.linspace(0, len(couplets) - 1, vmax).astype(int))))
        merged: List[Dict[str, Any]] = []
        prev_i = None
        for i in keep:
            c = dict(couplets[i])
            if prev_i is not None:
                bx = sum(couplets[k]["beat"][0] for k in range(prev_i + 1, i + 1))
                by = sum(couplets[k]["beat"][1] for k in range(prev_i + 1, i + 1))
                c["beat"] = (bx, by)
            merged.append(c)
            prev_i = i
        couplets = merged
    return couplets


def witness_song(env, water: GridXY, role: Dict[str, float],
                 gap: int = 2, vmax: int = 0):
    leg1, c1 = dijkstra(env, WITNESS_START, BAND_WAYPOINT, role)
    leg2, c2 = dijkstra(env, BAND_WAYPOINT, water, role)
    if leg1 is None or leg2 is None:
        return None
    return build_song_g(env, leg1 + leg2[1:], gap=gap, vmax=vmax)


# ── consumer replay (role-aware) ───────────────────────────────────

def consumer_cost(env, songs: List[List[Dict[str, Any]]],
                  role: Dict[str, float],
                  start: GridXY = TRAVELER_START) -> Dict[str, Any]:
    band_fps = {xy: fp_at(env, xy) for xy in BAND}
    pos, total = start, 0.0
    phantom_first: Optional[bool] = None
    used = 0
    for song in songs:
        res = arm_song(song, band_fps)
        t = res["transported"]
        if t is None:
            continue
        t = (int(t[0]), int(t[1]))
        path, cost = dijkstra(env, pos, t, role)
        if path is None:
            continue
        used += 1
        total += cost
        pos = t
        hit = env.cell(*t) == WATER
        if phantom_first is None:
            phantom_first = not hit
        if hit:
            return {"cost": total, "phantom_first": phantom_first,
                    "used": used}
    return {"cost": total + blind_cost(env, pos, role),
            "phantom_first": bool(phantom_first), "used": used}


def marginal_utility(env, songs: List[List[Dict[str, Any]]],
                     cand: List[Dict[str, Any]],
                     role: Dict[str, float]) -> float:
    return (consumer_cost(env, songs, role)["cost"]
            - consumer_cost(env, songs + [cand], role)["cost"])


# ── bit codec (same table as S0, registered there) ─────────────────

KEY_BITS, LEN_BITS, BEAT_BITS, COORD_BITS = 6, 4, 10, 8


def bits_of_song(song) -> int:
    return sum(LEN_BITS + KEY_BITS * len(c["sig"]) + BEAT_BITS
               for c in song)


def bits_of_snapshot(fps) -> int:
    return COORD_BITS + sum(COORD_BITS + LEN_BITS + KEY_BITS * len(s)
                            for s in fps.values())


# ── episode stream ─────────────────────────────────────────────────

@dataclass
class Episode:
    ep_id: str
    env: Any
    water: GridXY
    song: List[Dict[str, Any]]
    sender_role: str
    family: int
    kind: str          # new | repeat | appearance | conflict


def valid_world(env, water: GridXY) -> bool:
    if env.cell(*TRAVELER_START) == WALL:
        return False
    p, _ = dijkstra(env, TRAVELER_START, water, ROLES["robust"])
    return p is not None


def make_stream(seed: int, n_episodes: int, gap: int = 2,
                vmax: int = 0,
                kind_probs: Tuple[float, float, float] = (0.35, 0.40,
                                                          0.25),
                new_rate: float = 0.30) -> List[Episode]:
    """Deterministic mixed stream: new families, exact repeats,
    appearance variants (same structure, new hazard texture) and
    water-moved conflicts."""
    rng = np.random.default_rng(seed)
    stream: List[Episode] = []
    families: List[int] = []
    fam_counter = seed * 10_000
    tries = 0
    while len(stream) < n_episodes and tries < n_episodes * 20:
        tries += 1
        if not families or rng.random() < new_rate:
            kind, fam = "new", None
        else:
            kind = rng.choice(["repeat", "appearance", "conflict"],
                              p=list(kind_probs))
            fam = int(rng.choice(families))
        if kind == "new":
            fam_counter += 1
            fam = fam_counter
            env, water = family_world(fam, variant=0, water_idx=0)
        elif kind == "repeat":
            env, water = family_world(fam, variant=0, water_idx=0)
        elif kind == "appearance":
            env, water = family_world(fam, variant=int(rng.integers(1, 6)),
                                      water_idx=0)
        else:
            env, water = family_world(fam, variant=0,
                                      water_idx=int(rng.integers(1, 3)))
        if not valid_world(env, water):
            continue
        role_name = "fragile" if rng.random() < 0.5 else "robust"
        song = witness_song(env, water, ROLES[role_name],
                            gap=gap, vmax=vmax)
        if song is None:
            continue
        band_ok = arm_song(song, {xy: fp_at(env, xy) for xy in BAND})
        if band_ok["transported"] is None:
            continue
        if kind == "new":
            families.append(fam)
        stream.append(Episode(
            f"s{seed}e{len(stream)}", env, water, song, role_name,
            fam, kind))
    return stream


def eval_battery(stream: List[Episode], seed: int, n_fresh: int = 2,
                 max_families: int = 24) -> List[Episode]:
    """Held-out evaluation: latest true world of (a deterministic
    sample of) the stream's families, plus fresh appearance variants
    of the base structures never seen in the stream."""
    rng = np.random.default_rng(seed + 555)
    latest: Dict[int, Episode] = {}
    for ep in stream:
        latest[ep.family] = ep
    fams = sorted(latest.keys())
    if len(fams) > max_families:
        fams = sorted(rng.choice(fams, size=max_families,
                                 replace=False).tolist())
    battery: List[Episode] = [latest[f] for f in fams]
    for fam in fams[:6]:
        for k in range(n_fresh):
            env, water = family_world(fam, variant=100 + k, water_idx=0)
            if valid_world(env, water):
                battery.append(Episode(f"fresh{fam}v{k}", env, water,
                                       [], "robust", fam, "fresh"))
    return battery
