"""R0 — route-warp infrastructure + smoke (design: FRONTIER_ROUTE_WARP).

Semantic warp (W0--W6) transfers PLACE knowledge: broadcast snapshots
carry concept summaries but no graph edges --- the traveler learns
WHERE the water is and must derive the HOW itself.  A songline, in the
source culture, is the sequence: the song IS the route.  Route-warp
transfers traversed edges and measures phi_route --- the foreign share
of the edge mass along the path the planner commits to.

R0 delivers the infrastructure and one sharp smoke on a comb maze
(measured detour factor D >= 2):

  1. maze builder + detour measurement (BFS vs manhattan);
  2. RoutePeerMemory: place + OWN-edge snapshots broadcast on cadence K
     (no agent reads another's graph; transport only — the same
     privacy contract as peer_memory);
  3. phi_route and the strict RW* event: the traveler commits a
     connected path of >= L foreign edges it has never traversed;
  4. the three-arm preview of R1 on one maze:
       route arm  — follows the foreign path (edge-follower);
       place arm  — same foreign PLACE evidence, standard grid planner
                    (manhattan tier-1 + exploration tier-2);
       blind arm  — no foreign evidence at all.

Acceptance (registered):
  A1: measured detour factor D >= 2 on the smoke maze;
  A2: strict RW* logged: phi_route = 1.0, >= 5 foreign edges traversed,
      C* = 1;
  A3: t_succ(route) < t_succ(place) on the identical maze — route
      knowledge buys what place knowledge cannot (in an open grid the
      design predicts NO gap; that negative control is R1).

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_route_warp_r0.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.big_experiment.planner import (
    PlannerState, _turn_or_forward, _direction_toward, plan_action,
)
from multiagent_env import FORWARD, MultiAgentGridWorld, NOOP, WALL, WATER
from multiagent_env.grid_world import DIR_DELTAS

GridXY = Tuple[int, int]
Edge = Tuple[GridXY, GridXY]
OUT_DIR = "tmp/warp/r0_route"
GRID_W, GRID_H = 12, 10
WATER_TAG = "water_source"
ALPHA, TAU = 0.05, 0.30   # same gate constants; R2 will exercise decay
MIN_RW_EDGES = 5


def norm_edge(a: GridXY, b: GridXY) -> Edge:
    return (a, b) if a <= b else (b, a)


# ───────────────────────────────────── maze


def build_comb_maze() -> Tuple[MultiAgentGridWorld, GridXY, GridXY]:
    """Comb maze: vertical walls with alternating gaps force a
    serpentine path.  Start NW-ish, water SE-ish."""
    env = MultiAgentGridWorld(width=GRID_W, height=GRID_H, step_limit=300,
                              observation_radius=2, rng_seed=0)
    for wx, gap_y in [(2, 8), (4, 1), (6, 8), (8, 1)]:
        for y in range(GRID_H):
            if y != gap_y:
                env.set_cell(wx, y, WALL)
    start, water = (0, 1), (10, 8)
    env.set_cell(*water, WATER)
    return env, start, water


def passable(env: MultiAgentGridWorld, xy: GridXY) -> bool:
    x, y = xy
    return (0 <= x < env.width and 0 <= y < env.height
            and env.cell(x, y) != WALL)


def bfs_path(env: MultiAgentGridWorld, start: GridXY,
             goal: GridXY) -> Optional[List[GridXY]]:
    prev: Dict[GridXY, Optional[GridXY]] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = [cur]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1]
        for dx, dy in DIR_DELTAS.values():
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt not in prev and passable(env, nxt):
                prev[nxt] = cur
                q.append(nxt)
    return None


def detour_factor(env, start, goal) -> float:
    path = bfs_path(env, start, goal)
    manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    return (len(path) - 1) / manhattan if path else float("inf")


# ───────────────────────────────────── route memory (R0 scaffold)


class RoutePeerMemory:
    """Per-agent place + edge store with cadence-K broadcast of OWN
    experience.  Edges carry provenance; foreign edges are never
    re-broadcast (same contract as place snapshots in peer_memory)."""

    def __init__(self, agent_ids: List[str], broadcast_every_k: int = 4,
                 trust: float = 0.7, alpha: float = ALPHA):
        self.k = broadcast_every_k
        self.trust = trust
        self.alpha = alpha
        # aid -> {"places": {xy: {source}}, "edges": {edge: {source: meta}}}
        self.state: Dict[str, Dict[str, Any]] = {
            aid: {"places": {}, "edges": {}} for aid in agent_ids}

    def observe_place(self, aid: str, xy: GridXY, tag: str,
                      tick: int) -> None:
        if tag != WATER_TAG:
            return
        self.state[aid]["places"].setdefault(tuple(xy), set()).add(aid)

    def observe_move(self, aid: str, a: GridXY, b: GridXY,
                     tick: int) -> None:
        if a == b:
            return
        e = norm_edge(tuple(a), tuple(b))
        per_src = self.state[aid]["edges"].setdefault(e, {})
        meta = per_src.setdefault(aid, {"count": 0, "last_tick": tick})
        meta["count"] += 1
        meta["last_tick"] = tick

    def tick(self, tick_idx: int) -> None:
        if self.k <= 0 or tick_idx % self.k != 0:
            return
        for sender, st in self.state.items():
            own_places = [xy for xy, srcs in st["places"].items()
                          if sender in srcs]
            own_edges = {e: dict(meta[sender]) for e, meta in
                         st["edges"].items() if sender in meta}
            for receiver, rst in self.state.items():
                if receiver == sender:
                    continue
                for xy in own_places:
                    rst["places"].setdefault(xy, set()).add(sender)
                for e, m in own_edges.items():
                    rst["edges"].setdefault(e, {})[sender] = dict(m)

    # ── queries ───────────────────────────────────────────────────

    def known_waters(self, aid: str) -> List[GridXY]:
        return list(self.state[aid]["places"].keys())

    def edge_weight(self, aid: str, e: Edge, tick: int) -> float:
        per_src = self.state[aid]["edges"].get(e, {})
        w = 0.0
        for src, meta in per_src.items():
            trust = 1.0 if src == aid else self.trust
            age = max(0, tick - meta["last_tick"])
            w += trust * math.exp(-self.alpha * age) * math.log1p(meta["count"])
        return w

    def route_to_water(self, aid: str, from_xy: GridXY,
                       tick: int) -> Optional[Dict[str, Any]]:
        """BFS over gated edges to the nearest known water; returns the
        path with phi_route computed from the SAME weights the gate
        used (design §2.2, freezing at commit)."""
        st = self.state[aid]
        waters = self.known_waters(aid)
        if not waters:
            return None
        adj: Dict[GridXY, List[GridXY]] = {}
        for e, per_src in st["edges"].items():
            if self.edge_weight(aid, e, tick) < TAU:
                continue
            a, b = e
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        prev: Dict[GridXY, Optional[GridXY]] = {tuple(from_xy): None}
        q = deque([tuple(from_xy)])
        goal = None
        while q:
            cur = q.popleft()
            if cur in waters:
                goal = cur
                break
            for nxt in adj.get(cur, []):
                if nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        if goal is None:
            return None
        path = [goal]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        path = path[::-1]

        own_mass = foreign_mass = 0.0
        for a, b in zip(path, path[1:]):
            per_src = st["edges"].get(norm_edge(a, b), {})
            for src, meta in per_src.items():
                trust = 1.0 if src == aid else self.trust
                age = max(0, tick - meta["last_tick"])
                m = trust * math.exp(-self.alpha * age) * math.log1p(meta["count"])
                if src == aid:
                    own_mass += m
                else:
                    foreign_mass += m
        total = own_mass + foreign_mass
        return {"path": path, "goal": goal,
                "phi_route": foreign_mass / total if total > 0 else 0.0,
                "n_edges": len(path) - 1}


# ───────────────────────────────────── witness + traveler arms


def witness_walk(env, memory: RoutePeerMemory, start: GridXY,
                 water: GridXY) -> int:
    """The witness traverses the true shortest path once, recording
    places and edges into ITS OWN memory (teleop; its competence is
    not under test — its experience is the transferable object)."""
    path = bfs_path(env, start, water)
    for t, (a, b) in enumerate(zip(path, path[1:])):
        memory.observe_move("witness", a, b, t)
    memory.observe_place("witness", water, WATER_TAG, len(path))
    return len(path)


def follow_route_arm(env, memory: RoutePeerMemory,
                     start: GridXY) -> Dict[str, Any]:
    """Traveler commits the (fully foreign) route and follows it."""
    env.spawn("traveler", start_xy=start, target_tag=WATER_TAG, direction=0)
    ag = env.agents["traveler"]
    route = memory.route_to_water("traveler", start, tick=0)
    if route is None:
        return {"committed": False, "completed": False, "t_succ": None}
    rw_star = {
        "phi_route": round(route["phi_route"], 3),
        "n_edges": route["n_edges"],
        "strict_RW": route["phi_route"] >= 0.999
                     and route["n_edges"] >= MIN_RW_EDGES,
        "foreign_edges_traversed": 0,
    }
    path = route["path"]
    idx = 1
    for tick in range(300):
        if ag.success:
            break
        if idx >= len(path):
            break
        nxt = path[idx]
        cur = (ag.x, ag.y)
        if cur == nxt:
            idx += 1
            continue
        d = _direction_toward(cur, nxt)
        action = _turn_or_forward(ag.direction, d)
        prev = (ag.x, ag.y)
        env.step({"traveler": action})
        if (ag.x, ag.y) != prev:
            memory.observe_move("traveler", prev, (ag.x, ag.y), tick)
            rw_star["foreign_edges_traversed"] += 1
            if (ag.x, ag.y) == nxt:
                idx += 1
    rw_star["completed"] = ag.success
    t = None
    if ag.success:
        t = env.episode_step
    return {"committed": True, "rw_star": rw_star,
            "n_moves": rw_star["foreign_edges_traversed"],
            "completed": ag.success, "t_succ": t}


def place_arm(env, water: GridXY, start: GridXY) -> Dict[str, Any]:
    """Same foreign PLACE evidence (water location), standard planner."""
    env.spawn("traveler", start_xy=start, target_tag=WATER_TAG, direction=0)
    ag = env.agents["traveler"]
    ps = PlannerState("traveler")
    n_moves = 0
    for tick in range(300):
        if ag.success:
            break
        action = plan_action(ps, env, [water], tick, "route-r0-place")
        prev = (ag.x, ag.y)
        env.step({"traveler": action})
        if (ag.x, ag.y) != prev:
            n_moves += 1
    return {"completed": ag.success, "n_moves": n_moves,
            "t_succ": env.episode_step if ag.success else None}


def blind_arm(env, start: GridXY) -> Dict[str, Any]:
    env.spawn("traveler", start_xy=start, target_tag=WATER_TAG, direction=0)
    ag = env.agents["traveler"]
    ps = PlannerState("traveler")
    n_moves = 0
    for tick in range(300):
        if ag.success:
            break
        action = plan_action(ps, env, [], tick, "route-r0-blind")
        prev = (ag.x, ag.y)
        env.step({"traveler": action})
        if (ag.x, ag.y) != prev:
            n_moves += 1
    return {"completed": ag.success, "n_moves": n_moves,
            "t_succ": env.episode_step if ag.success else None}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "r0_registered.json"), "w") as f:
        json.dump({
            "A1": "measured detour factor D >= 2 on the comb maze",
            "A2": "strict RW*: phi_route=1.0, >=5 foreign edges, C*=1",
            "A3": "t_succ(route) < t_succ(place) on the identical maze",
        }, f, indent=2)

    env0, start, water = build_comb_maze()
    D = detour_factor(env0, start, water)
    print(f"comb maze: start={start} water={water} detour factor D={D:.2f}")

    # witness records experience, one broadcast wave transfers it
    memory = RoutePeerMemory(["witness", "traveler"], broadcast_every_k=4)
    path_len = witness_walk(env0, memory, start, water)
    memory.tick(4)  # broadcast wave
    print(f"witness path: {path_len - 1} edges; broadcast delivered")

    env_r, _, _ = build_comb_maze()
    route = follow_route_arm(env_r, memory, start)
    env_p, _, _ = build_comb_maze()
    place = place_arm(env_p, water, start)
    env_b, _, _ = build_comb_maze()
    blind = blind_arm(env_b, start)

    print(f"route arm: {route}")
    print(f"place arm: {place}")
    print(f"blind arm: {blind}")

    a1 = D >= 2.0
    rw = route.get("rw_star", {})
    a2 = (route["completed"] and rw.get("strict_RW", False)
          and rw.get("phi_route", 0) >= 0.999)
    a3 = (route["t_succ"] is not None
          and (place["t_succ"] is None or route["t_succ"] < place["t_succ"]))

    verdict = {"A1_detour_ge_2": a1, "A2_strict_RW_completed": a2,
               "A3_route_beats_place": a3,
               "detour_factor": round(D, 2),
               "t_succ": {"route": route["t_succ"], "place": place["t_succ"],
                          "blind": blind["t_succ"]}}
    with open(os.path.join(OUT_DIR, "r0_results.json"), "w") as f:
        json.dump({"route": route, "place": place, "blind": blind,
                   "verdict": verdict}, f, indent=2, default=str)

    print("=" * 60)
    for k, v in verdict.items():
        if isinstance(v, bool):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  t_succ: route={route['t_succ']} place={place['t_succ']} "
          f"blind={blind['t_succ']}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/r0_results.json")


if __name__ == "__main__":
    main()
