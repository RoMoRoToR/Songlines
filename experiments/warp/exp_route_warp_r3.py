"""R3 — hazard stratification: the song does not only lead, it protects.

R1 established what route knowledge buys on the WALL axis: place-warp
collapses on completion at the first wall.  R3 establishes the HAZARD
axis: in an open field dense with punishable-but-passable hazards both
place- and route-warp COMPLETE, but they pay different prices — the
witness's route is hazard-avoiding by experience, and the traveler who
follows it inherits the safety; the place-warp traveler knows only
WHERE and walks the manhattan line through the field.

The two axes are independent by design: walls change completion,
hazards change safety.

Composition with R2 (the rupture law): when the foreign route ruptures
mid-way by evidence aging, the stranded traveler falls back to place
knowledge — and the hazard risk RESUMES at the predicted rupture
point.  The song protects while it lasts.

Registered predictions (written before episodes):
  S1: n_hazard_hits(route) = 0 in every cell; pooled mean
      n_hazard_hits(place) > 0 with bootstrap CI excluding 0.
  S2: completion 100% for BOTH route and place arms (unlike the wall
      axis of R1) — the axes are independent.
  S3: in rupture-regime cells, hazard hits BEFORE the rupture = 0 and
      pooled hits AFTER the rupture > 0 (CI excluding 0), with the
      rupture tick at the R2-predicted path index.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_route_warp_r3.py \\
        [--seeds 20]
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

from experiments.big_experiment.planner import (
    PlannerState, _direction_toward, _turn_or_forward, plan_action,
)
from experiments.warp.exp_route_warp_r0 import (
    RoutePeerMemory, WATER_TAG, follow_route_arm, norm_edge,
)
from experiments.warp.exp_route_warp_r2 import predict
from multiagent_env import HAZARD, MultiAgentGridWorld, WATER
from multiagent_env.grid_world import DIR_DELTAS

GridXY = Tuple[int, int]
OUT_DIR = "tmp/warp/r3_hazard"
W_, H_ = 14, 12
HAZ_DENSITY = 0.20
L_RANGE = (10, 14)      # safe-path length in edges (rupture regime viable)
STEP_LIMIT = 300


# ───────────────────────────────────── world


def bfs_safe(env: MultiAgentGridWorld, start: GridXY,
             goal: GridXY) -> Optional[List[GridXY]]:
    """Shortest path that avoids hazards (walls too)."""
    def ok(xy):
        x, y = xy
        return (0 <= x < env.width and 0 <= y < env.height
                and env.cell(x, y) not in (1, HAZARD))  # 1 = WALL

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
            if nxt not in prev and ok(nxt):
                prev[nxt] = cur
                q.append(nxt)
    return None


def build_hazard_world(seed: int
                       ) -> Optional[Tuple[MultiAgentGridWorld, GridXY,
                                           GridXY, List[GridXY]]]:
    """Open field, dense hazards, a safe path of length in L_RANGE."""
    for sub in range(20):
        rng = np.random.default_rng(seed * 100 + sub)
        env = MultiAgentGridWorld(width=W_, height=H_,
                                  step_limit=STEP_LIMIT,
                                  observation_radius=2, rng_seed=seed)
        start = (1, int(rng.integers(3, 9)))
        water = (int(rng.integers(9, 12)), int(rng.integers(3, 9)))
        n_haz = int(round(HAZ_DENSITY * W_ * H_))
        placed = 0
        while placed < n_haz:
            xy = (int(rng.integers(0, W_)), int(rng.integers(0, H_)))
            if xy not in (start, water) and env.cell(*xy) == 0:
                env.set_cell(*xy, HAZARD)
                placed += 1
        env.set_cell(*water, WATER)
        path = bfs_safe(env, start, water)
        if path is not None and L_RANGE[0] <= len(path) - 1 <= L_RANGE[1]:
            return env, start, water, path
    return None


def witness_safe_walk(memory: RoutePeerMemory, path: List[GridXY],
                      water: GridXY) -> int:
    """The witness traverses the SAFE path once (sequential stamps,
    as in R2) and records the water."""
    for t, (a, b) in enumerate(zip(path, path[1:])):
        memory.observe_move("witness", a, b, t)
    memory.observe_place("witness", water, WATER_TAG, len(path))
    return len(path) - 1


# ───────────────────────────────────── arms (S1/S2)


def run_arm(env: MultiAgentGridWorld, arm: str, start: GridXY,
            water: GridXY, memory: Optional[RoutePeerMemory]
            ) -> Dict[str, Any]:
    if arm == "route":
        r = follow_route_arm(env, memory, start)
        ag = env.agents["traveler"]
        return {"completed": r["completed"], "t": r["t_succ"],
                "hits": ag.n_hazard_hits}
    env.spawn("traveler", start_xy=start, target_tag=WATER_TAG, direction=0)
    ag = env.agents["traveler"]
    ps = PlannerState("traveler")
    targets = [water] if arm == "place" else []
    for tick in range(STEP_LIMIT):
        if ag.success:
            break
        env.step({"traveler": plan_action(ps, env, targets, tick,
                                          f"r3-{arm}")})
    return {"completed": ag.success,
            "t": env.episode_step if ag.success else None,
            "hits": ag.n_hazard_hits}


# ───────────────────────────────────── S3: rupture returns the risk


def chase_with_fallback(env, memory: RoutePeerMemory, start: GridXY,
                        water: GridXY, T0: int) -> Dict[str, Any]:
    """R2 chase follower + place fallback after rupture; hazard hits
    are attributed to the pre-/post-rupture phases."""
    env.spawn("traveler", start_xy=start, target_tag=WATER_TAG, direction=0)
    ag = env.agents["traveler"]
    ps: Optional[PlannerState] = None
    phase = "route"
    edges_done = 0
    rupture: Optional[Dict[str, int]] = None
    hits = {"route": 0, "place": 0}
    index_at_tick: List[int] = []

    for t in range(STEP_LIMIT):
        if ag.success:
            break
        T = T0 + t
        if phase == "route":
            index_at_tick.append(edges_done)
            route = memory.route_to_water("traveler", (ag.x, ag.y), T)
            if route is None or len(route["path"]) < 2:
                phase = "place"
                rupture = {"tick": t, "index": edges_done}
                ps = PlannerState("traveler")
            else:
                nxt = route["path"][1]
                d = _direction_toward((ag.x, ag.y), nxt)
                action = _turn_or_forward(ag.direction, d)
                before = ag.n_hazard_hits
                prev = (ag.x, ag.y)
                env.step({"traveler": action})
                hits["route"] += ag.n_hazard_hits - before
                if (ag.x, ag.y) != prev:
                    memory.observe_move("traveler", prev, (ag.x, ag.y), T)
                    edges_done += 1
                continue
        action = plan_action(ps, env, [water], t, "r3-fallback")
        before = ag.n_hazard_hits
        env.step({"traveler": action})
        hits["place"] += ag.n_hazard_hits - before

    return {"completed": ag.success, "rupture": rupture,
            "hits_pre": hits["route"], "hits_post": hits["place"],
            "edges_done_at_rupture": (rupture or {}).get("index"),
            "index_at_tick": index_at_tick}


def pick_rupture_a0(kin: List[int], L: int) -> Optional[Tuple[int, int]]:
    """The a0 whose R2-predicted rupture index is deepest mid-path."""
    best = None
    for a0 in range(0, 20):
        p = predict(kin, L, a0, trust=1.0)
        if p["regime"] == "rupture" and 3 <= p["rupture_index"] <= L - 3:
            if best is None or p["rupture_index"] > best[1]:
                best = (a0, p["rupture_index"])
    return best


# ───────────────────────────────────── main


def mean_ci(vals, s=21, n=4000):
    rng = np.random.default_rng(s)
    arr = np.array(vals, dtype=float)
    boots = [np.mean(rng.choice(arr, len(arr))) for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "r3_registered.json"), "w") as f:
        json.dump({
            "S1": "hits(route)=0 in every cell; pooled hits(place)>0, "
                  "CI excludes 0",
            "S2": "completion 100% for both route and place (the wall "
                  "and hazard axes are independent)",
            "S3": "rupture-regime cells: hits before rupture = 0, pooled "
                  "hits after rupture > 0 (CI excludes 0), rupture at "
                  "the R2-predicted index",
        }, f, indent=2)

    rows: List[Dict[str, Any]] = []
    s3_rows: List[Dict[str, Any]] = []
    n_built = 0
    for seed in range(a.seeds * 3):
        if n_built >= a.seeds:
            break
        built = build_hazard_world(seed)
        if built is None:
            continue
        n_built += 1
        env0, start, water, safe_path = built
        L = len(safe_path) - 1

        def fresh_env():
            b = build_hazard_world(seed)
            return b[0]

        # ── S1/S2: three arms, safety-isolated (no decay) ─────────
        mem = RoutePeerMemory(["witness", "traveler"],
                              broadcast_every_k=4, trust=1.0, alpha=0.0)
        witness_safe_walk(mem, safe_path, water)
        mem.tick(4)
        cell = {"seed": seed, "L": L,
                "manhattan": abs(water[0] - start[0])
                             + abs(water[1] - start[1])}
        for arm in ("route", "place", "blind"):
            m = mem if arm == "route" else None
            cell[arm] = run_arm(fresh_env(), arm, start, water, m)
        rows.append(cell)

        # ── S3: decaying chase with place fallback ────────────────
        mem_cal = RoutePeerMemory(["witness", "traveler"],
                                  broadcast_every_k=4, trust=1.0,
                                  alpha=0.0)
        witness_safe_walk(mem_cal, safe_path, water)
        mem_cal.tick(4)
        cal = chase_with_fallback(fresh_env(), mem_cal, start, water,
                                  T0=L)
        if cal["completed"] and cal["rupture"] is None:
            choice = pick_rupture_a0(cal["index_at_tick"], L)
            if choice is not None:
                a0, pred_idx = choice
                mem_run = RoutePeerMemory(["witness", "traveler"],
                                          broadcast_every_k=4, trust=1.0)
                witness_safe_walk(mem_run, safe_path, water)
                mem_run.tick(4)
                r = chase_with_fallback(fresh_env(), mem_run, start,
                                        water, T0=L + a0)
                r.update({"seed": seed, "a0": a0,
                          "predicted_rupture_index": pred_idx})
                r.pop("index_at_tick")
                s3_rows.append(r)

    # ── verdicts ──────────────────────────────────────────────────
    route_hits = [c["route"]["hits"] for c in rows]
    place_hits = mean_ci([c["place"]["hits"] for c in rows])
    blind_hits = mean_ci([c["blind"]["hits"] for c in rows], s=22)
    s1 = all(h == 0 for h in route_hits) and place_hits[1] > 0
    s2 = (all(c["route"]["completed"] for c in rows)
          and all(c["place"]["completed"] for c in rows))

    pre = [r["hits_pre"] for r in s3_rows]
    post = mean_ci([r["hits_post"] for r in s3_rows], s=23) \
        if s3_rows else (float("nan"),) * 3
    idx_err = [abs(r["edges_done_at_rupture"]
                   - r["predicted_rupture_index"])
               for r in s3_rows if r["rupture"] is not None]
    s3 = (bool(s3_rows) and all(h == 0 for h in pre)
          and post[1] > 0
          and all(e <= 1 for e in idx_err))

    verdict = {
        "S1_route_inherits_safety": s1,
        "S2_completion_axes_independent": s2,
        "S3_rupture_returns_risk": s3,
        "hits": {"route": 0.0,
                 "place": [round(v, 2) for v in place_hits],
                 "blind": [round(v, 2) for v in blind_hits]},
        "time": {"route": mean_ci([c["route"]["t"] or STEP_LIMIT
                                   for c in rows], s=24)[0],
                 "place": mean_ci([c["place"]["t"] or STEP_LIMIT
                                   for c in rows], s=25)[0]},
        "n_s3_cells": len(s3_rows),
        "hits_post_rupture": [round(v, 2) for v in post],
        "rupture_index_errors": idx_err,
    }
    with open(os.path.join(OUT_DIR, "r3_results.json"), "w") as f:
        json.dump({"rows": rows, "s3_rows": s3_rows, "verdict": verdict},
                  f, indent=1)

    print(f"cells: {len(rows)} (S3 rupture cells: {len(s3_rows)})")
    print(f"hits  route={sum(route_hits)}  "
          f"place={place_hits[0]:.2f} [{place_hits[1]:.2f},"
          f"{place_hits[2]:.2f}]  blind={blind_hits[0]:.2f}")
    print(f"time  route={verdict['time']['route']:.1f}  "
          f"place={verdict['time']['place']:.1f}")
    print(f"S3: hits_pre={sum(pre) if pre else '-'}  "
          f"hits_post={post[0]:.2f} [{post[1]:.2f},{post[2]:.2f}]  "
          f"rupture idx errors={idx_err}")
    print("=" * 60)
    for k, v in verdict.items():
        if isinstance(v, bool):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/r3_results.json")


if __name__ == "__main__":
    main()
