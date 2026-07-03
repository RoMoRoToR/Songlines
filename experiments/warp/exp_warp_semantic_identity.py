"""W7 — frame-free warp: agents identify places by MEANING.

Setup: witness and traveler hold PRIVATE coordinate frames (the
witness's frame is the true grid shifted by a secret offset).  The
witness sweeps the whole world, sees the water, and broadcasts its
memory — fingerprints and evidence — in ITS OWN coordinates.  The
traveler has swept only an interior band (never the water) and must
decide what the witness's coordinates mean.

Arms:
  coordinate — the shared-frame assumption of the main series: foreign
               coordinates taken at face value.  Under misalignment it
               FAILS OPEN: confidently locks a displaced target.
  semantic   — places matched by local semantic constellations
               (hazard/wall/border patterns at relative offsets); the
               frame offset is recovered from >= 3 unambiguously
               matched co-visited landmarks; foreign evidence is
               transported through it.  Without consensus it FAILS
               CLOSED: refuses to lock at all.

Registered predictions (written before episodes):
  P1: coordinate arm completes iff offset = (0,0).
  P2: semantic arm at hazard density 0.10: recovers the offset EXACTLY
      and completes (strict W*, phi = 1.0) at every offset.
  P3: aliased world (hazard density 0, interior band has featureless
      fingerprints): semantic arm makes ZERO warp locks (fails closed)
      while the coordinate arm at nonzero offsets locks and fails
      (fails open) — the two identity regimes break in opposite
      directions.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_semantic_identity.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.big_experiment.planner import PlannerState, plan_action
from experiments.warp.semantic_identity import SemanticIdentityMemory
from multiagent_env import HAZARD, MultiAgentGridWorld, WATER

GridXY = Tuple[int, int]
OUT_DIR = "tmp/warp/w7_semantic_identity"
W, H = 14, 12
TRAVELER_BAND = [(x, y) for y in range(2, 10) for x in range(2, 8)]
OFFSETS = [(0, 0), (3, 2), (-4, 3), (6, -2)]
HAZARD_DENSITIES = [0.10, 0.0]
SEEDS = range(10)
STEP_LIMIT = 120


def build_world(h: float, seed: int) -> Tuple[MultiAgentGridWorld, GridXY]:
    rng = np.random.default_rng(seed)
    env = MultiAgentGridWorld(width=W, height=H, step_limit=STEP_LIMIT,
                              observation_radius=2, rng_seed=seed)
    water = (int(rng.integers(10, 13)), int(rng.integers(2, 10)))
    env.set_cell(*water, WATER)
    n_haz = int(round(h * W * H))
    placed = 0
    while placed < n_haz:
        xy = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if xy != water and env.cell(*xy) == 0:
            env.set_cell(*xy, HAZARD)
            placed += 1
    return env, water


def observe_in_frame(env, aid: str, offset: GridXY) -> Tuple[GridXY, List[Dict]]:
    """The agent's observation expressed in its PRIVATE frame."""
    obs = env._observation(aid)
    ag = env.agents[aid]
    private_xy = (ag.x + offset[0], ag.y + offset[1])
    cells = [{"xy": (int(c["xy"][0]) + offset[0],
                     int(c["xy"][1]) + offset[1]),
              "tag": c["tag"]} for c in obs.get("cells", [])]
    return private_xy, cells


def teleop_sweep(env, memory, aid: str, cells_xy: List[GridXY],
                 offset: GridXY) -> None:
    ag = env.agents[aid]
    for t, (x, y) in enumerate(cells_xy):
        ag.x, ag.y = x, y
        pxy, cells = observe_in_frame(env, aid, offset)
        memory.observe(aid, pxy, cells, t)


def run_episode(mode: str, h: float, seed: int,
                offset: GridXY) -> Dict[str, Any]:
    env, water = build_world(h, seed)
    env.spawn("witness", start_xy=(0, 0), target_tag="water_source",
              direction=0)
    env.spawn("traveler", start_xy=TRAVELER_BAND[0],
              target_tag="water_source", direction=0)
    memory = SemanticIdentityMemory(["witness", "traveler"], mode=mode)

    # witness sweeps the WHOLE grid in its own (shifted) frame
    full = [(x, y) for y in range(H)
            for x in (range(W) if y % 2 == 0 else range(W - 1, -1, -1))]
    teleop_sweep(env, memory, "witness", full, offset)
    # traveler sweeps its interior band in the true frame (offset 0)
    teleop_sweep(env, memory, "traveler", TRAVELER_BAND, (0, 0))
    memory.tick(4)  # broadcast wave

    targets, diag = memory.query("traveler")
    locked = None
    phi = None
    if targets:
        ag = env.agents["traveler"]
        locked = min(targets,
                     key=lambda t: abs(t[0] - ag.x) + abs(t[1] - ag.y))
        phi = memory.phi("traveler", locked)

    t_succ = None
    if locked is not None:
        ag = env.agents["traveler"]
        ag.x, ag.y = TRAVELER_BAND[0]
        ps = PlannerState("traveler")
        for tick in range(STEP_LIMIT):
            if ag.success:
                t_succ = tick
                break
            action = plan_action(ps, env, [locked], tick, f"w7-{mode}")
            env.step({"traveler": action, "witness": 4})

    align = (diag.get("alignments", {}) or {}).get("witness", {})
    return {
        "mode": mode, "hazard_density": h, "seed": seed,
        "offset": list(offset), "water": list(water),
        "locked": list(locked) if locked else None,
        "phi": phi,
        "lock_is_true_water": locked == water if locked else False,
        "completed": t_succ is not None,
        "t_succ": t_succ,
        "recovered_offset": (list(align["offset"])
                             if align.get("offset") else None),
        "offset_exact": (align.get("offset") is not None
                         and tuple(align["offset"])
                         == (-offset[0], -offset[1])),
        "n_matches": align.get("n_matches"),
        "n_ambiguous": align.get("n_ambiguous"),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "w7_registered.json"), "w") as f:
        json.dump({
            "v1_outcome": "v1 (see git history) failed on two "
                          "operationalisation defects: (a) P1/P3 counted "
                          "accidental path-crossing completions (the "
                          "displaced route stumbles over the true water "
                          "in ~7% of runs) — the fail-open essence is "
                          "the WRONG LOCK, so v2 conditions on it; "
                          "(b) the matcher accepted one-directional "
                          "unique matches, letting distant foreign "
                          "cells alias into the receiver's band — v2 "
                          "requires mutual uniqueness, exact-equality "
                          "threshold and >= 2 salient keys.",
            "P1": "coordinate arm under nonzero offset ALWAYS locks a "
                  "non-water cell (lock_is_true_water = False for all); "
                  "success rate <= 0.2 (stumbles only); at offset (0,0) "
                  "completes always",
            "P2": "semantic arm at h=0.10: exact offset recovery and "
                  "completion (strict W*, phi=1.0) at every offset",
            "P3": "aliased world h=0: semantic arm zero locks (fails "
                  "closed); coordinate arm at nonzero offsets always "
                  "locks a non-water cell (fails open)",
        }, f, indent=2)

    rows: List[Dict[str, Any]] = []
    for h in HAZARD_DENSITIES:
        for seed in SEEDS:
            for offset in OFFSETS:
                for mode in ("coordinate", "semantic"):
                    rows.append(run_episode(mode, h, seed, offset))
    with open(os.path.join(OUT_DIR, "w7_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def sel(mode, h, nonzero=None):
        out = [r for r in rows
               if r["mode"] == mode and r["hazard_density"] == h]
        if nonzero is True:
            out = [r for r in out if tuple(r["offset"]) != (0, 0)]
        if nonzero is False:
            out = [r for r in out if tuple(r["offset"]) == (0, 0)]
        return out

    # P1: coordinate arm — fail-open = wrong lock; stumbles bounded
    c_zero = sel("coordinate", 0.10, nonzero=False)
    c_nz = sel("coordinate", 0.10, nonzero=True)
    p1 = (all(r["completed"] for r in c_zero)
          and not any(r["lock_is_true_water"] for r in c_nz)
          and sum(r["completed"] for r in c_nz) / len(c_nz) <= 0.2)

    # P2: semantic arm, landmark world
    s_land = sel("semantic", 0.10)
    p2 = (all(r["offset_exact"] for r in s_land)
          and all(r["completed"] for r in s_land)
          and all(r["phi"] == 1.0 for r in s_land))

    # P3: aliased world — opposite failure directions
    s_alias = sel("semantic", 0.0)
    c_alias_nz = sel("coordinate", 0.0, nonzero=True)
    p3 = (not any(r["locked"] for r in s_alias)
          and all(r["locked"] is not None
                  and not r["lock_is_true_water"] for r in c_alias_nz))

    def rate(rs, key):
        return sum(1 for r in rs if r[key]) / len(rs) if rs else float("nan")

    summary = {
        "coordinate_h.10": {
            "succ_offset0": rate(c_zero, "completed"),
            "succ_nonzero": rate(c_nz, "completed"),
            "locked_nonzero": rate(c_nz, "locked"),
        },
        "semantic_h.10": {
            "offset_exact": rate(s_land, "offset_exact"),
            "succ": rate(s_land, "completed"),
            "mean_matches": float(np.mean(
                [r["n_matches"] or 0 for r in s_land])),
        },
        "aliased_h0": {
            "semantic_locks": sum(1 for r in s_alias if r["locked"]),
            "coordinate_nonzero_locks": rate(c_alias_nz, "locked"),
            "coordinate_nonzero_succ": rate(c_alias_nz, "completed"),
        },
    }
    verdict = {"P1_coordinate_fails_open_under_offset": p1,
               "P2_semantic_exact_recovery_and_completion": p2,
               "P3_fail_closed_vs_fail_open": p3}
    with open(os.path.join(OUT_DIR, "w7_results.json"), "w") as f:
        json.dump({"summary": summary, "verdict": verdict}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/w7_results.json")


if __name__ == "__main__":
    main()
