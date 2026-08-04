"""W10 — landmark vocabulary ablation: what makes a landmark.

The identity machinery of W7--W9 rides on a fixed landmark vocabulary
Sigma = {wall, hazard_edge, water_source, void}: the tag classes allowed
into a place fingerprint.  W7--W9 established that constellations over
this vocabulary recover secret frames exactly and fail closed when the
world cannot support identification.  W10 makes the vocabulary itself
the treatment: which feature classes actually CARRY the correspondence,
and what happens to the identity machinery as the alphabet is
impoverished?

Setup: witness/traveler as in W7 (witness sweeps the whole world in a
secretly shifted frame; the traveler sweeps an interior band in the true
frame and never sees the water).  Worlds carry BOTH scattered walls and
scattered hazards, so two content classes are available inside the band.
No planner is run: the outcome of interest is the alignment itself and
the transported water target (exact-offset recovery implies the
transported lock is the true water, wrong-offset transport implies a
phantom lock -- the fail-open event).

Arms (vocabularies):
  full         {wall, hazard_edge, water_source, void}
  no_void      {wall, hazard_edge, water_source}
  no_hazard    {wall, water_source, void}
  no_wall      {hazard_edge, water_source, void}
  hazard_only  {hazard_edge}
  wall_only    {wall}
  void_only    {void}   (excluded by the content rule -- the corner-trap
                         lesson -- so it must transport nothing)

Registered predictions (written before execution):
  A1 (fail-safe ablation): across ALL vocabularies and cells, transport
     happens only through the exact offset -- ablation converts recovery
     into refusal (fail closed), never into a wrong lock (zero fail-open
     cells).
  A2 (void is scaffolding, not identity): dropping 'void' leaves the
     recovery rate unchanged relative to full Sigma.
  A3 (a single content class suffices): hazard_only recovers the exact
     offset in >= 0.9 of cells; wall_only in the majority (> 0.5);
     void_only transports in 0 cells.
  A4 (attribution): under full Sigma, hazard_edge carries the plurality
     of key mass among consensus pairs (hazards are sparser and less
     self-similar than walls in these worlds).

Usage::

    PYTHONPATH=. python experiments/warp/exp_warp_landmark_ablation.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.warp.semantic_identity import align_frames, fingerprint
from multiagent_env import HAZARD, WALL, WATER, MultiAgentGridWorld

GridXY = Tuple[int, int]
OUT_DIR = "tmp/warp/w10_landmark_ablation"
W, H = 14, 12
TRAVELER_BAND = [(x, y) for y in range(2, 10) for x in range(2, 8)]
OFFSETS = [(3, 2), (-4, 3), (6, -2)]
SEEDS = range(12)
HAZARD_DENSITY = 0.10
WALL_DENSITY = 0.08

VOCABULARIES: Dict[str, set] = {
    "full": {"wall", "hazard_edge", "water_source", "void"},
    "no_void": {"wall", "hazard_edge", "water_source"},
    "no_hazard": {"wall", "water_source", "void"},
    "no_wall": {"hazard_edge", "water_source", "void"},
    "hazard_only": {"hazard_edge"},
    "wall_only": {"wall"},
    "void_only": {"void"},
}


def build_world(seed: int) -> Tuple[MultiAgentGridWorld, GridXY]:
    rng = np.random.default_rng(seed)
    env = MultiAgentGridWorld(width=W, height=H, step_limit=1,
                              observation_radius=2, rng_seed=seed)
    water = (int(rng.integers(10, 13)), int(rng.integers(2, 10)))
    env.set_cell(*water, WATER)
    for kind, density in ((HAZARD, HAZARD_DENSITY), (WALL, WALL_DENSITY)):
        n = int(round(density * W * H))
        placed = 0
        while placed < n:
            xy = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            if xy != water and env.cell(*xy) == 0:
                env.set_cell(*xy, kind)
                placed += 1
    return env, water


def cells_around(env: MultiAgentGridWorld, x: int, y: int,
                 radius: int = 2) -> List[Dict[str, Any]]:
    """In-grid cells within the observation window (off-grid -> void)."""
    out = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            cx, cy = x + dx, y + dy
            if 0 <= cx < W and 0 <= cy < H:
                out.append({"xy": (cx, cy), "tag": env.cell_tag(cx, cy)})
    return out


def sweep_fps(env: MultiAgentGridWorld, cells_xy: List[GridXY],
              offset: GridXY, vocab: set
              ) -> Dict[GridXY, Dict[str, float]]:
    """Fingerprints of swept cells, expressed in a private frame."""
    fps: Dict[GridXY, Dict[str, float]] = {}
    for (x, y) in cells_xy:
        pxy = (x + offset[0], y + offset[1])
        cells = [{"xy": (int(c["xy"][0]) + offset[0],
                         int(c["xy"][1]) + offset[1]), "tag": c["tag"]}
                 for c in cells_around(env, x, y)]
        fps[pxy] = fingerprint(pxy, cells, salient=vocab)
    return fps


def key_class(key: str) -> str:
    return key.rsplit("@", 1)[0]


def run_cell(vocab_name: str, seed: int, offset: GridXY) -> Dict[str, Any]:
    env, water = build_world(seed)
    vocab = VOCABULARIES[vocab_name]
    full = [(x, y) for y in range(H)
            for x in (range(W) if y % 2 == 0 else range(W - 1, -1, -1))]
    witness_fps = sweep_fps(env, full, offset, vocab)
    traveler_fps = sweep_fps(env, TRAVELER_BAND, (0, 0), vocab)

    res = align_frames(traveler_fps, witness_fps)
    transported: Optional[GridXY] = None
    if res.offset is not None:
        dx, dy = res.offset
        # witness recorded the water in its private frame
        wx, wy = water[0] + offset[0], water[1] + offset[1]
        transported = (wx + dx, wy + dy)

    # attribution over consensus pairs (pairs agreeing with the modal
    # delta): which tag classes the matched constellations are made of
    attribution: Counter = Counter()
    if res.offset is not None:
        for fxy, oxy in res.matched_pairs:
            if (oxy[0] - fxy[0], oxy[1] - fxy[1]) != res.offset:
                continue
            for key in traveler_fps[oxy]:
                attribution[key_class(key)] += 1

    return {
        "vocab": vocab_name, "seed": seed, "offset": list(offset),
        "water": list(water),
        "n_pairs": res.n_matches,
        "n_ambiguous": res.n_ambiguous,
        "recovered": res.offset is not None,
        "offset_exact": (res.offset is not None
                         and tuple(res.offset) == (-offset[0], -offset[1])),
        "transported": list(transported) if transported else None,
        "lock_is_true_water": transported == water,
        "fail_open": (transported is not None and transported != water),
        "attribution": dict(attribution),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "w10_registered.json"), "w") as f:
        json.dump({
            "A1_fail_safe": "across all vocabularies, zero fail-open "
                            "cells: transport implies the exact offset; "
                            "ablation converts recovery into refusal",
            "A2_void_scaffolding": "no_void recovery rate == full "
                                   "recovery rate",
            "A3_single_class": "hazard_only recovery >= 0.9; wall_only "
                               "> 0.5; void_only transports in 0 cells",
            "A4_attribution": "under full Sigma, hazard_edge holds the "
                              "plurality of consensus-pair key mass",
        }, f, indent=2)

    rows: List[Dict[str, Any]] = []
    for vocab_name in VOCABULARIES:
        for seed in SEEDS:
            for offset in OFFSETS:
                rows.append(run_cell(vocab_name, seed, offset))
    with open(os.path.join(OUT_DIR, "w10_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def sel(vocab_name: str) -> List[Dict[str, Any]]:
        return [r for r in rows if r["vocab"] == vocab_name]

    def rate(rs: List[Dict[str, Any]], key: str) -> float:
        return sum(1 for r in rs if r[key]) / len(rs) if rs else float("nan")

    summary: Dict[str, Any] = {}
    for vocab_name in VOCABULARIES:
        rs = sel(vocab_name)
        summary[vocab_name] = {
            "recovery_exact": rate(rs, "offset_exact"),
            "transport": rate(rs, "recovered"),
            "fail_open": sum(1 for r in rs if r["fail_open"]),
            "mean_pairs": float(np.mean([r["n_pairs"] for r in rs])),
            "lock_precision": (
                sum(1 for r in rs if r["lock_is_true_water"])
                / max(1, sum(1 for r in rs if r["transported"]))),
        }

    attribution: Counter = Counter()
    for r in sel("full"):
        attribution.update(r["attribution"])
    total_mass = sum(attribution.values())
    attribution_share = {k: round(v / total_mass, 3)
                         for k, v in attribution.most_common()}

    a1 = all(r["fail_open"] is False for r in rows)
    a2 = summary["no_void"]["recovery_exact"] == \
        summary["full"]["recovery_exact"]
    a3 = (summary["hazard_only"]["recovery_exact"] >= 0.9
          and summary["wall_only"]["recovery_exact"] > 0.5
          and summary["void_only"]["transport"] == 0.0)
    a4 = (max(attribution_share, key=attribution_share.get)
          == "hazard_edge")
    verdict = {"A1_fail_safe_ablation": a1,
               "A2_void_is_scaffolding": a2,
               "A3_single_content_class_suffices": a3,
               "A4_hazard_plurality": a4}

    with open(os.path.join(OUT_DIR, "w10_results.json"), "w") as f:
        json.dump({"summary": summary,
                   "attribution_share_full": attribution_share,
                   "verdict": verdict}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("attribution (full):", json.dumps(attribution_share))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/w10_results.json")


if __name__ == "__main__":
    main()
