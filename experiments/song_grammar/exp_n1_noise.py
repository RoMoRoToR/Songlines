"""N1 — the identity machinery under independent FN x FP perception
noise (reviewer item 4, isolated).

Witness--traveler never-seen-target transport (the S0/L1 core) with a
perceptual front end: every observation independently drops true
features (false negatives) and hallucinates spurious ones (false
positives); the matching margin is loosened from exact equality to
cosine >= 0.75 whenever noise > 0 (registered).

Measured per (FN, FP) cell: transport coverage (exact target),
fail-open rate (wrong target committed), refusal rate --- the
fail-closed contract must degrade into REFUSALS, not into wrong
locks, under real noisy predictions and not only under artificial
class removal (W10).

Registered predictions:
  N1.1 (fail-closed survives noise): fail-open rate <= 0.05 in every
       cell up to 20% FN x 20% FP.
  N1.2 (graceful coverage): exact-transport coverage at 10%/10% >=
       0.5 of the clean value.

Usage::

    PYTHONPATH=. python experiments/song_grammar/exp_n1_noise.py \
        --seeds 24 --out tmp/cluster/song_grammar/n1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.song_grammar.exp_s0_song_smoke import (
    BAND, TRAVELER_START, WITNESS_START, BAND_WAYPOINT, bfs_path)
from experiments.song_grammar.exp_i1_integration import ALL_TAGS
from experiments.song_grammar.runtime import Config, song_target
from experiments.song_grammar.exp_i1_integration import build_song_cfg
from experiments.warp.exp_warp_landmark_ablation import (
    H, W, build_world, cells_around)
from experiments.warp.semantic_identity import fingerprint
from multiagent_env import WALL

GridXY = Tuple[int, int]
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
CONSENSUS = 1


def make_fp2(fn: float, fpr: float, rng: np.random.Generator):
    def fpf(env, xy: GridXY) -> Dict[str, float]:
        sig = fingerprint(xy, cells_around(env, *xy))
        out = {k: v for k, v in sig.items() if rng.random() >= fn}
        for _ in range(int(rng.binomial(4, fpr))):
            tag = ALL_TAGS[int(rng.integers(len(ALL_TAGS)))]
            dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            out[f"{tag}@{dx},{dy}"] = 1.0
        return out
    return fpf


def cell_stats(fn: float, fpr: float, n_seeds: int) -> Dict[str, float]:
    exact = fail_open = refuse = n = 0
    sim = 0.999 if fn == 0 and fpr == 0 else 0.75
    cfg = Config(sim_threshold=sim)
    for seed in range(n_seeds):
        env, water = build_world(seed)
        if env.cell(*TRAVELER_START) == WALL:
            continue
        leg1 = bfs_path(env, WITNESS_START, BAND_WAYPOINT)
        leg2 = bfs_path(env, BAND_WAYPOINT, water)
        if leg1 is None or leg2 is None:
            continue
        rng = np.random.default_rng(seed * 977 + int(fn * 100) * 31
                                    + int(fpr * 100))
        fpf = make_fp2(fn, fpr, rng)
        song = build_song_cfg(env, leg1 + leg2[1:], fpf, cfg)
        band_fps = {xy: fpf(env, xy) for xy in BAND}
        n += 1
        t = song_target(song, band_fps, sim, min_anchors=CONSENSUS)
        if t is None:
            refuse += 1
        elif t == water:
            exact += 1
        else:
            fail_open += 1
    return {"n": n, "coverage": exact / max(1, n),
            "fail_open": fail_open / max(1, n),
            "refusal": refuse / max(1, n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--consensus", type=int, default=1)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/n1")
    a = ap.parse_args()
    global CONSENSUS
    CONSENSUS = a.consensus
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "n1_registered.json"), "w") as f:
        json.dump({
            "consensus": a.consensus,
            "N1v2": "with anchor consensus >= 2: fail-open <= 0.01 in "
                    "every cell up to 20%x20% (the <1% safety bar); "
                    "coverage cost reported honestly",
            "N1.1": "fail-open <= 0.05 in every cell up to 20%x20%",
            "N1.2": "coverage at 10%/10% >= 0.5 of clean coverage",
            "margin": "cosine >= 0.75 under noise (registered)",
        }, f, indent=2)
    grid = {}
    for fn in LEVELS:
        for fpr in LEVELS:
            grid[f"fn{fn}_fp{fpr}"] = cell_stats(fn, fpr, a.seeds)
            g = grid[f"fn{fn}_fp{fpr}"]
            print(f"FN {fn:.2f} FP {fpr:.2f}: cov {g['coverage']:.2f} "
                  f"fo {g['fail_open']:.3f} ref {g['refusal']:.2f}",
                  flush=True)
    clean = grid["fn0.0_fp0.0"]["coverage"]
    n11 = all(v["fail_open"] <= 0.05 for k, v in grid.items()
              if "fn0.3" not in k and "fp0.3" not in k)
    n12 = grid["fn0.1_fp0.1"]["coverage"] >= 0.5 * clean
    verdict = {"N1.1_fail_closed_survives_noise": n11,
               "N1.2_graceful_coverage": n12}
    with open(os.path.join(a.out, "n1_results.json"), "w") as f:
        json.dump({"grid": grid, "verdict": verdict}, f, indent=2)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Saved: {a.out}/n1_results.json")


if __name__ == "__main__":
    main()
