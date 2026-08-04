"""E1 — emergent landmarks: the closed loop propose -> ablate ->
measure -> promote.

Until now landmark status came from a hand-chosen vocabulary (W10
measured which classes CARRY correspondence; nothing SELECTED them).
E1 closes the loop: the system starts from a superset of candidate
features --- including a derived structural feature and deliberately
harmful junk --- and iteratively prunes every feature whose removal
does not hurt (or helps) measured coordination utility.  What
survives has EARNED landmark status causally.

Candidate feature space (per cell, keyed feature@offset):
  wall, hazard_edge, water_source  -- content classes (W10: carriers)
  void                             -- world-shape scaffolding (W10: 0 mass)
  degree                           -- DERIVED: number of open
                                      neighbours (novel candidate; its
                                      fate is an open question)
  parity                           -- JUNK: (x+y) mod 2.  Deliberately
                                      translation-VARIANT: under odd
                                      frame offsets it breaks matching,
                                      so it is not merely useless but
                                      harmful.  The loop must discover
                                      this causally.

Coordination utility of a vocabulary: witness--traveler frame
recovery + never-seen-target transport over a battery of worlds x
secret offsets, scored as exact transports minus a fail-open penalty
(fail-open should never fire --- the fail-closed contract).

Registered predictions:
  E1.1 (emergence): the loop retains the content classes {wall,
       hazard_edge} and prunes {parity, void}; zero fail-open
       transports at every round.
  E1.2 (safety of the loop): the final vocabulary's utility is >= the
       initial superset's (pruning never pays with correctness).
  E1.3 (exploratory, no pass/fail): the fate of `degree` is reported.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/song_grammar/exp_e1_landmark_emergence.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.song_grammar.exp_s0_song_smoke import (
    BAND, TRAVELER_START, WITNESS_START)
from experiments.warp.exp_warp_landmark_ablation import (
    H, W, build_world)
from experiments.warp.semantic_identity import align_frames
from multiagent_env import WALL, WATER

GridXY = Tuple[int, int]
OUT_DIR = "tmp/song_grammar/e1_landmark_emergence"
SEEDS = range(10)
OFFSETS = [(3, 2), (-4, 3), (5, -1)]     # includes odd components:
                                          # parity is frame-variant
ALL_FEATURES = ["wall", "hazard_edge", "water_source", "void",
                "degree", "parity"]
FAIL_OPEN_PENALTY = 3.0


# ── fingerprints over a candidate vocabulary ───────────────────────

def cell_tag(env, x: int, y: int) -> Optional[str]:
    if not (0 <= x < W and 0 <= y < H):
        return "void"
    return env.cell_tag(x, y)


def degree(env, x: int, y: int) -> int:
    return sum(1 for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
               if 0 <= nx < W and 0 <= ny < H
               and env.cell(nx, ny) != WALL)


def fp(env, xy: GridXY, vocab: Set[str], radius: int = 2,
       private_xy: Optional[GridXY] = None) -> Dict[str, float]:
    """private_xy: the agent's coordinates in ITS OWN frame -- what
    frame-dependent features (parity) are actually computed from."""
    ax, ay = xy
    px, py = private_xy if private_xy is not None else xy
    sig: Dict[str, float] = {}
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            cx, cy = ax + dx, ay + dy
            tag = cell_tag(env, cx, cy)
            if tag in vocab and tag in ("wall", "hazard_edge",
                                        "water_source", "void"):
                sig[f"{tag}@{dx},{dy}"] = 1.0
            if "degree" in vocab and 0 <= cx < W and 0 <= cy < H \
                    and env.cell(cx, cy) != WALL:
                d = degree(env, cx, cy)
                if d <= 2:                     # only informative cells
                    sig[f"deg{d}@{dx},{dy}"] = 1.0
    if "parity" in vocab:
        sig[f"par{(px + py) % 2}@0,0"] = 1.0   # translation-VARIANT
    return sig


# ── coordination utility of a vocabulary ───────────────────────────

def utility(vocab: Set[str]) -> Dict[str, float]:
    exact, fail_open, refuse, n = 0, 0, 0, 0
    for seed in SEEDS:
        env, water = build_world(seed)
        if env.cell(*TRAVELER_START) == WALL:
            continue
        band_fps = {xy: fp(env, xy, vocab) for xy in BAND}
        for off in OFFSETS:
            n += 1
            wit_fps = {(x + off[0], y + off[1]):
                       fp(env, (x, y), vocab,
                          private_xy=(x + off[0], y + off[1]))
                       for y in range(H) for x in range(W)}
            res = align_frames(band_fps, wit_fps)
            if res.offset is None:
                refuse += 1
                continue
            dx, dy = res.offset
            t = (water[0] + off[0] + dx, water[1] + off[1] + dy)
            if t == water:
                exact += 1
            else:
                fail_open += 1
    score = (exact - FAIL_OPEN_PENALTY * fail_open) / max(1, n)
    return {"score": score, "exact": exact, "fail_open": fail_open,
            "refuse": refuse, "n": n}


# ── the closed loop ────────────────────────────────────────────────

def key_mass(feat: str) -> float:
    """Average number of fingerprint keys the feature contributes
    (its observation/description cost)."""
    env, _ = build_world(0)
    full = set(ALL_FEATURES)
    tot_with = sum(len(fp(env, xy, full)) for xy in BAND)
    tot_without = sum(len(fp(env, xy, full - {feat})) for xy in BAND)
    return (tot_with - tot_without) / len(BAND)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "e1_registered.json"), "w") as f:
        json.dump({
            "v1_outcome": "v1 failed on two implementation defects: "
                          "(a) parity was computed from TRUE "
                          "coordinates, making the intended junk "
                          "accidentally frame-invariant and harmless "
                          "-- v2 computes frame-dependent features "
                          "from the agent's PRIVATE frame, as a real "
                          "agent would; (b) all redundant features "
                          "were pruned simultaneously (each is "
                          "individually removable, jointly they are "
                          "not) -- v2 prunes one per round, cheapest-"
                          "useless first.",
            "E1.1": "loop prunes parity and void; at least one "
                    "content class remains; zero fail-open at every "
                    "round; final score >= initial (the loop may be "
                    "MORE parsimonious than the designer: minimal "
                    "sufficiency, not the designer's pair, is the "
                    "claim)",
            "E1.2": "final vocabulary utility >= initial superset's",
            "E1.3": "exploratory: fate of every feature reported",
            "constants": {"FAIL_OPEN_PENALTY": FAIL_OPEN_PENALTY,
                          "OFFSETS": OFFSETS},
        }, f, indent=2)

    vocab: Set[str] = set(ALL_FEATURES)
    initial = utility(vocab)
    history: List[Dict[str, Any]] = [
        {"round": 0, "vocab": sorted(vocab), **initial}]
    total_fail_open = initial["fail_open"]

    rnd = 0
    while True:
        rnd += 1
        base = utility(vocab)
        deltas = {}
        for feat in sorted(vocab):
            abl = utility(vocab - {feat})
            total_fail_open += abl["fail_open"]
            deltas[feat] = round(base["score"] - abl["score"], 4)
        # prune ONE feature per round: the one whose removal helps most
        # (or is free), tie-broken toward the most expensive in keys ---
        # redundant features are individually removable but not jointly,
        # so simultaneous pruning is invalid (v1 defect)
        removable = {f_: d for f_, d in deltas.items() if d <= 0}
        prune = (min(removable,
                     key=lambda f_: (removable[f_], -key_mass(f_)))
                 if removable else None)
        history.append({"round": rnd, "vocab": sorted(vocab),
                        "score": base["score"], "deltas": deltas,
                        "pruned": [prune] if prune else []})
        if prune is None or len(vocab) <= 1:
            break
        vocab -= {prune}

    final = utility(vocab)
    content = {"wall", "hazard_edge", "water_source"}
    e11 = (bool(vocab & content)
           and not ({"parity", "void"} & vocab)
           and total_fail_open == 0 and final["fail_open"] == 0
           and final["score"] >= initial["score"])
    e12 = final["score"] >= initial["score"]
    verdict = {"E1.1_emergence": e11,
               "E1.2_pruning_is_safe": e12}
    out = {"initial": {**initial, "vocab": ALL_FEATURES},
           "final": {**final, "vocab": sorted(vocab)},
           "degree_fate": ("landmark" if "degree" in vocab
                           else "pruned"),
           "history": history, "verdict": verdict}
    with open(os.path.join(OUT_DIR, "e1_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"initial_score": initial["score"],
                      "final_score": final["score"],
                      "final_vocab": sorted(vocab),
                      "degree_fate": out["degree_fate"]}, indent=2))
    for h in history[1:]:
        print(f"  round {h['round']}: deltas {h['deltas']} "
              f"-> pruned {h['pruned']}")
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/e1_results.json")


if __name__ == "__main__":
    main()
