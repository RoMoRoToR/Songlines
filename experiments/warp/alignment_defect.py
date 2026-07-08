"""
Adjunction (round-trip) defect of frame recovery -- the categorical indicator.

The coordinate-anchor-removal result (semantic place identity) recovers the
inter-agent frame with align_frames / align_frames_se2. This script measures the
*categorical* quantity that recovery corresponds to: the adjunction round-trip
defect eta_X : X -> G(F(X)).

  F = align_frames(A, B)  recovers B's frame in A's coordinates;
  G = align_frames(B, A)  recovers A's frame in B's coordinates.
If the two recovered functors are genuinely adjoint (frame recovered), the
round trip is the identity:
  translation:  || F.offset + G.offset ||_1  = 0
  SE(2):        rotation r_F + r_G = 0 (mod 4)  AND  || R_{r_G} F.delta + G.delta ||_1 = 0
When recovery fails closed (featureless / symmetric world) F or G is None: no
adjoint exists -- the defect is undefined (reported as "no adjunction").

Levels (forwarded theorem):  defect = 0 -> Level 3 (strict adjoint);
0 < defect < inf -> Level 2 (lax, bounded);  None/inf -> Level 1 (no gluing).

Deterministic. Reuses experiments/warp/{semantic_identity, exp_warp_semantic_identity}.
Run: PYTHONPATH=. .venv/bin/python experiments/warp/alignment_defect.py
"""
from __future__ import annotations
import os, sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from experiments.warp.exp_warp_semantic_identity import build_world, W, H, TRAVELER_BAND
from experiments.warp.semantic_identity import (
    align_frames, align_frames_se2, fingerprint, rotate_point,
)

GridXY = Tuple[int, int]


def _sweep_true(env, aid: str, cells_xy=None):
    """One pass over cells_xy (default: whole grid) in the TRUE frame."""
    ag = env.agents[aid]
    if cells_xy is None:
        cells_xy = [(x, y) for y in range(H) for x in range(W)]
    out = []
    for (x, y) in cells_xy:
        ag.x, ag.y = x, y
        obs = env._observation(aid)
        cells = [{"xy": (int(c["xy"][0]), int(c["xy"][1])), "tag": c["tag"]}
                 for c in obs.get("cells", [])]
        out.append(((x, y), cells))
    return out


def _fps_in_frame(sweep, r: int, off: GridXY) -> Dict[GridXY, Dict[str, float]]:
    """Fingerprints as an agent with private frame (rotation r, offset off) sees them."""
    fps: Dict[GridXY, Dict[str, float]] = {}
    for (tx, ty), cells in sweep:
        rp = rotate_point((tx, ty), r)
        pxy = (rp[0] + off[0], rp[1] + off[1])
        pcells = []
        for c in cells:
            rc = rotate_point((int(c["xy"][0]), int(c["xy"][1])), r)
            pcells.append({"xy": (rc[0] + off[0], rc[1] + off[1]), "tag": c["tag"]})
        fps[pxy] = fingerprint(pxy, pcells)
    return fps


def _probe(env, cells_xy=None):
    if "probe" not in env.agents:
        env.spawn("probe", start_xy=(0, 0), target_tag="water_source")
    return _sweep_true(env, "probe", cells_xy)


def translation_defect(hazard_density, offset_b, seed):
    env, _ = build_world(hazard_density, seed)
    sweep = _probe(env)
    fps_a = _fps_in_frame(sweep, 0, (0, 0))          # agent A: identity frame
    fps_b = _fps_in_frame(sweep, 0, offset_b)        # agent B: secret translation
    F = align_frames(fps_a, fps_b)                   # B -> A
    G = align_frames(fps_b, fps_a)                   # A -> B
    if F.offset is None or G.offset is None:
        return None                                  # fails closed: no adjunction
    return abs(F.offset[0] + G.offset[0]) + abs(F.offset[1] + G.offset[1])


def partial_defect(offset_b, seed):
    """Agent B observes only a featureless interior band (no water, no hazards):
    its informative fingerprints are insufficient -> align returns None ->
    no adjoint gluing exists (Level 1, fails closed)."""
    env, _ = build_world(0.0, seed)                  # featureless interior
    sweep_full = _probe(env)
    sweep_band = _probe(env, TRAVELER_BAND)
    fps_a = _fps_in_frame(sweep_full, 0, (0, 0))
    fps_b = _fps_in_frame(sweep_band, 0, offset_b)
    F = align_frames(fps_a, fps_b)
    G = align_frames(fps_b, fps_a)
    if F.offset is None or G.offset is None:
        return None
    return abs(F.offset[0] + G.offset[0]) + abs(F.offset[1] + G.offset[1])


def se2_defect(hazard_density, rot_b, offset_b, seed):
    env, _ = build_world(hazard_density, seed)
    sweep = _probe(env)
    fps_a = _fps_in_frame(sweep, 0, (0, 0))
    fps_b = _fps_in_frame(sweep, rot_b, offset_b)
    F = align_frames_se2(fps_a, fps_b)
    G = align_frames_se2(fps_b, fps_a)
    if F.rotation is None or G.rotation is None:
        return None
    rot_err = (F.rotation + G.rotation) % 4
    rot_err = min(rot_err, 4 - rot_err)              # 90-deg units, 0..2
    rd = rotate_point(F.delta, G.rotation)
    trans_err = abs(rd[0] + G.delta[0]) + abs(rd[1] + G.delta[1])
    return rot_err, trans_err


def main():
    OFFSETS = [(3, 2), (-4, 3), (6, -2)]
    ROTS = [0, 1, 2, 3]
    SEEDS = range(10)

    print("Adjunction round-trip defect of frame recovery (real warp align_frames)\n")

    # ── translation regime ──
    print("=== Translation frames ===")
    print(f"{'hazard':>7} {'regime':>14} | {'mean defect':>11} {'=0 (Level 3)':>13} {'no-adj (L1)':>12}")
    for h, name in [(0.10, "content-rich"), (0.0, "featureless")]:
        defects = [translation_defect(h, off, s) for off in OFFSETS for s in SEEDS]
        vals = [d for d in defects if d is not None]
        none = sum(1 for d in defects if d is None)
        mean = np.mean(vals) if vals else float("nan")
        zero = sum(1 for d in vals if d == 0)
        print(f"{h:>7.2f} {name:>14} | {mean:>11.3f} {zero:>7}/{len(vals):<5} {none:>7}/{len(defects):<4}")

    # ── fail-closed regime: partial featureless observation ──
    print("\n=== Fail-closed: agent B sees only a featureless interior band ===")
    res = [partial_defect(off, s) for off in OFFSETS for s in SEEDS]
    none = sum(1 for d in res if d is None)
    vals = [d for d in res if d is not None]
    print(f"  no adjunction (F or G = None, Level 1): {none}/{len(res)}"
          f"   |  spurious gluings (defect>0): {sum(1 for d in vals if d>0)}/{len(res)}")

    # ── SE(2) regime ──
    print("\n=== SE(2) frames (rotation + translation), content-rich (h=0.10) ===")
    print(f"{'rot_b':>5} | {'mean rot-err':>12} {'mean trans-err':>14} {'no-adj (L1)':>12}")
    for r in ROTS:
        res = [se2_defect(0.10, r, off, s) for off in OFFSETS for s in SEEDS]
        ok = [x for x in res if x is not None]
        none = sum(1 for x in res if x is None)
        rot = np.mean([x[0] for x in ok]) if ok else float("nan")
        tr = np.mean([x[1] for x in ok]) if ok else float("nan")
        print(f"{r:>5} | {rot:>12.3f} {tr:>14.3f} {none:>7}/{len(res):<4}")

    print("\nReading: content-rich -> defect exactly 0 (strict adjoint / Level 3);")
    print("featureless -> F or G is None (no adjunction / Level 1, fails closed).")
    print("The defect is the categorical value frame recovery computes -- zero iff")
    print("the round trip G(F(X)) returns X, i.e. the frames are genuinely glued.")


if __name__ == "__main__":
    main()
