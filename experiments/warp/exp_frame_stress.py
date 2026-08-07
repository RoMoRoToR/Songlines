"""F — stress surface for private-frame correspondence + tag misalignment.

W7/W9 established the clean-world contract of the semantic place matcher
(experiments/warp/semantic_identity.py): exact SE(2) = Z^2 x C4 recovery
on landmark worlds, fail-closed refusal on aliased/symmetric ones
(80/80 exact in W9-P2).  This experiment replaces that unit-test picture
with a STATISTICAL SURFACE: where does matching start to break, and —
critically — does it break CLOSED (refusal, a coverage cost) or OPEN
(a confidently wrong transform, the poisoning failure)?

Scenarios (each swept over levels x seeds x 4 SE(2) transforms):

  1. noise      — per-observation FN/FP noise on landmark tags of the
                  SENDER (foreign) side: FN drops a salient tag from a
                  window cell, FP paints a random content tag onto a
                  neutral cell.  Curves of exact / false-match /
                  false-reject / ambiguity-detection vs noise level.
  2. missing    — the sender permanently lacks k of the world's content
                  landmarks (never perceives them).  Same curves vs k.
  3. repeated   — constructed worlds whose only content is a motif
                  stamped at C >= 2 interior locations (identical
                  radius-2 constellations): the transform is
                  unrecoverable in principle and the matcher MUST fail
                  closed.  A '+unique' variant adds one extra landmark
                  near copy 1, restoring recoverability.
  4. misalign   — heterogeneous tag vocabulary on the sender side:
                  * perm         — silent permutation (derangement) of
                                   the content-tag names;
                  * syn:rho      — sender uses synonyms; the receiver
                                   holds a translator dictionary that
                                   covers a fraction rho of the
                                   vocabulary (rho=1 full, rho=0 none);
                  * concept-     — sender's vocabulary lacks whole tag
                                   classes (those cells look neutral);
                  * coarse       — sender merges wall+hazard_edge into
                                   one tag 'obstacle' (one-to-many);
                                   arms: receiver maps obstacle ->
                                   hazard_edge by fiat / no translator /
                                   receiver coarsens its own side too.

Outcome classification per trial (truth transform known by design):

  exact        — aligner returns exactly the ground-truth (r*, delta*);
  false_match  — aligner ACCEPTS a wrong transform (fail-open; the
                 critical failure — foreign evidence would be
                 transported to wrong cells);
  reject       — aligner refuses (fail-closed).  On recoverable levels
                 this is a FALSE REJECT (coverage cost); on
                 unrecoverable levels (repeated pure) it is CORRECT.
  On unrecoverable levels an 'exact' return is an UNWARRANTED accept
  (the evidence cannot distinguish it from a wrong transform) and is
  reported in the fail-open column.

  amb_signal   — the refusal/acceptance is accompanied by explicit
                 ambiguity evidence: >= 2 rotation hypotheses reached
                 consensus (n_winners >= 2) or the true-rotation
                 translation aligner counted ambiguous candidates.

The matcher itself is imported UNCHANGED from semantic_identity.py;
this file only builds worlds, observations and perturbations.
Note: the 'void' tag (world border) is produced by fingerprint() itself
and is therefore implicitly shared vocabulary on both sides.

Usage (smoke)::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_frame_stress.py \\
        --seeds 5 --out tmp/frame_stress_smoke

Full sweep (cluster, see cluster/submit_frame_stress.sh)::

    ... exp_frame_stress.py --full --seeds 20 --out tmp/cluster/warp/frame_stress_full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
from zlib import crc32

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.warp.semantic_identity import (
    SALIENT_TAGS, align_frames, align_frames_se2, fingerprint,
    rotate_point, rotate_sig,
)

GridXY = Tuple[int, int]

CONTENT_TAGS = ["hazard_edge", "wall", "water_source", "goal"]
CONTENT_SET = set(CONTENT_TAGS)
CONTENT_COUNTS = {"hazard_edge": 12, "wall": 6, "water_source": 2, "goal": 1}
SYNONYM = {"hazard_edge": "danger_rim", "wall": "barrier",
           "water_source": "spring", "goal": "target"}
# one fixed offset per rotation class — the W9 pattern (rotations x offsets)
TRANSFORMS: List[Tuple[int, GridXY]] = [
    (0, (3, 2)), (1, (-4, 3)), (2, (5, -2)), (3, (-3, -4))]
DEF_W, DEF_H = 14, 12


# ───────────────────────────── worlds ─────────────────────────────


def make_world(W: int, H: int, rng) -> Dict[GridXY, str]:
    """Random landmark world: content tags scattered on a neutral grid."""
    world: Dict[GridXY, str] = {}
    for tag, n in CONTENT_COUNTS.items():
        placed = 0
        while placed < n:
            xy = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            if xy not in world:
                world[xy] = tag
                placed += 1
    return world


def make_repeated_world(W: int, H: int, rng, copies: int, unique: bool,
                        interior: bool = True
                        ) -> Tuple[Dict[GridXY, str], List[GridXY], GridXY]:
    """World whose ONLY content is one hazard motif stamped at `copies`
    anchors >= 7 apart in x (no observation window sees two motifs).

    interior=True places anchors so that EVERY informative fingerprint
    (anything within radius 2 of a motif cell) has a fully in-grid
    window on every copy — no 'void' keys anywhere near any motif.  The
    copies' constellations are then pixel-identical and the frame is
    unrecoverable in principle, unless `unique` adds a goal landmark
    next to copy 1.  Requires W >= 11 + 7*(copies-1).

    interior=False (the '@border' variant) anchors copy 1 two cells
    from the left border: the world LOOKS aliased (identical motifs)
    but the border 'void' context differs between copies and acts as an
    implicit landmark — recoverability via world-shape, not content."""
    x0 = 4 if interior else 2
    y0 = 4 if interior else 2
    anchors = [(x0 + 7 * i, y0) for i in range(copies)]
    lim = W - 7 if interior else W - 5
    if anchors[-1][0] > lim:
        raise ValueError(f"grid W={W} too narrow for {copies} copies "
                         f"(interior={interior})")
    # motif: 4 distinct cells in the 3x3 patch (at least 2 per window)
    offs = [(dx, dy) for dx in range(3) for dy in range(3)]
    idx = rng.choice(len(offs), size=4, replace=False)
    motif = [offs[i] for i in idx]
    world: Dict[GridXY, str] = {}
    for ax, ay in anchors:
        for dx, dy in motif:
            world[(ax + dx, ay + dy)] = "hazard_edge"
    if unique:
        world[(anchors[0][0] + 1, anchors[0][1] + 4)] = "goal"
    return world, anchors, anchors[0]


def own_band(W: int, H: int) -> List[GridXY]:
    """Receiver's visited region (interior band, W7 convention)."""
    return [(x, y) for y in range(2, H - 2)
            for x in range(2, min(9, W - 4))]


def full_sweep(W: int, H: int) -> List[GridXY]:
    return [(x, y) for y in range(H) for x in range(W)]


# ───────────────────────────── observation ────────────────────────


def observe_fps(world: Dict[GridXY, str], W: int, H: int,
                visits: List[GridXY], r: int, offset: GridXY, *,
                salient: Optional[Set[str]] = None,
                vocab_map: Optional[Dict[str, str]] = None,
                fn: float = 0.0, fp: float = 0.0,
                rng=None, radius: int = 2) -> Dict[GridXY, Dict[str, float]]:
    """Fingerprints of `visits` (true-frame cells) expressed in a private
    frame (rotation r, then translation `offset`).

    Noise is applied per (visit, cell) draw — each observation window is
    independently corrupted: FN drops a content tag (-> neutral), FP
    paints a random content tag onto a neutral cell.  `vocab_map`
    renames tags AFTER noise (the sensor sees the class, the vocabulary
    names it); mapping to 'safe_neutral' models a missing concept."""
    fps: Dict[GridXY, Dict[str, float]] = {}
    for (x, y) in visits:
        cells: List[Dict[str, Any]] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                wx, wy = x + dx, y + dy
                if not (0 <= wx < W and 0 <= wy < H):
                    continue  # absent cell -> 'void' inside fingerprint()
                tag = world.get((wx, wy), "safe_neutral")
                if rng is not None:
                    if tag in CONTENT_SET and fn > 0 \
                            and rng.random() < fn:
                        tag = "safe_neutral"
                    elif tag == "safe_neutral" and fp > 0 \
                            and rng.random() < fp:
                        tag = CONTENT_TAGS[int(
                            rng.integers(len(CONTENT_TAGS)))]
                if vocab_map:
                    tag = vocab_map.get(tag, tag)
                px, py = rotate_point((wx, wy), r)
                cells.append({"xy": (px + offset[0], py + offset[1]),
                              "tag": tag})
        prx, pry = rotate_point((x, y), r)
        pxy = (prx + offset[0], pry + offset[1])
        fps[pxy] = fingerprint(pxy, cells, radius=radius, salient=salient)
    return fps


def translate_fps(fps: Dict[GridXY, Dict[str, float]],
                  tag_map: Dict[str, str]) -> Dict[GridXY, Dict[str, float]]:
    """Receiver-side translator: rename the tag part of fingerprint keys."""
    out: Dict[GridXY, Dict[str, float]] = {}
    for xy, sig in fps.items():
        ns: Dict[str, float] = {}
        for key, v in sig.items():
            tag, off = key.rsplit("@", 1)
            ns[f"{tag_map.get(tag, tag)}@{off}"] = v
        out[xy] = ns
    return out


def derangement(rng, items: List[str]) -> Dict[str, str]:
    while True:
        perm = rng.permutation(len(items))
        if all(perm[i] != i for i in range(len(items))):
            return {items[i]: items[int(perm[i])]
                    for i in range(len(items))}


# ───────────────────────────── one trial ──────────────────────────


def run_trial(level: Dict[str, Any], seed: int, t_idx: int,
              W: int, H: int) -> Dict[str, Any]:
    r_w, o_w = TRANSFORMS[t_idx]
    W = level.get("W", W)
    H = level.get("H", H)
    rng = np.random.default_rng(
        [20260807, crc32(level["name"].encode()) & 0xFFFF, seed, t_idx])
    scen = level["scenario"]
    recoverable = True

    if scen == "repeated":
        world, anchors, a1 = make_repeated_world(
            W, H, rng, level["copies"], level["unique"],
            interior=level.get("interior", True))
        # interior pure-repeated worlds are unrecoverable in principle;
        # '+unique' restores content landmarks, '@border' restores
        # recoverability through void (world border) context
        recoverable = level["unique"] or not level.get("interior", True)
        own_visits = [(a1[0] + dx, a1[1] + dy)
                      for dx in range(-2, 5) for dy in range(-2, 5)]
    else:
        world = make_world(W, H, rng)
        own_visits = own_band(W, H)
    foreign_visits = full_sweep(W, H)

    own_fps = observe_fps(world, W, H, own_visits, 0, (0, 0))

    fn = fp = 0.0
    vocab_map: Optional[Dict[str, str]] = None
    salient: Optional[Set[str]] = None
    receiver_map: Optional[Dict[str, str]] = None
    foreign_world = world

    if scen == "noise":
        fn, fp = level["fn"], level["fp"]
    elif scen == "missing":
        content = [xy for xy, t in world.items()]
        k = min(level["k"], len(content))
        drop_idx = rng.choice(len(content), size=k, replace=False)
        drop = {content[i] for i in drop_idx}
        foreign_world = {xy: t for xy, t in world.items()
                         if xy not in drop}
    elif scen == "misalign":
        kind = level["kind"]
        if kind == "perm":
            vocab_map = derangement(rng, CONTENT_TAGS)
        elif kind == "syn":
            vocab_map = dict(SYNONYM)
            salient = set(SYNONYM.values()) | {"void"}
            n_cov = int(round(level["rho"] * len(SYNONYM)))
            order = rng.permutation(len(CONTENT_TAGS))
            covered = [CONTENT_TAGS[int(i)] for i in order[:n_cov]]
            receiver_map = {SYNONYM[a]: a for a in covered}
        elif kind == "concept":
            vocab_map = {c: "safe_neutral" for c in level["drop"]}
        elif kind == "coarse":
            vocab_map = {"wall": "obstacle", "hazard_edge": "obstacle"}
            salient = {"obstacle", "water_source", "goal", "void"}
            if level["arm"] == "fiat":
                receiver_map = {"obstacle": "hazard_edge"}
            elif level["arm"] == "both":
                receiver_map = {"obstacle": "hazard_edge"}
                own_fps = translate_fps(own_fps,
                                        {"wall": "hazard_edge"})
            # arm == 'none': obstacle keys stay untranslated

    foreign_fps = observe_fps(foreign_world, W, H, foreign_visits,
                              r_w, o_w, salient=salient,
                              vocab_map=vocab_map, fn=fn, fp=fp, rng=rng)
    if receiver_map:
        foreign_fps = translate_fps(foreign_fps, receiver_map)

    res = align_frames_se2(own_fps, foreign_fps)

    r_star = (4 - r_w) % 4
    ro = rotate_point(o_w, r_star)
    truth = (r_star, (-ro[0], -ro[1]))
    if res.rotation is not None:
        got = (res.rotation, tuple(res.delta))
        outcome = "exact" if got == truth else "false_match"
    else:
        outcome = "reject"

    # ambiguity diagnostics at the TRUE rotation
    rot_f = {rotate_point(xy, r_star): rotate_sig(sig, r_star)
             for xy, sig in foreign_fps.items()}
    tres = align_frames(own_fps, rot_f)
    amb_signal = (res.n_winners >= 2) or (tres.n_ambiguous > 0)

    return {"scenario": scen, "level": level["name"], "seed": seed,
            "r_w": r_w, "o_w": list(o_w), "outcome": outcome,
            "recoverable": recoverable,
            "recovered": ([res.rotation, list(res.delta)]
                          if res.rotation is not None else None),
            "n_winners": res.n_winners, "amb_signal": bool(amb_signal),
            "true_rot_matches": tres.n_matches,
            "true_rot_ambiguous": tres.n_ambiguous}


# ───────────────────────────── level grids ────────────────────────


def build_levels(full: bool) -> List[Dict[str, Any]]:
    lv: List[Dict[str, Any]] = []
    # 1. noise
    fns = ([round(x, 3) for x in np.arange(0.0, 0.601, 0.05)]
           if full else [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    fps_ = ([round(x, 3) for x in np.arange(0.025, 0.301, 0.025)]
            if full else [0.02, 0.05, 0.10, 0.15, 0.20])
    for v in fns:
        lv.append({"scenario": "noise", "name": f"fn={v:.3f}",
                   "fn": v, "fp": 0.0})
    for v in fps_:
        lv.append({"scenario": "noise", "name": f"fp={v:.3f}",
                   "fn": 0.0, "fp": v})
    joint = ([(a, b) for a in (0.05, 0.10, 0.20, 0.30)
              for b in (0.025, 0.05, 0.10, 0.15)]
             if full else [(0.10, 0.05), (0.20, 0.10)])
    for a, b in joint:
        lv.append({"scenario": "noise", "name": f"fn={a:.2f}+fp={b:.3f}",
                   "fn": a, "fp": b})
    # 2. missing landmarks (of 21 content cells)
    ks = list(range(1, 17)) if full else [1, 2, 4, 8, 12]
    for k in ks:
        lv.append({"scenario": "missing", "name": f"k={k}", "k": k})
    # 3. repeated constellations (grids fixed: anchors need width;
    #    interior needs W >= 11 + 7*(C-1))
    reps = [(2, False, True, 18, 12), (2, True, True, 18, 12),
            (2, False, False, 14, 12)]
    if full:
        reps += [(3, False, True, 25, 12), (3, True, True, 25, 12),
                 (4, False, True, 32, 12)]
    for c, uniq, inter, w, h in reps:
        name = (f"C={c}" + ("+unique" if uniq else "")
                + ("" if inter else "@border"))
        lv.append({"scenario": "repeated", "name": name, "copies": c,
                   "unique": uniq, "interior": inter, "W": w, "H": h})
    # 4. tag misalignment
    n_perm = 5 if full else 2
    for i in range(n_perm):
        lv.append({"scenario": "misalign", "name": f"perm#{i}",
                   "kind": "perm", "variant": i})
    for rho in [1.0, 0.75, 0.5, 0.25, 0.0]:
        lv.append({"scenario": "misalign", "name": f"syn:rho={rho:.2f}",
                   "kind": "syn", "rho": rho})
    for drop in (["hazard_edge"], ["wall"], ["hazard_edge", "wall"]):
        lv.append({"scenario": "misalign",
                   "name": f"concept-{'+'.join(drop)}",
                   "kind": "concept", "drop": drop})
    for arm in ("fiat", "none", "both"):
        lv.append({"scenario": "misalign", "name": f"coarse:{arm}",
                   "kind": "coarse", "arm": arm})
    return lv


# ───────────────────────────── aggregation ────────────────────────


def summarize(rows: List[Dict[str, Any]],
              levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = {l["name"]: i for i, l in enumerate(levels)}
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["scenario"], r["level"]), []).append(r)
    out = []
    for (sc, name), rs in sorted(by.items(),
                                 key=lambda kv: order[kv[0][1]]):
        n = len(rs)
        recov = rs[0]["recoverable"]
        exact = sum(r["outcome"] == "exact" for r in rs)
        fm = sum(r["outcome"] == "false_match" for r in rs)
        rej = sum(r["outcome"] == "reject" for r in rs)
        # on unrecoverable levels an 'exact' return is unwarranted:
        # count it into fail-open
        fail_open = fm + (0 if recov else exact)
        false_rej = rej if recov else 0
        amb = sum(r["amb_signal"] for r in rs)
        out.append({"scenario": sc, "level": name, "n": n,
                    "recoverable": recov,
                    "exact": round(exact / n, 3),
                    "false_match": round(fail_open / n, 3),
                    "false_reject": round(false_rej / n, 3),
                    "ambiguity_detected": round(amb / n, 3)})
    return out


def print_table(summary: List[Dict[str, Any]]) -> None:
    cur = None
    for s in summary:
        if s["scenario"] != cur:
            cur = s["scenario"]
            print(f"\n=== {cur} ===")
            print(f"{'level':>24} | {'n':>4} | {'exact':>6} "
                  f"{'FALSE-MATCH':>11} {'false-rej':>9} {'ambig-det':>9}")
        print(f"{s['level']:>24} | {s['n']:>4} | {s['exact']:>6.2f} "
              f"{s['false_match']:>11.2f} {s['false_reject']:>9.2f} "
              f"{s['ambiguity_detected']:>9.2f}"
              + ("" if s["recoverable"] else "   [unrecoverable: "
                 "reject=correct]"))


# ───────────────────────────── main ───────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--W", type=int, default=DEF_W)
    ap.add_argument("--H", type=int, default=DEF_H)
    ap.add_argument("--full", action="store_true",
                    help="dense level grids (cluster run)")
    ap.add_argument("--scenario", default="all",
                    choices=["all", "noise", "missing", "repeated",
                             "misalign"])
    ap.add_argument("--out", default="tmp/frame_stress_smoke")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    levels = build_levels(a.full)
    if a.scenario != "all":
        levels = [l for l in levels if l["scenario"] == a.scenario]

    rows: List[Dict[str, Any]] = []
    for lvl in levels:
        for seed in range(a.seeds):
            for t_idx in range(len(TRANSFORMS)):
                rows.append(run_trial(lvl, seed, t_idx, a.W, a.H))
        done = [r for r in rows if r["level"] == lvl["name"]]
        fm = sum(r["outcome"] == "false_match" for r in done)
        print(f"[{lvl['scenario']:>8}] {lvl['name']:<24} "
              f"{len(done)} trials, false-match={fm}", flush=True)

    summary = summarize(rows, levels)
    with open(os.path.join(a.out, "frame_stress_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(a.out, "frame_stress_summary.json"), "w") as f:
        json.dump({"seeds": a.seeds, "W": a.W, "H": a.H, "full": a.full,
                   "transforms": TRANSFORMS, "summary": summary},
                  f, indent=1)

    print_table(summary)
    total_fm = sum(1 for r in rows if r["outcome"] == "false_match")
    unwarr = sum(1 for r in rows if r["outcome"] == "exact"
                 and not r["recoverable"])
    print(f"\nTOTAL trials={len(rows)}  wrong-transform accepts="
          f"{total_fm}  unwarranted accepts (ambiguous worlds)={unwarr}")
    print(f"Saved: {a.out}/frame_stress_summary.json")


if __name__ == "__main__":
    main()
