"""S0 — song smoke: the first MEASURED fact of the Song Grammar frontier.

Everything in docs/FRONTIER_SONG_GRAMMAR_2026-07-25.md was, until this
experiment, closed only on paper.  S0 implements the minimal hand-written
song grammar and measures the Phase-0 core in miniature (witness/traveler
instead of the full N>T deficit bench):

  A song is a purely RELATIONAL object: a sequence of couplets
  v = (sigma, e), where sigma is a landmark signature (the local
  semantic constellation -- W7 fingerprint) and e is the beat descriptor
  from the previous couplet (exact displacement vector, frame-invariant
  under translation).  A song carries NO global coordinates at all.

Arms (what the witness broadcasts):
  (a) sigma+beats -- full song; traveler matches couplets by signature
      (mutual-unique within the song), checks METRIC CONSISTENCY between
      matched couplets against the beat chain (the loop-closure check),
      and dead-reckons from the LAST matched couplet to the final
      (never-seen) water couplet.  Refuses without an anchor or on an
      inconsistent chain: fail closed.
  (b) beats-only -- "n steps from the previous GOOD": the beat chain
      with no signatures.  Usable only by ASSUMING the anchor (the
      traveler grafts the chain onto its own start).  Run in two
      conditions: co-anchored (witness starts where the traveler
      starts -- the shared-GOOD assumption holds) and anchor-broken
      (witness starts elsewhere).
  (g) signatures-only -- couplets without beats: can re-identify
      co-visited places but has no edges to walk to a never-seen
      target: transports nothing by construction.
  (v) full snapshots -- the W7 machinery (all fingerprints + evidence
      in the sender's private frame, semantic alignment): the
      information upper bound, at full bit cost.

Bit codec (registered BEFORE runs; one code table for all arms):
  tag id 2 bits (|{wall,hazard,water,void}|=4); offset-in-window 4 bits
  (13 cells at radius 2); fingerprint key = 6 bits; signature length
  field 4 bits; beat = 10 bits (dx,dy each 5 bits signed); coordinate =
  8 bits (4+4 for a 14x12 grid).
  couplet = 4 + 6*n_keys + 10;  snapshot cell = 8 + 4 + 6*n_keys;
  snapshot evidence = 8.

Registered predictions (written to disk before episodes):
  S0.1 (beats do not carry identity): arm (b) anchor-broken locks in
       100% of runs and is NEVER on the true water (fail open, mislock
       offset = witness_start - traveler_start); co-anchored control is
       100% correct.  Beats parasitise on anchor identity; they do not
       supply it.
  S0.2 (songs are fail-safe transport): arm (a) has ZERO fail-open
       locks; every transported lock is the true water; coverage >= 0.8
       of seeds (refusals, not errors, in the rest).
  S0.3 (a fraction of the bits): mean bits(a) <= 0.35 * mean bits(v)
       with equal per-transport correctness.
  S0.4 (nodes alone cannot transport the unseen): arm (g) makes zero
       target transports.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/song_grammar/exp_s0_song_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.warp.exp_warp_landmark_ablation import (
    W, H, build_world, cells_around)
from experiments.warp.semantic_identity import (
    align_frames, cosine, fingerprint)
from multiagent_env import WALL

GridXY = Tuple[int, int]
OUT_DIR = "tmp/song_grammar/s0_song_smoke"
SEEDS = range(12)
TRAVELER_START: GridXY = (2, 2)
WITNESS_START: GridXY = (0, 0)          # anchor-broken condition
BAND = [(x, y) for y in range(2, 10) for x in range(2, 8)]
BAND_WAYPOINT: GridXY = (5, 6)          # witness route passes the band
GAP_MIN = 2                             # min steps between couplets
SIM = 0.999

# ── bit codec (registered) ─────────────────────────────────────────
KEY_BITS = 6      # tag id (2) + offset-in-window (4)
LEN_BITS = 4
BEAT_BITS = 10    # dx, dy: 5 signed bits each
COORD_BITS = 8    # 4 + 4 for a 14x12 grid


def bits_song(couplets: List[Dict], with_sigs: bool, with_beats: bool) -> int:
    total = 0
    for c in couplets:
        if with_sigs:
            total += LEN_BITS + KEY_BITS * len(c["sig"])
        if with_beats:
            total += BEAT_BITS
    return total


def bits_snapshot(fps: Dict[GridXY, Dict[str, float]]) -> int:
    total = COORD_BITS  # evidence (water) coordinate
    for sig in fps.values():
        total += COORD_BITS + LEN_BITS + KEY_BITS * len(sig)
    return total


# ── world helpers ──────────────────────────────────────────────────

def bfs_path(env, start: GridXY, goal: GridXY) -> Optional[List[GridXY]]:
    if env.cell(*goal) == WALL or env.cell(*start) == WALL:
        return None
    prev: Dict[GridXY, Optional[GridXY]] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = [cur]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1]
        x, y = cur
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (0 <= nxt[0] < W and 0 <= nxt[1] < H
                    and nxt not in prev and env.cell(*nxt) != WALL):
                prev[nxt] = cur
                q.append(nxt)
    return None


def informative(sig: Dict[str, float]) -> bool:
    return len(sig) >= 2 and any(not k.startswith("void") for k in sig)


def fp_at(env, xy: GridXY) -> Dict[str, float]:
    return fingerprint(xy, cells_around(env, *xy))


def build_song(env, path: List[GridXY]) -> List[Dict[str, Any]]:
    """Hand grammar: a couplet where the constellation is informative
    (>= GAP_MIN steps since the last), always closing on the target.
    The couplet keeps NO coordinates: only (sig, beat_from_prev)."""
    couplets: List[Dict[str, Any]] = []
    last_xy = path[0]
    last_idx = -10
    for i, xy in enumerate(path):
        is_last = i == len(path) - 1
        sig = fp_at(env, xy)
        if not is_last and (not informative(sig) or i - last_idx < GAP_MIN):
            continue
        couplets.append({
            "sig": sig,
            "beat": (xy[0] - last_xy[0], xy[1] - last_xy[1]),
            "is_target": is_last,
        })
        last_xy, last_idx = xy, i
    return couplets


# ── arms ───────────────────────────────────────────────────────────

def arm_song(couplets: List[Dict], band_fps: Dict[GridXY, Dict[str, float]]
             ) -> Dict[str, Any]:
    """(a) sigma+beats: mutual-unique couplet matching + beat-chain
    consistency (loop closure) + dead reckoning to the target."""
    own_rich = {xy: s for xy, s in band_fps.items() if informative(s)}
    matches: List[Tuple[int, GridXY]] = []
    for j, c in enumerate(couplets[:-1]):        # target never matchable
        if not informative(c["sig"]):
            continue
        cands = [xy for xy, s in own_rich.items()
                 if cosine(c["sig"], s) >= SIM]
        if len(cands) != 1:
            continue
        back = [k for k, c2 in enumerate(couplets[:-1])
                if cosine(own_rich[cands[0]], c2["sig"]) >= SIM]
        if back == [j]:
            matches.append((j, cands[0]))

    if not matches:
        return {"transported": None, "reason": "no_anchor",
                "n_matches": 0}

    # loop closure: displacement between matched couplets must equal
    # the beat-chain sum between them
    for (j1, p1), (j2, p2) in zip(matches, matches[1:]):
        chain = np.sum([couplets[k]["beat"]
                        for k in range(j1 + 1, j2 + 1)], axis=0)
        if (p2[0] - p1[0], p2[1] - p1[1]) != (int(chain[0]), int(chain[1])):
            return {"transported": None, "reason": "inconsistent_chain",
                    "n_matches": len(matches)}

    j_last, p_last = matches[-1]
    tail = np.sum([couplets[k]["beat"]
                   for k in range(j_last + 1, len(couplets))], axis=0)
    est = (p_last[0] + int(tail[0]), p_last[1] + int(tail[1]))
    return {"transported": est, "reason": "ok", "n_matches": len(matches)}


def arm_beats(couplets: List[Dict], anchor_assumed: GridXY
              ) -> Dict[str, Any]:
    """(b) beats-only: graft the whole chain onto the assumed anchor."""
    chain = np.sum([c["beat"] for c in couplets], axis=0)
    est = (anchor_assumed[0] + int(chain[0]),
           anchor_assumed[1] + int(chain[1]))
    return {"transported": est, "reason": "dead_reckoning"}


def arm_sigs_only(couplets: List[Dict],
                  band_fps: Dict[GridXY, Dict[str, float]]
                  ) -> Dict[str, Any]:
    """(g) signatures-only: the target couplet can only be transported
    if the traveler itself has a matching fingerprint -- it never saw
    the water, so transport is impossible by construction."""
    target_sig = couplets[-1]["sig"]
    cands = [xy for xy, s in band_fps.items()
             if informative(s) and cosine(target_sig, s) >= SIM]
    if len(cands) == 1:
        return {"transported": cands[0], "reason": "direct_match"}
    return {"transported": None, "reason": "target_never_seen"}


def arm_snapshot(witness_fps: Dict[GridXY, Dict[str, float]],
                 water_private: GridXY,
                 band_fps: Dict[GridXY, Dict[str, float]]
                 ) -> Dict[str, Any]:
    """(v) full snapshots through the W7 aligner (upper bound)."""
    res = align_frames(band_fps, witness_fps)
    if res.offset is None:
        return {"transported": None, "reason": "no_consensus"}
    dx, dy = res.offset
    return {"transported": (water_private[0] + dx, water_private[1] + dy),
            "reason": "aligned"}


# ── episode ────────────────────────────────────────────────────────

def run_seed(seed: int, offset: GridXY) -> Optional[Dict[str, Any]]:
    env, water = build_world(seed)
    leg1 = bfs_path(env, WITNESS_START, BAND_WAYPOINT)
    leg2 = bfs_path(env, BAND_WAYPOINT, water)
    if leg1 is None or leg2 is None:
        return None  # walled-off world; logged, not silently dropped
    path = leg1 + leg2[1:]
    song = build_song(env, path)

    # co-anchored variant of the witness route (for the S0.1 control)
    leg1c = bfs_path(env, TRAVELER_START, BAND_WAYPOINT)
    path_co = (leg1c + leg2[1:]) if leg1c else None
    song_co = build_song(env, path_co) if path_co else None

    band_fps = {xy: fp_at(env, xy) for xy in BAND}
    witness_fps = {(x + offset[0], y + offset[1]):
                   fp_at(env, (x, y))
                   for y in range(H) for x in range(W)}
    water_private = (water[0] + offset[0], water[1] + offset[1])

    def judge(r: Dict[str, Any]) -> Dict[str, Any]:
        t = r["transported"]
        return {**r,
                "transported": list(t) if t else None,
                "correct": t == water,
                "fail_open": t is not None and t != water}

    row: Dict[str, Any] = {
        "seed": seed, "offset": list(offset), "water": list(water),
        "n_couplets": len(song),
        "a_song": judge(arm_song(song, band_fps)),
        "b_broken": judge(arm_beats(song, TRAVELER_START)),
        "b_co": (judge(arm_beats(song_co, TRAVELER_START))
                 if song_co else None),
        "g_sigs": judge(arm_sigs_only(song, band_fps)),
        "v_snap": judge(arm_snapshot(witness_fps, water_private,
                                     band_fps)),
        "bits": {
            "a_song": bits_song(song, True, True),
            "b_beats": bits_song(song, False, True),
            "g_sigs": bits_song(song, True, False),
            "v_snap": bits_snapshot(witness_fps),
        },
    }
    return row


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "s0_registered.json"), "w") as f:
        json.dump({
            "codec": {"KEY_BITS": KEY_BITS, "LEN_BITS": LEN_BITS,
                      "BEAT_BITS": BEAT_BITS, "COORD_BITS": COORD_BITS},
            "S0.1": "beats-only anchor-broken: locks 100%, correct 0% "
                    "(fail open); co-anchored control 100% correct",
            "S0.2": "song arm: zero fail-open; every transport exact; "
                    "coverage >= 0.8 (rest are refusals)",
            "S0.3": "mean bits(song) <= 0.35 * mean bits(snapshot) at "
                    "equal per-transport correctness",
            "S0.4": "signatures-only: zero target transports",
        }, f, indent=2)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for seed in SEEDS:
        row = run_seed(seed, offset=(3, 2))
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    with open(os.path.join(OUT_DIR, "s0_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def rate(key: str, field: str) -> float:
        sel = [r[key] for r in rows if r.get(key)]
        return (sum(1 for a in sel if a[field]) / len(sel)) if sel else 0.0

    n = len(rows)
    a_transports = [r for r in rows if r["a_song"]["transported"]]
    mean_bits = {k: float(np.mean([r["bits"][k] for r in rows]))
                 for k in rows[0]["bits"]}
    summary = {
        "n_worlds": n, "skipped_walled_off": skipped,
        "a_song": {"coverage": len(a_transports) / n,
                   "correct_of_transported":
                       (sum(1 for r in a_transports
                            if r["a_song"]["correct"]) /
                        max(1, len(a_transports))),
                   "fail_open": sum(1 for r in rows
                                    if r["a_song"]["fail_open"]),
                   "refusal_reasons":
                       [r["a_song"]["reason"] for r in rows
                        if not r["a_song"]["transported"]]},
        "b_broken": {"locked": rate("b_broken", "transported") or
                     sum(1 for r in rows
                         if r["b_broken"]["transported"]) / n,
                     "correct": rate("b_broken", "correct"),
                     "fail_open": sum(1 for r in rows
                                      if r["b_broken"]["fail_open"])},
        "b_co": {"correct": rate("b_co", "correct")},
        "g_sigs": {"transports": sum(1 for r in rows
                                     if r["g_sigs"]["transported"])},
        "v_snap": {"correct": rate("v_snap", "correct"),
                   "fail_open": sum(1 for r in rows
                                    if r["v_snap"]["fail_open"])},
        "mean_bits": mean_bits,
        "bits_ratio_song_vs_snapshot":
            mean_bits["a_song"] / mean_bits["v_snap"],
    }

    s01 = (summary["b_broken"]["fail_open"] == n
           and summary["b_broken"]["correct"] == 0.0
           and summary["b_co"]["correct"] == 1.0)
    s02 = (summary["a_song"]["fail_open"] == 0
           and summary["a_song"]["correct_of_transported"] == 1.0
           and summary["a_song"]["coverage"] >= 0.8)
    s03 = (summary["bits_ratio_song_vs_snapshot"] <= 0.35
           and summary["a_song"]["correct_of_transported"]
           >= summary["v_snap"]["correct"])
    s04 = summary["g_sigs"]["transports"] == 0
    verdict = {"S0.1_beats_do_not_carry_identity": s01,
               "S0.2_songs_fail_safe_transport": s02,
               "S0.3_fraction_of_the_bits": s03,
               "S0.4_nodes_cannot_transport_unseen": s04}

    with open(os.path.join(OUT_DIR, "s0_results.json"), "w") as f:
        json.dump({"summary": summary, "verdict": verdict}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/s0_results.json")


if __name__ == "__main__":
    main()
