"""G1 — the graph-matching analogy engine vs the LCS proxy.

The first-wave analogy was signature-LCS over EXACT constellation
equality: an appearance variant (same walls-and-water structure, new
hazard texture) changes most couplet signatures, so LCS collapses and
the two-axis controller files the same structure as a NEW schema ---
bloat by blindness to partial similarity.

G1 replaces it with structural graph matching: Needleman--Wunsch
alignment over couplets with a PARTIAL node similarity (Jaccard over
signature keys), a gap penalty, a relational defect (beat-chain
consistency between consecutive matched pairs --- do the aligned
songs walk the same skeleton?), and the same decision-distortion gate
(where the songs ultimately send the walker).

Registered predictions (appearance-heavy streams, both roles):
  G1.1 (abstraction pays): the graph-matching controller stores
       <= 0.75x the LCS controller's schemas AND <= 0.80x its bits,
       with mean battery cost within 5% on both roles.
  G1.2 (no safety price): the graph controller's phantom-first rate
       exceeds the LCS controller's by no more than 0.03 --- partial
       matching must not turn refusals into wrong locks.

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_g1_graph_analogy.py \
        --seeds 0 3 --episodes 300 --out tmp/cluster/song_grammar/g1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.song_grammar.exp_s0_song_smoke import BAND, arm_song, fp_at
from experiments.song_grammar.exp_u7_seven_arms import (
    UcsmStaleArm, walk_targets)
from experiments.song_grammar.u7_common import (
    ROLES, bits_of_song, eval_battery, make_stream, marginal_utility)

U_THR, SHARE_THR, D_THR = 5.0, 0.4, 3
NODE_SIM_MIN, GAP = 0.5, 0.45
DEFECT_THR = 4.0


# ── structural graph matching ──────────────────────────────────────

def jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
    ka, kb = set(a), set(b)
    return len(ka & kb) / max(1, len(ka | kb))


def graph_analogy(cand, schema) -> Dict[str, float]:
    """Needleman--Wunsch over couplets with partial node similarity;
    relational defect from beat-chain consistency of the alignment."""
    n, m = len(cand), len(schema)
    dp = np.zeros((n + 1, m + 1))
    dp[:, 0] = -GAP * np.arange(n + 1)
    dp[0, :] = -GAP * np.arange(m + 1)
    back = np.zeros((n + 1, m + 1), dtype=int)   # 0 diag, 1 up, 2 left
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = jaccard(cand[i - 1]["sig"], schema[j - 1]["sig"])
            cands = (dp[i - 1, j - 1] + sim,
                     dp[i - 1, j] - GAP, dp[i, j - 1] - GAP)
            back[i, j] = int(np.argmax(cands))
            dp[i, j] = cands[back[i, j]]
    pairs: List[Tuple[int, int, float]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if back[i, j] == 0:
            sim = jaccard(cand[i - 1]["sig"], schema[j - 1]["sig"])
            if sim >= NODE_SIM_MIN:
                pairs.append((i - 1, j - 1, sim))
            i, j = i - 1, j - 1
        elif back[i, j] == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    share = len(pairs) / max(1, min(n, m))
    # relational defect: aligned songs must walk the same skeleton
    defect = 0.0
    for (i1, j1, _), (i2, j2, _) in zip(pairs, pairs[1:]):
        bc = np.sum([cand[k]["beat"] for k in range(i1 + 1, i2 + 1)],
                    axis=0)
        bs = np.sum([schema[k]["beat"] for k in range(j1 + 1, j2 + 1)],
                    axis=0)
        defect += abs(int(bc[0] - bs[0])) + abs(int(bc[1] - bs[1]))
    defect /= max(1, len(pairs) - 1)
    da = np.sum([c["beat"] for c in cand], axis=0)
    db = np.sum([c["beat"] for c in schema], axis=0)
    return {"share": share, "defect": defect,
            "L": (n - len(pairs)) + (m - len(pairs)),
            "D": abs(int(da[0] - db[0])) + abs(int(da[1] - db[1]))}


class UcsmGraphArm(UcsmStaleArm):
    """Same staleness-gated consumption, graph-matching consolidation:
    simple = enough aligned skeleton (share, low relational defect);
    conflict = same skeleton but different destination."""
    name = "ucsm_graph"

    def observe(self, ep, t):
        self.now = t
        role_u = {r: marginal_utility(
            ep.env, [it["song"] for it in self.items], ep.song,
            ROLES[r]) for r in ROLES}
        u = max(role_u.values())
        best_i, best = None, None
        for i, it in enumerate(self.items):
            a = graph_analogy(ep.song, it["song"])
            if best is None or a["share"] > best["share"]:
                best_i, best = i, a
        simple = (best is not None and best["share"] >= SHARE_THR
                  and best["defect"] <= DEFECT_THR)
        conflict = simple and best["D"] >= D_THR
        if u >= U_THR:
            if conflict:
                self.items.append({"song": ep.song, "t": t,
                                   "role_u": role_u,
                                   "kind": "exception",
                                   "family": ep.family,
                                   "parent": best_i})
            elif simple:
                it = self.items[best_i]
                it["song"], it["t"] = ep.song, t
                it["role_u"] = {r: max(it["role_u"][r], role_u[r])
                                for r in ROLES}
                it["support"] = it.get("support", 1) + 1
            else:
                self.items.append({"song": ep.song, "t": t,
                                   "role_u": role_u, "kind": "schema",
                                   "family": ep.family, "support": 1})
        elif simple:
            self.items[best_i]["support"] = \
                self.items[best_i].get("support", 1) + 1


def run_seed(seed: int, n_episodes: int) -> Dict[str, Any]:
    # appearance-heavy stream: same structures return in new textures
    stream = make_stream(seed, n_episodes,
                         kind_probs=(0.15, 0.60, 0.25), new_rate=0.25)
    arms = [UcsmStaleArm(), UcsmGraphArm()]
    for t, ep in enumerate(stream):
        for arm in arms:
            arm.observe(ep, t)
    battery = eval_battery(stream, seed)
    out: Dict[str, Any] = {"seed": seed, "episodes": len(stream),
                           "battery": len(battery), "arms": {}}
    for arm in arms:
        per_role = {}
        for role_name, role in ROLES.items():
            costs, ph = [], 0
            for ep in battery:
                band_fps = {xy: fp_at(ep.env, xy) for xy in BAND}
                r = walk_targets(ep.env,
                                 arm.targets(ep.env, band_fps,
                                             role_name), role)
                costs.append(r["cost"])
                ph += int(r["phantom_first"])
            per_role[role_name] = {
                "mean_cost": float(np.mean(costs)),
                "phantom_rate": ph / len(battery)}
        out["arms"][arm.name] = {
            "roles": per_role, "n_items": len(arm.items),
            "bits": sum(bits_of_song(it["song"]) for it in arm.items)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/g1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "g1_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "G1.1": "graph controller: schemas <= 0.75x and bits "
                        "<= 0.80x of LCS controller, cost within 5% "
                        "both roles",
                "G1.2": "graph phantom rate <= LCS's + 0.03",
                "constants": {"NODE_SIM_MIN": NODE_SIM_MIN,
                              "GAP": GAP, "DEFECT_THR": DEFECT_THR,
                              "stream": "appearance-heavy "
                                        "(0.15/0.60/0.25, new 0.25)"},
            }, f, indent=2)
    shard = f"g1_e{a.episodes}_s{a.seeds[0]}-{a.seeds[1]}.jsonl"
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            row = run_seed(seed, a.episodes)
            f.write(json.dumps(row) + "\n")
            s, g = row["arms"]["ucsm_stale"], row["arms"]["ucsm_graph"]
            print(f"seed {seed}: LCS items {s['n_items']} "
                  f"cost {s['roles']['fragile']['mean_cost']:.0f} | "
                  f"GRAPH items {g['n_items']} "
                  f"cost {g['roles']['fragile']['mean_cost']:.0f}",
                  flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
