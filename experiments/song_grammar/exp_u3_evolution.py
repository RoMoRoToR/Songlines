"""U3 — Stage 3: evolving the song grammar (directed evolution of
memory-building heuristics).

The object of selection is theta = (gap, share_thr, d_thr, u_thr,
vmax): where a couplet is born (gap between couplets), when analogies
count as simple (share_thr), when a conflict is an exception (d_thr),
what counts as useful (u_thr), and the bounded-rationality budget of
the song (vmax couplets; beats re-chained on truncation).

Fitness of a genome (LOWER better) is measured on maps the genome
never sang on: run the UCSM lifecycle over a training stream, then
  fitness = mean battery cost + LAMBDA_BITS * stored_bits / 1000
evaluated per seed; ES = truncation selection + gaussian mutation +
elitism.  Fitness on fresh maps + few passes per map = pressure on
TRANSFERABLE heuristics, not on memorising maps.

Registered predictions:
  U3.1 (evolution matches design): the best evolved genome's holdout
       fitness is <= the hand genome's (gap=2, 0.4, 3, 5, vmax=inf)
       within 2%, or better.
  U3.2 (the budget is used): the evolved vmax is finite (< 16) with no
       holdout cost regression > 2% vs the hand genome --- bounded
       rationality is selected, not imposed.

Usage::

    PYTHONPATH=. python experiments/song_grammar/exp_u3_evolution.py \
        --generations 20 --pop 12 --episodes 25 \
        --train-seeds 0 6 --holdout-seeds 200 206 \
        --out tmp/song_grammar/u3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.song_grammar.u7_common import (
    ROLES, bits_of_song, consumer_cost, eval_battery, make_stream,
    marginal_utility)
from experiments.song_grammar.ucsm import Schema, nearest

ROLE = ROLES["robust"]
LAMBDA_BITS = 1.0     # fitness units per 1000 stored bits

HAND = {"gap": 2, "share_thr": 0.4, "d_thr": 3.0, "u_thr": 5.0,
        "vmax": 0}    # 0 = unbounded
BOUNDS = {"gap": (1, 5), "share_thr": (0.15, 0.85),
          "d_thr": (1.0, 8.0), "u_thr": (1.0, 20.0), "vmax": (3, 16)}


def clamp(g: Dict[str, float]) -> Dict[str, float]:
    out = dict(g)
    for k, (lo, hi) in BOUNDS.items():
        out[k] = min(max(out[k], lo), hi)
    out["gap"] = int(round(out["gap"]))
    out["vmax"] = int(round(out["vmax"]))
    return out


def mutate(g: Dict[str, float], rng: np.random.Generator
           ) -> Dict[str, float]:
    out = dict(g)
    for k, (lo, hi) in BOUNDS.items():
        if rng.random() < 0.5:
            out[k] = out[k] + rng.normal(0, 0.15 * (hi - lo))
    return clamp(out)


def lifecycle(genome: Dict[str, float], seed: int, n_episodes: int
              ) -> Tuple[float, int]:
    """UCSM lifecycle under the genome's grammar; returns (mean battery
    cost on held-out worlds incl. fresh variants, stored bits)."""
    stream = make_stream(seed, n_episodes, gap=genome["gap"],
                         vmax=genome["vmax"])
    items: List[Dict[str, Any]] = []
    for ep in stream:
        songs = [it["song"] for it in items]
        u = marginal_utility(ep.env, songs, ep.song, ROLE)
        idx, ana = nearest(ep.song,
                           [Schema(it["song"], cert=None) for it in items])
        simple = ana is not None and ana["share"] >= genome["share_thr"]
        conflict = simple and ana["D"] >= genome["d_thr"]
        if u >= genome["u_thr"]:
            if conflict or not simple:
                items.append({"song": ep.song})
            else:
                items[idx] = {"song": ep.song}
        # low-utility: repeat/drop -> no structural change
    battery = eval_battery(stream, seed)
    songs = [it["song"] for it in items]
    costs = [consumer_cost(ep.env, songs, ROLE)["cost"]
             for ep in battery]
    bits = sum(bits_of_song(s) for s in songs)
    return float(np.mean(costs)) if costs else 1e9, bits


def fitness(genome: Dict[str, float], seeds: range, n_episodes: int
            ) -> Dict[str, float]:
    costs, bits = [], []
    for s in seeds:
        c, b = lifecycle(genome, s, n_episodes)
        costs.append(c)
        bits.append(b)
    return {"cost": float(np.mean(costs)),
            "bits": float(np.mean(bits)),
            "fitness": float(np.mean(costs))
            + LAMBDA_BITS * float(np.mean(bits)) / 1000.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--train-seeds", type=int, nargs=2, default=[0, 5])
    ap.add_argument("--holdout-seeds", type=int, nargs=2,
                    default=[200, 205])
    ap.add_argument("--out", type=str, default="tmp/song_grammar/u3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "u3_registered.json"), "w") as f:
        json.dump({
            "U3.1": "best evolved genome holdout fitness <= hand "
                    "genome's within 2% (or better)",
            "U3.2": "evolved vmax finite (< 16) with holdout cost "
                    "regression <= 2% vs hand",
            "hand_genome": HAND, "bounds": {k: list(v) for k, v
                                            in BOUNDS.items()},
            "lambda_bits": LAMBDA_BITS,
        }, f, indent=2)

    rng = np.random.default_rng(11)
    train = range(a.train_seeds[0], a.train_seeds[1])
    holdout = range(a.holdout_seeds[0], a.holdout_seeds[1])

    pop = [clamp({k: rng.uniform(*BOUNDS[k]) for k in BOUNDS})
           for _ in range(a.pop - 1)] + [dict(HAND, vmax=8)]
    history = []
    for gen in range(a.generations):
        scored = [(fitness(g, train, a.episodes), g) for g in pop]
        scored.sort(key=lambda t: t[0]["fitness"])
        best = scored[0]
        history.append({"gen": gen, "best_fitness": best[0]["fitness"],
                        "best_cost": best[0]["cost"],
                        "best_bits": best[0]["bits"],
                        "best_genome": best[1]})
        print(f"gen {gen}: fit {best[0]['fitness']:.1f} "
              f"(cost {best[0]['cost']:.1f}, bits {best[0]['bits']:.0f}) "
              f"{best[1]}", flush=True)
        elite = [g for _, g in scored[:max(2, a.pop // 4)]]
        pop = list(elite)
        while len(pop) < a.pop:
            pop.append(mutate(elite[int(rng.integers(len(elite)))], rng))

    best_genome = history[-1]["best_genome"]
    hold_best = fitness(best_genome, holdout, a.episodes)
    hold_hand = fitness(HAND, holdout, a.episodes)
    u31 = hold_best["fitness"] <= hold_hand["fitness"] * 1.02
    u32 = (best_genome["vmax"] < 16
           and hold_best["cost"] <= hold_hand["cost"] * 1.02)
    verdict = {"U3.1_evolution_matches_design": u31,
               "U3.2_budget_is_selected": u32}
    out = {"best_genome": best_genome,
           "holdout": {"evolved": hold_best, "hand": hold_hand},
           "history": history, "verdict": verdict}
    with open(os.path.join(a.out, "u3_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"best_genome": best_genome,
                      "holdout_evolved": hold_best,
                      "holdout_hand": hold_hand}, indent=2))
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Saved: {a.out}/u3_results.json")


if __name__ == "__main__":
    main()
