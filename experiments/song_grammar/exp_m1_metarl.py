"""M1 — long-horizon meta-RL over the memory controller.

The U2 bandit optimised each formation decision against an IMMEDIATE
two-world probe. M1 removes that crutch: an evolution-strategies
meta-learner optimises the same policy class (softmax over the five
operations, linear in the U2 features) against a TERMINAL-ONLY
reward --- one number per full stream (final battery cost + storage
tax), delivered after dozens of interleaved decisions. Credit
assignment across the whole lifetime of the memory is exactly what a
contextual bandit cannot represent: a DROP today can starve a family
weeks later; an EXCEPTION pays only when the conflicting world
returns.

Registered predictions:
  M1.1 (terminal credit suffices): the meta-learned policy's held-out
       final cost is <= the immediate-reward bandit's, and within 5%
       of the hand matrix (or better) --- long-horizon selection
       recovers what dense shaping had to be designed to provide.
  M1.2 (no corruption shortcut): on candidates where the hand matrix
       says EXCEPTION, the meta-learned policy picks a STORE-class
       action (new/exception) in >= 80% of cases and never a
       replace-majority --- the terminal reward alone must teach it
       not to overwrite.

Usage::

    PYTHONPATH=. python experiments/song_grammar/exp_m1_metarl.py \
        --generations 40 --pop 32 --episodes 30 \
        --train-seeds 0 4 --holdout-seeds 300 308 \
        --out tmp/cluster/song_grammar/m1
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

from experiments.song_grammar.exp_u2_bandit import (
    ACTIONS, DIM, LinUCB, apply_op, features, hand_policy,
    ordered_songs)
from experiments.song_grammar.u7_common import (
    ROLES, bits_of_song, consumer_cost, eval_battery, make_stream,
    marginal_utility)
from experiments.song_grammar.ucsm import Schema, nearest

ROLE = ROLES["robust"]
LAMBDA_BITS = 1.0


def lifecycle(policy, seed: int, n_episodes: int,
              rng: Optional[np.random.Generator] = None,
              record: Optional[List[Tuple[str, str]]] = None
              ) -> float:
    """Run a full stream under `policy(x, rng) -> op`; terminal-only
    reward (negative cost; NO per-decision signal)."""
    stream = make_stream(seed, n_episodes)
    items: List[Dict[str, Any]] = []
    for ep in stream:
        songs = [it["song"] for it in items]
        u = marginal_utility(ep.env, songs, ep.song, ROLE)
        idx, ana = nearest(ep.song, [Schema(it["song"], cert=None)
                                     for it in items])
        x = features(u, ana, len(items))
        op = policy(x, rng)
        if record is not None:
            record.append((hand_policy(u, ana), op))
        apply_op(op, items, ep.song, idx, ep.env, ep.family)
    battery = eval_battery(stream, seed)
    songs = ordered_songs(items)
    cost = float(np.mean([consumer_cost(ep.env, songs, ROLE)["cost"]
                          for ep in battery])) if battery else 1e9
    bits = sum(bits_of_song(it["song"]) for it in items)
    return -(cost + LAMBDA_BITS * bits / 1000.0)


# ── policy classes ─────────────────────────────────────────────────

def softmax_policy(W: np.ndarray):
    def policy(x, rng):
        z = W @ x
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        if rng is None:
            return ACTIONS[int(np.argmax(p))]
        return ACTIONS[int(rng.choice(len(ACTIONS), p=p))]
    return policy


def bandit_policy(bandit: LinUCB):
    def policy(x, rng):
        return bandit.choose(x, explore=False)
    return policy


def train_bandit(train: range, n_episodes: int) -> LinUCB:
    """The U2 immediate-reward comparator, same training budget."""
    from experiments.song_grammar.exp_u2_bandit import probe_cost
    rng = np.random.default_rng(7)
    bandit = LinUCB(rng)
    for seed in train:
        stream = make_stream(seed, n_episodes)
        items: List[Dict[str, Any]] = []
        for ep in stream:
            songs = [it["song"] for it in items]
            u = marginal_utility(ep.env, songs, ep.song, ROLE)
            idx, ana = nearest(ep.song, [Schema(it["song"], cert=None)
                                         for it in items])
            x = features(u, ana, len(items))
            op = bandit.choose(x, explore=True)
            probe = [ep.env] + ([items[idx]["env"]]
                                if idx is not None else [])
            before = probe_cost(items, probe)
            trial = [dict(it) for it in items]
            apply_op(op, trial, ep.song, idx, ep.env, ep.family)
            after = probe_cost(trial, probe)
            added = (bits_of_song(ep.song)
                     if op in ("NEW_SCHEMA", "EXCEPTION") else 0)
            bandit.update(x, op, (before - after) / 50.0
                          - 0.002 * added / 10.0)
            items = trial
    return bandit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--train-seeds", type=int, nargs=2, default=[0, 4])
    ap.add_argument("--holdout-seeds", type=int, nargs=2,
                    default=[300, 308])
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/m1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "m1_registered.json"), "w") as f:
        json.dump({
            "M1.1": "meta-learned held-out cost <= bandit's and "
                    "within 5% of hand (or better)",
            "M1.2": "on hand-EXCEPTION candidates: store-class >= "
                    "0.8, never replace-majority",
            "reward": "terminal-only: -(battery cost + bits/1000) "
                      "per full stream",
        }, f, indent=2)

    train = range(a.train_seeds[0], a.train_seeds[1])
    holdout = range(a.holdout_seeds[0], a.holdout_seeds[1])
    rng = np.random.default_rng(13)

    # ES over W (5 x DIM), terminal-only fitness
    pop = [rng.normal(0, 0.5, size=(len(ACTIONS), DIM))
           for _ in range(a.pop)]
    best_hist = []
    for gen in range(a.generations):
        fits = []
        for W in pop:
            pol = softmax_policy(W)
            fits.append(float(np.mean([
                lifecycle(pol, s, a.episodes,
                          rng=np.random.default_rng(1000 + gen))
                for s in train])))
        order = np.argsort(fits)[::-1]
        elite = [pop[i] for i in order[:max(2, a.pop // 4)]]
        best_hist.append({"gen": gen, "fit": fits[order[0]]})
        print(f"gen {gen}: best {fits[order[0]]:.1f} "
              f"median {np.median(fits):.1f}", flush=True)
        pop = list(elite)
        while len(pop) < a.pop:
            base = elite[int(rng.integers(len(elite)))]
            pop.append(base + rng.normal(0, a.sigma, size=base.shape))
    W_best = pop[0]

    bandit = train_bandit(train, a.episodes)
    policies = {
        "meta_es": softmax_policy(W_best),
        "bandit": bandit_policy(bandit),
        "random": lambda x, r: ACTIONS[int(
            np.random.default_rng(int(abs(x.sum()) * 1e6) % 2**31
                                  ).integers(len(ACTIONS)))],
    }

    def hand_pol_runner(seed):
        stream = make_stream(seed, a.episodes)
        items: List[Dict[str, Any]] = []
        for ep in stream:
            songs = [it["song"] for it in items]
            u = marginal_utility(ep.env, songs, ep.song, ROLE)
            idx, ana = nearest(ep.song, [Schema(it["song"], cert=None)
                                         for it in items])
            apply_op(hand_policy(u, ana), items, ep.song, idx, ep.env,
                     ep.family)
        battery = eval_battery(stream, seed)
        songs = ordered_songs(items)
        cost = float(np.mean([consumer_cost(e.env, songs, ROLE)["cost"]
                              for e in battery])) if battery else 1e9
        bits = sum(bits_of_song(it["song"]) for it in items)
        return -(cost + LAMBDA_BITS * bits / 1000.0)

    summary: Dict[str, Any] = {"holdout_reward": {}}
    rec_meta: List[Tuple[str, str]] = []
    for name in ("meta_es", "bandit", "random"):
        pol = policies[name]
        vals = [lifecycle(pol, s, a.episodes,
                          record=rec_meta if name == "meta_es"
                          else None)
                for s in holdout]
        summary["holdout_reward"][name] = float(np.mean(vals))
    summary["holdout_reward"]["hand"] = float(np.mean(
        [hand_pol_runner(s) for s in holdout]))

    exc = [taken for hand, taken in rec_meta if hand == "EXCEPTION"]
    store_share = (sum(1 for op in exc
                       if op in ("NEW_SCHEMA", "EXCEPTION"))
                   / max(1, len(exc)))
    merge_major = (sum(1 for op in exc if op == "MERGE")
                   > len(exc) / 2) if exc else False
    hr = summary["holdout_reward"]
    m11 = (hr["meta_es"] >= hr["bandit"]
           and hr["meta_es"] >= hr["hand"] - abs(hr["hand"]) * 0.05)
    m12 = store_share >= 0.8 and not merge_major
    verdict = {"M1.1_terminal_credit_suffices": m11,
               "M1.2_no_corruption_shortcut": m12}
    out = {"summary": {**summary,
                       "exception_cells": len(exc),
                       "exception_store_share": store_share},
           "history": best_hist, "verdict": verdict,
           "W_best": W_best.tolist()}
    with open(os.path.join(a.out, "m1_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Saved: {a.out}/m1_results.json")


if __name__ == "__main__":
    main()
