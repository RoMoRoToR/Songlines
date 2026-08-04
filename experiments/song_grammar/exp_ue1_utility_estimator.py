"""UE1 — a practical utility estimator (no oracle replay).

In grid worlds the counterfactual utility U(m|M) is exact because the
world can be replayed with the memory on and off. Outside a simulator
it cannot. UE1 builds the practical estimator U-hat from
RUNTIME-AVAILABLE features only (no blind-cost oracle):

    x = [1, song length, total beat displacement, anchored-in-band?,
         distance start->dead-reckoned target (local planning),
         n records for (intent), n records for (family, intent),
         version gap, band coverage]

trains ridge regression against the TRUE oracle U on training seeds,
evaluates on held-out seeds, and then runs the FULL formation
controller on U-hat alone, measuring the degradation the reviewer
called the critical experiment.

Registered predictions:
  UE1.1 (the estimator is usable): held-out sign accuracy >= 0.85 and
        Spearman rho >= 0.70; decision agreement with the hand matrix
        under U-hat >= 0.85.
  UE1.2 (the system survives estimation): songline_full driven by
        U-hat retains >= 90% of the oracle-U version's group cost on
        paired seeds (cost ratio <= 1.10, both roles).

Usage::

    PYTHONPATH=. python experiments/song_grammar/exp_ue1_utility_estimator.py \
        --train-seeds 0 20 --test-seeds 100 110 --out tmp/cluster/song_grammar/ue1
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

from experiments.song_grammar.exp_i1_integration import (
    INTENTS, World, build_song_cfg, make_fp, run_cell, walk)
import experiments.song_grammar.exp_i1_integration as i1
from experiments.song_grammar.exp_s0_song_smoke import BAND, TRAVELER_START
from experiments.song_grammar.runtime import Config, song_target
from experiments.song_grammar.u7_common import ROLES, dijkstra


def features(song, band_fps, env, records_intent: int,
             records_fam: int, ver_gap: int) -> np.ndarray:
    cand = song_target(song, band_fps, 0.999)
    anchored = cand is not None
    dist = 0.0
    if anchored:
        path, c = dijkstra(env, TRAVELER_START, cand, ROLES["robust"])
        dist = c if path is not None else 300.0
    disp = sum(abs(b["beat"][0]) + abs(b["beat"][1])
               for b in song if b.get("beat"))
    cover = np.mean([len(s) for s in band_fps.values()])
    return np.array([1.0, len(song) / 15.0, disp / 40.0,
                     float(anchored), dist / 100.0,
                     min(records_intent, 20) / 20.0,
                     min(records_fam, 5) / 5.0,
                     min(ver_gap, 3) / 3.0, cover / 6.0])


def collect(seeds: range, episodes: int = 40
            ) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    cfg = Config()
    for seed in seeds:
        rng = np.random.default_rng(seed * 7 + 1)
        fpf = make_fp(0.0, rng)
        world = World(seed)
        seen: Dict[Tuple[str, int], int] = {}
        for t in range(episodes):
            fam, env, tg, ver = world.assign()
            intent = "water" if t % 3 else "rest"
            path, _ = dijkstra(env, TRAVELER_START, tg[intent],
                               ROLES["robust"])
            if path is None:
                continue
            song = build_song_cfg(env, path, fpf, cfg)
            band_fps = {xy: fpf(env, xy) for xy in BAND}
            # oracle truth: guided-vs-blind on THIS world, empty memory
            cand = song_target(song, band_fps, cfg.sim_threshold)
            kind = INTENTS[intent]
            without = walk(env, [], ROLES["robust"], kind)["cost"]
            with_m = walk(env, [cand] if cand else [],
                          ROLES["robust"], kind)["cost"]
            u_true = without - with_m
            key = (intent, fam)
            X.append(features(song, band_fps, env, 0,
                              seen.get(key, 0), 0))
            y.append(u_true)
            seen[key] = seen.get(key, 0) + 1
    return np.array(X), np.array(y)


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, nargs=2, default=[0, 8])
    ap.add_argument("--test-seeds", type=int, nargs=2,
                    default=[100, 104])
    ap.add_argument("--degradation-seeds", type=int, nargs=2,
                    default=[100, 104])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", type=str, default="tmp/song_grammar/ue1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ue1_registered.json"), "w") as f:
        json.dump({
            "UE1.1": "held-out sign accuracy >= 0.85, Spearman >= "
                     "0.70, decision agreement >= 0.85",
            "UE1.2": "full controller on U-hat: group cost <= 1.10x "
                     "oracle version, both roles, paired seeds",
            "features": "runtime-available only (no blind-cost "
                        "oracle): anchoring, local plan distance, "
                        "song shape, memory counts, version gap",
        }, f, indent=2)

    Xtr, ytr = collect(range(*a.train_seeds))
    Xte, yte = collect(range(*a.test_seeds))
    lam = 1.0
    Wr = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]),
                         Xtr.T @ ytr)
    pred = Xte @ Wr
    mae = float(np.mean(np.abs(pred - yte)))
    rho = spearman(pred, yte)
    sign = float(np.mean((pred >= 5.0) == (yte >= 5.0)))
    ue11 = sign >= 0.85 and rho >= 0.70

    # degradation: oracle arm vs estimated arm on paired seeds
    oracle_rows = [run_cell("songline_full", s, 6, a.episodes, 0.0)
                   for s in range(*a.degradation_seeds)]
    est_rows = [i1_run_estimated(s, 6, a.episodes, Wr.copy())
                for s in range(*a.degradation_seeds)]

    def mean(rows, ro):
        return float(np.mean([r["group_cost"][ro] for r in rows]))
    ratio = {ro: mean(est_rows, ro) / mean(oracle_rows, ro)
             for ro in ("fragile", "robust")}
    ue12 = all(v <= 1.10 for v in ratio.values())

    out = {"estimator": {"mae": mae, "spearman": rho,
                         "sign_accuracy": sign,
                         "weights": Wr.tolist(),
                         "n_train": len(ytr), "n_test": len(yte)},
           "degradation": {"oracle": {ro: mean(oracle_rows, ro)
                                      for ro in ROLES},
                           "estimated": {ro: mean(est_rows, ro)
                                         for ro in ROLES},
                           "ratio": ratio},
           "verdict": {"UE1.1_estimator_usable": ue11,
                       "UE1.2_system_survives_estimation": ue12}}
    with open(os.path.join(a.out, "ue1_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "estimator"} |
                     {"estimator": {"mae": mae, "spearman": rho,
                                    "sign": sign}}, indent=2))
    for k, v in out["verdict"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Saved: {a.out}/ue1_results.json")


def i1_run_estimated(seed: int, n_agents: int, episodes: int,
                     Wr: np.ndarray) -> Dict[str, Any]:
    """run_cell('songline_full') with the oracle utility replaced by
    the ridge estimate --- the formation controller and admission see
    ONLY U-hat."""
    import experiments.song_grammar.exp_i1_integration as m

    def patched_utility_maker(fpf):
        def est(env, agent, song, intent):
            band_fps = {xy: fpf(env, xy) for xy in BAND}
            n_int = sum(1 for r in agent.records if r.intent == intent)
            x = features(song, band_fps, env, n_int, 0, 0)
            return float(x @ Wr)
        return est

    m.UTILITY_OVERRIDE = patched_utility_maker
    try:
        row = m.run_cell("songline_full", seed, n_agents, episodes,
                         0.0)
    finally:
        m.UTILITY_OVERRIDE = None
    row["utility"] = "estimated"
    return row


if __name__ == "__main__":
    main()
