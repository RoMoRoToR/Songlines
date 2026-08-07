"""UE1-EVAL — full evaluation of the practical utility estimator.

The paper currently reports one number for UE1 (held-out Spearman
rho = 0.989, tmp/song_grammar/ue1_cluster.json). Reviewers asked for a
complete characterization. This script imports the FROZEN estimator
code from exp_ue1_utility_estimator.py (features, ridge lambda = 1.0)
and exp_i1_integration.py (worlds, oracle utility) — nothing is
retuned — and reports, on the same held-out worlds plus OOD slices:

  1. MAE, RMSE (plus Spearman / Pearson for continuity with the paper).
  2. Sign accuracy (sign of U-hat vs sign of U*), and precision /
     recall for the class U* > 0; both at threshold 0 and at the
     registered UE1 decision threshold tau = 5.0.
  3. Calibration: 10 equal-count bins of U-hat -> mean U* per bin
     (reliability table) + weighted calibration error.
  4. Top-k recall (k = 5, 10, 20): per held-out seed pool, the
     fraction of the true top-k utility records (by U*) that the
     estimator's top-k captures.
  5. Policy regret (the key metric): retention under a fixed memory
     budget B. pi_score keeps the top-B records of a seed pool ranked
     by the score; J(pi) = sum of exact-replay utilities U* of the
     retained set. Regret = J(pi_U*) - J(pi_U-hat), reported per
     budget with a random-retention baseline for scale.
  6. OOD slices (train on some, eval on others):
       - layout / hazard texture: train worlds with variant == 0
         (fresh families), eval held-out worlds with variant > 0;
       - layout / water slot: train widx in {0, 1}, eval widx == 2
         (the moved-water conflict layout);
       - agent role: train on robust-role oracle U*, eval on
         fragile-role oracle U* (features are frozen / role-agnostic);
       - horizon: train on episodes t < 40 (the training horizon),
         eval held-out streams run to t = 120, sliced by t.

Protocol discipline: train seeds 0-19, held-out test seeds 100+ (the
exact split behind rho = 0.989: 20 x 40 train, 10 x 40 test). Ridge
lambda = 1.0 is frozen from the original UE1 run — nothing here is
tuned, so test seeds are used for evaluation only. World families are
keyed as seed * 100_000 + counter, so train and test families (wall
layouts) are disjoint by construction: even the in-distribution eval
is on never-seen layouts.

Usage::

    PYTHONPATH=. python experiments/song_grammar/exp_utility_estimator_eval.py \
        --train-seeds 0 20 --test-seeds 100 110 --horizon-episodes 120 \
        --out tmp/song_grammar/ue1_eval

    # quick smoke
    PYTHONPATH=. python experiments/song_grammar/exp_utility_estimator_eval.py \
        --smoke --out tmp/song_grammar/ue1_eval_smoke
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
    INTENTS, World, build_song_cfg, make_fp, walk)
from experiments.song_grammar.exp_s0_song_smoke import BAND, TRAVELER_START
from experiments.song_grammar.exp_ue1_utility_estimator import (
    features, spearman)
from experiments.song_grammar.runtime import Config, song_target
from experiments.song_grammar.u7_common import ROLES, dijkstra

RIDGE_LAMBDA = 1.0       # frozen from exp_ue1_utility_estimator.py
SIGN_TAU = 5.0           # registered UE1.1 decision threshold
TOPK = (5, 10, 20)
BUDGETS = (5, 10, 20)
N_BINS = 10


# ── data collection (UE1 collect + metadata + role parameter) ──────

def collect_meta(seeds: range, episodes: int = 40,
                 role_name: str = "robust"
                 ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Byte-for-byte the sampling loop of UE1's collect() (same rng
    streams, same feature map), extended with per-sample metadata and
    a role parameter for the oracle walks. role='robust' reproduces
    the original data exactly (verified by --selfcheck)."""
    X, y, meta = [], [], []
    cfg = Config()
    role = ROLES[role_name]
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
            cand = song_target(song, band_fps, cfg.sim_threshold)
            kind = INTENTS[intent]
            without = walk(env, [], role, kind)["cost"]
            with_m = walk(env, [cand] if cand else [], role,
                          kind)["cost"]
            u_true = without - with_m
            key = (intent, fam)
            X.append(features(song, band_fps, env, 0,
                              seen.get(key, 0), 0))
            y.append(u_true)
            variant, widx = world.state[fam]
            meta.append({"seed": seed, "t": t, "intent": intent,
                         "family": fam, "version": ver,
                         "variant": variant, "widx": widx,
                         "role": role_name})
            seen[key] = seen.get(key, 0) + 1
    return np.array(X), np.array(y), meta


def fit_ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(X.T @ X + RIDGE_LAMBDA * np.eye(X.shape[1]),
                           X.T @ y)


# ── metric blocks ──────────────────────────────────────────────────

def core_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    err = pred - y
    n = len(y)
    out: Dict[str, Any] = {
        "n": int(n),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "spearman": spearman(pred, y) if n > 2 else None,
        "pearson": (float(np.corrcoef(pred, y)[0, 1])
                    if n > 2 and pred.std() > 0 and y.std() > 0
                    else None),
        "mean_u_true": float(np.mean(y)),
        "std_u_true": float(np.std(y)),
    }
    for tau, tag in ((0.0, "0"), (SIGN_TAU, "tau5")):
        p_pos, t_pos = pred > tau, y > tau
        tp = int(np.sum(p_pos & t_pos))
        fp = int(np.sum(p_pos & ~t_pos))
        fn = int(np.sum(~p_pos & t_pos))
        out[f"sign_accuracy_{tag}"] = float(np.mean(p_pos == t_pos))
        out[f"precision_pos_{tag}"] = (tp / (tp + fp)
                                       if tp + fp else None)
        out[f"recall_pos_{tag}"] = (tp / (tp + fn)
                                    if tp + fn else None)
        out[f"base_rate_pos_{tag}"] = float(np.mean(t_pos))
    return out


def calibration_table(pred: np.ndarray, y: np.ndarray,
                      n_bins: int = N_BINS) -> Dict[str, Any]:
    """Equal-count bins of U-hat; per bin the mean U-hat vs mean U*."""
    order = np.argsort(pred)
    bins = np.array_split(order, n_bins)
    rows = []
    for idx in bins:
        if len(idx) == 0:
            continue
        rows.append({"n": int(len(idx)),
                     "uhat_lo": float(pred[idx].min()),
                     "uhat_hi": float(pred[idx].max()),
                     "uhat_mean": float(pred[idx].mean()),
                     "ustar_mean": float(y[idx].mean()),
                     "ustar_std": float(y[idx].std())})
    ce = sum(r["n"] * abs(r["uhat_mean"] - r["ustar_mean"])
             for r in rows) / max(1, sum(r["n"] for r in rows))
    return {"bins": rows, "weighted_calibration_error": float(ce)}


def pool_ids(meta: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    pools: Dict[int, List[int]] = {}
    for i, m in enumerate(meta):
        pools.setdefault(m["seed"], []).append(i)
    return {s: np.array(v) for s, v in pools.items()}


def topk_recall(pred: np.ndarray, y: np.ndarray,
                meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    pools = pool_ids(meta)
    out: Dict[str, Any] = {}
    for k in TOPK:
        vals, used = [], 0
        for _, idx in sorted(pools.items()):
            if len(idx) < k:
                continue
            true_top = set(idx[np.argsort(y[idx])[-k:]].tolist())
            est_top = set(idx[np.argsort(pred[idx])[-k:]].tolist())
            vals.append(len(true_top & est_top) / k)
            used += 1
        out[f"top{k}"] = {"recall": (float(np.mean(vals))
                                     if vals else None),
                          "n_pools": used}
    return out


def policy_regret(pred: np.ndarray, y: np.ndarray,
                  meta: List[Dict[str, Any]], seed0: int = 12345
                  ) -> Dict[str, Any]:
    """Retention under a fixed budget B per seed pool. J(pi) = sum of
    exact-replay U* over the B records pi keeps. Regret is
    J(pi_U*) - J(pi_U-hat) >= 0; random retention gives scale."""
    pools = pool_ids(meta)
    rng = np.random.default_rng(seed0)
    out: Dict[str, Any] = {}
    for B in BUDGETS:
        rows = []
        for _, idx in sorted(pools.items()):
            if len(idx) < B:
                continue
            j_star = float(np.sort(y[idx])[-B:].sum())
            keep = idx[np.argsort(pred[idx])[-B:]]
            j_hat = float(y[keep].sum())
            j_rand = float(np.mean(
                [y[rng.choice(idx, B, replace=False)].sum()
                 for _ in range(200)]))
            rows.append((j_star, j_hat, j_rand))
        if not rows:
            out[f"B{B}"] = {"n_pools": 0}
            continue
        js, jh, jr = (np.array([r[i] for r in rows])
                      for i in range(3))
        out[f"B{B}"] = {
            "n_pools": len(rows),
            "J_oracle_mean": float(js.mean()),
            "J_estimator_mean": float(jh.mean()),
            "J_random_mean": float(jr.mean()),
            "regret_mean": float((js - jh).mean()),
            "regret_std": float((js - jh).std()),
            "regret_frac_of_oracle": (float((js - jh).mean()
                                            / js.mean())
                                      if js.mean() > 0 else None),
            "captured_vs_random":
                (float((jh.mean() - jr.mean())
                       / (js.mean() - jr.mean()))
                 if js.mean() - jr.mean() > 0 else None),
        }
    return out


def full_block(pred: np.ndarray, y: np.ndarray,
               meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"core": core_metrics(pred, y),
            "calibration": calibration_table(pred, y),
            "topk_recall": topk_recall(pred, y, meta),
            "policy_regret": policy_regret(pred, y, meta)}


# ── printing ───────────────────────────────────────────────────────

def show(name: str, block: Dict[str, Any]) -> None:
    c = block["core"]

    def f(v, spec=".3f"):
        return "n/a" if v is None else format(v, spec)

    print(f"\n== {name} (n={c['n']}) ==")
    print(f"  MAE {c['mae']:.2f}  RMSE {c['rmse']:.2f}  "
          f"Spearman {f(c['spearman'])}  Pearson {f(c['pearson'])}"
          f"  (U* mean {c['mean_u_true']:.1f} "
          f"std {c['std_u_true']:.1f})")
    print(f"  sign acc @0 {c['sign_accuracy_0']:.3f} "
          f"(P {f(c['precision_pos_0'])} R {f(c['recall_pos_0'])}, "
          f"base {c['base_rate_pos_0']:.2f}) | @tau=5 "
          f"{c['sign_accuracy_tau5']:.3f} "
          f"(P {f(c['precision_pos_tau5'])} "
          f"R {f(c['recall_pos_tau5'])})")
    cal = block["calibration"]
    print(f"  calibration (weighted err "
          f"{cal['weighted_calibration_error']:.2f}):")
    print("    bin      U-hat range        mean U-hat   mean U*   n")
    for i, r in enumerate(cal["bins"]):
        print(f"    {i:2d}  [{r['uhat_lo']:8.1f},{r['uhat_hi']:8.1f}]"
              f"  {r['uhat_mean']:9.1f}  {r['ustar_mean']:8.1f}"
              f"  {r['n']:4d}")
    tk = block["topk_recall"]
    print("  top-k recall: " + "  ".join(
        f"k={k}: {f(tk[f'top{k}']['recall'])} "
        f"({tk[f'top{k}']['n_pools']} pools)" for k in TOPK))
    pr = block["policy_regret"]
    for B in BUDGETS:
        r = pr[f"B{B}"]
        if r.get("n_pools", 0) == 0:
            print(f"  regret B={B}: no pools large enough")
            continue
        print(f"  regret B={B}: J* {r['J_oracle_mean']:.0f} vs "
              f"J-hat {r['J_estimator_mean']:.0f} "
              f"(random {r['J_random_mean']:.0f}) -> regret "
              f"{r['regret_mean']:.1f} "
              f"({f(r['regret_frac_of_oracle'], '.1%')} of oracle; "
              f"captures {f(r['captured_vs_random'], '.1%')} of "
              f"oracle-over-random)")


# ── main ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, nargs=2, default=[0, 20])
    ap.add_argument("--test-seeds", type=int, nargs=2,
                    default=[100, 110])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--horizon-episodes", type=int, default=120)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: train 0-4, test 100-102")
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify collect_meta == frozen UE1 collect")
    ap.add_argument("--out", type=str,
                    default="tmp/song_grammar/ue1_eval")
    a = ap.parse_args()
    if a.smoke:
        a.train_seeds, a.test_seeds = [0, 4], [100, 102]
        a.horizon_episodes = 60
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.out, "ue1_eval_registered.json"),
              "w") as f:
        json.dump({
            "frozen": "features + ridge lambda=1.0 + tau=5.0 from "
                      "exp_ue1_utility_estimator.py; no tuning "
                      "anywhere in this script",
            "split": f"train seeds {a.train_seeds}, held-out test "
                     f"seeds {a.test_seeds} (families disjoint by "
                     f"construction: fam = seed*100000 + counter)",
            "metrics": ["mae/rmse", "sign acc + P/R (0 and tau=5)",
                        "reliability table (10 quantile bins)",
                        "top-k recall k=5/10/20 per seed pool",
                        "policy regret at budgets B=5/10/20 "
                        "(retention ranked by score, J = sum of "
                        "exact-replay U* retained)",
                        "OOD: hazard variant, water slot, agent "
                        "role, horizon"],
        }, f, indent=2)

    if a.selfcheck:
        from experiments.song_grammar.exp_ue1_utility_estimator import (
            collect as collect_orig)
        X0, y0 = collect_orig(range(0, 2))
        X1, y1, _ = collect_meta(range(0, 2))
        assert np.allclose(X0, X1) and np.allclose(y0, y1), \
            "collect_meta diverged from frozen UE1 collect()"
        print("[selfcheck] collect_meta == UE1 collect: OK")

    print(f"[collect] train seeds {a.train_seeds} "
          f"e={a.episodes} (robust)")
    Xtr, ytr, mtr = collect_meta(range(*a.train_seeds), a.episodes)
    print(f"[collect] test seeds {a.test_seeds} e={a.episodes} "
          f"(robust)")
    Xte, yte, mte = collect_meta(range(*a.test_seeds), a.episodes)
    print(f"[collect] test seeds {a.test_seeds} e={a.episodes} "
          f"(fragile oracle)")
    Xfr, yfr, mfr = collect_meta(range(*a.test_seeds), a.episodes,
                                 role_name="fragile")
    print(f"[collect] test seeds {a.test_seeds} "
          f"e={a.horizon_episodes} (horizon)")
    Xho, yho, mho = collect_meta(range(*a.test_seeds),
                                 a.horizon_episodes)

    W = fit_ridge(Xtr, ytr)
    results: Dict[str, Any] = {
        "config": {"train_seeds": a.train_seeds,
                   "test_seeds": a.test_seeds,
                   "episodes": a.episodes,
                   "horizon_episodes": a.horizon_episodes,
                   "ridge_lambda": RIDGE_LAMBDA,
                   "n_train": int(len(ytr)),
                   "weights": W.tolist()}}

    # 1-5: in-distribution held-out (the rho = 0.989 split)
    results["held_out"] = full_block(Xte @ W, yte, mte)
    show("HELD-OUT (train robust -> test robust, the paper split)",
         results["held_out"])

    # 6a: OOD hazard texture (train variant==0 only -> eval variant>0)
    tr_v0 = np.array([m["variant"] == 0 for m in mtr])
    te_v1 = np.array([m["variant"] > 0 for m in mte])
    ood: Dict[str, Any] = {}
    if tr_v0.sum() >= 30 and te_v1.sum() >= 10:
        Wv = fit_ridge(Xtr[tr_v0], ytr[tr_v0])
        sub = [m for m, k in zip(mte, te_v1) if k]
        ood["layout_hazard_variant"] = {
            "design": "train on variant==0 worlds only "
                      f"(n={int(tr_v0.sum())}), eval held-out "
                      f"variant>0 (n={int(te_v1.sum())})",
            **full_block(Xte[te_v1] @ Wv, yte[te_v1], sub)}
        show("OOD layout: hazard variant (train v==0 -> test v>0)",
             ood["layout_hazard_variant"])
    else:
        ood["layout_hazard_variant"] = {
            "skipped": f"too few samples (train v0 "
                       f"{int(tr_v0.sum())}, test v>0 "
                       f"{int(te_v1.sum())})"}
        print("\n[skip] hazard-variant OOD: too few samples")

    # 6b: OOD water slot (train widx in {0,1} -> eval widx == 2)
    tr_w = np.array([m["widx"] in (0, 1) for m in mtr])
    te_w = np.array([m["widx"] == 2 for m in mte])
    if tr_w.sum() >= 30 and te_w.sum() >= 10:
        Ww = fit_ridge(Xtr[tr_w], ytr[tr_w])
        sub = [m for m, k in zip(mte, te_w) if k]
        ood["layout_water_slot"] = {
            "design": "train on widx 0/1 "
                      f"(n={int(tr_w.sum())}), eval held-out "
                      f"widx==2 (n={int(te_w.sum())})",
            **full_block(Xte[te_w] @ Ww, yte[te_w], sub)}
        show("OOD layout: water slot (train widx 0/1 -> test "
             "widx 2)", ood["layout_water_slot"])
    else:
        ood["layout_water_slot"] = {
            "skipped": f"too few samples (train {int(tr_w.sum())}, "
                       f"test widx2 {int(te_w.sum())})"}
        print("\n[skip] water-slot OOD: too few samples")

    # 6c: OOD role (train robust -> eval fragile oracle)
    ood["agent_role"] = {
        "design": "estimator trained on robust-role U*, evaluated "
                  "against fragile-role U* on the same held-out "
                  "streams (features are frozen and role-agnostic)",
        **full_block(Xfr @ W, yfr, mfr)}
    show("OOD role: train robust -> eval fragile U*",
         ood["agent_role"])

    # 6d: OOD horizon (train t < episodes -> eval late episodes)
    hor: Dict[str, Any] = {
        "design": f"train horizon t<{a.episodes}; held-out streams "
                  f"run to t={a.horizon_episodes} and sliced"}
    for lo, hi, tag in ((0, a.episodes, "in_horizon"),
                        (a.episodes, a.horizon_episodes,
                         "beyond_horizon")):
        keep = np.array([lo <= m["t"] < hi for m in mho])
        if keep.sum() < 10:
            hor[tag] = {"skipped": f"n={int(keep.sum())}"}
            continue
        sub = [m for m, k in zip(mho, keep) if k]
        hor[tag] = full_block(Xho[keep] @ W, yho[keep], sub)
        show(f"OOD horizon: t in [{lo},{hi})", hor[tag])
    ood["horizon"] = hor
    results["ood"] = ood

    # persist raw pairs for downstream analysis / figures
    pairs = os.path.join(a.out, "ue1_eval_pairs.jsonl")
    with open(pairs, "w") as f:
        for tag, (Xs, ys, ms) in (("test_robust", (Xte, yte, mte)),
                                  ("test_fragile", (Xfr, yfr, mfr)),
                                  ("test_horizon", (Xho, yho, mho))):
            preds = Xs @ W
            for p, u, m in zip(preds, ys, ms):
                f.write(json.dumps({"slice": tag, "u_hat": float(p),
                                    "u_star": float(u), **m}) + "\n")

    with open(os.path.join(a.out, "ue1_eval_results.json"),
              "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {a.out}/ue1_eval_results.json and "
          f"{pairs}")


if __name__ == "__main__":
    main()
