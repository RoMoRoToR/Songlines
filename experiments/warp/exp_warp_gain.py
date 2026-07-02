"""W1 — counterfactual warp-gain sweep (design §8/W1, H-W1, H-W2, H-W5).

For every (N, M, layout, architecture, seed) cell the episode is run
TWICE with the same deterministic seed: once full, once with foreign
evidence masked at merge (``mask_foreign=True``).  The per-episode
difference

    WG = t_cens(masked) − t_cens(full)      (censored time-to-success)

is the causally attributed warp gain: same seed, same planner, only the
foreign contribution zeroed.

Hypotheses evaluated on the pooled logs:

  H-W1  P(C*|W*) < P(C*|M*, own) in scarcity; gap smaller for CSM than
        for fixed-K peer.        (episode-cluster bootstrap CI)
  H-W2  Warp gain changes sign across (layout, scarcity rho).
  H-W5  co-locked pressure predicts warp collision (Spearman).

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_gain.py \\
        --seeds 20 --workers 8 --out_dir tmp/warp/w1_gain
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.big_experiment.runner import RunConfig
from experiments.warp.warp_runner import run_one_config_warp

# (N agents, M waters): three scarcity cells from the design plus one
# abundance cell (rho = 1.0) and one hard-scarcity cell (rho = 3.0) so
# the H-W2 sign map has an actual rho axis to vary over.
NM_CELLS = [(3, 2), (5, 3), (8, 5), (4, 4), (6, 2)]
LAYOUTS = ["symmetric", "asymmetric", "random"]
# Architectures: warp gain is only defined where foreign evidence exists.
ARCHS = [
    ("peer", 2),    # fast share
    ("peer", 8),    # sweet-spot cadence
    ("csm", 8),     # trust × staleness gate
    ("shared", -1),  # maximal warp share
]
HAZARD = 0.05
STEP_LIMIT = 120


def _one_job(job: Tuple) -> Dict[str, Any]:
    (n, m, layout, arch, k, seed, mask) = job
    cfg = RunConfig(
        n_agents=n, n_waters=m, layout=layout, architecture=arch,
        broadcast_every_k=k, hazard_density=HAZARD, seed=seed,
        step_limit=STEP_LIMIT,
    )
    metrics, log = run_one_config_warp(cfg, mask_foreign=mask)
    # censored time-to-success: failures count as step_limit
    t_cens = float(np.mean([
        (t if t is not None else STEP_LIMIT)
        for t in log.first_success_tick.values()
    ]))
    metrics["t_cens"] = t_cens
    metrics["events"] = [e.to_dict() for e in log.m_star_events()]
    return metrics


def _bootstrap_rate(events: List[Dict], key_num: str,
                    n_boot: int = 4000, seed: int = 0
                    ) -> Tuple[float, float, float]:
    """Episode-cluster bootstrap of a completion rate over events.

    ``events`` carry an ``_episode`` cluster id.  Returns (mean, lo, hi).
    """
    if not events:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    clusters: Dict[str, List[int]] = {}
    for e in events:
        clusters.setdefault(e["_episode"], []).append(int(e[key_num]))
    cluster_vals = list(clusters.values())
    point = float(np.mean([v for vals in cluster_vals for v in vals]))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cluster_vals), len(cluster_vals))
        pooled = [v for i in idx for v in cluster_vals[i]]
        if pooled:
            stats.append(np.mean(pooled))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def analyze(rows: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    from scipy.stats import spearmanr

    # ── pair up full/masked per cell+seed for warp gain ───────────
    by_key: Dict[Tuple, Dict[bool, Dict]] = {}
    for r in rows:
        key = (r["architecture"], r["broadcast_every_k"], r["layout"],
               r["n_agents"], r["n_waters"], r["seed"])
        by_key.setdefault(key, {})[bool(r["mask_foreign"])] = r

    gain_rows = []
    for (arch, k, layout, n, m, seed), pair in by_key.items():
        if True not in pair or False not in pair:
            continue
        full, masked = pair[False], pair[True]
        gain_rows.append({
            "architecture": arch, "k": k, "layout": layout,
            "n_agents": n, "n_waters": m, "rho": n / m, "seed": seed,
            "wg_time": masked["t_cens"] - full["t_cens"],   # >0 → warp helps
            "wg_succ": full["success_rate"] - masked["success_rate"],
            "t_full": full["t_cens"], "t_masked": masked["t_cens"],
        })

    # ── H-W2: sign map over (arch, layout, rho) ───────────────────
    sign_map: Dict[str, Dict[str, Any]] = {}
    for arch, k in ARCHS:
        arch_key = f"{arch}-k{k}"
        for layout in LAYOUTS:
            for n, m in NM_CELLS:
                cell = [g for g in gain_rows
                        if g["architecture"] == arch and g["k"] == k
                        and g["layout"] == layout
                        and g["n_agents"] == n and g["n_waters"] == m]
                if not cell:
                    continue
                wg = np.array([c["wg_time"] for c in cell])
                rng = np.random.default_rng(1)
                boots = [np.mean(rng.choice(wg, len(wg))) for _ in range(4000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                sign_map[f"{arch_key}|{layout}|N{n}M{m}"] = {
                    "rho": round(n / m, 2),
                    "wg_time_mean": float(np.mean(wg)),
                    "ci": [float(lo), float(hi)],
                    "sign": ("+" if lo > 0 else ("-" if hi < 0 else "0")),
                    "n_seeds": len(cell),
                }

    # ── H-W1: completion strata over pooled events ────────────────
    strata: Dict[str, Any] = {}
    for arch, k in ARCHS:
        evts = []
        for r in rows:
            if r["mask_foreign"] or r["architecture"] != arch \
                    or r["broadcast_every_k"] != k:
                continue
            # only scarcity episodes (N > M) per H-W1 statement
            if r["n_agents"] <= r["n_waters"]:
                continue
            for e in r["events"]:
                e = dict(e)
                e["_episode"] = r["tag"]
                e["_completed"] = int(e["completed"])
                evts.append(e)
        warp_soft = [e for e in evts if e["w_star_soft"]]
        own = [e for e in evts if not e["w_star_soft"]]
        strict = [e for e in evts if e["w_star_strict"]]
        strata[f"{arch}-k{k}"] = {
            "n_events": len(evts),
            "n_warp_soft": len(warp_soft), "n_warp_strict": len(strict),
            "p_C_given_W_soft": _bootstrap_rate(warp_soft, "_completed"),
            "p_C_given_W_strict": _bootstrap_rate(strict, "_completed"),
            "p_C_given_M_own": _bootstrap_rate(own, "_completed"),
        }

    # ── H-W5: co-locked pressure vs collision ─────────────────────
    warp_events_all = []
    for r in rows:
        if r["mask_foreign"]:
            continue
        for e in r["events"]:
            if e["w_star_soft"]:
                warp_events_all.append(
                    (e["co_locked"], 0 if e["completed"] else 1,
                     r["architecture"]))
    hw5: Dict[str, Any] = {"n_warp_events": len(warp_events_all)}
    if len(warp_events_all) >= 10:
        co = [w[0] for w in warp_events_all]
        fail = [w[1] for w in warp_events_all]
        rho_s, p = spearmanr(co, fail)
        hw5.update({"spearman_co_locked_vs_failure": float(rho_s),
                    "p_value": float(p)})

    summary = {"strata_H_W1": strata, "sign_map_H_W2": sign_map,
               "H_W5": hw5, "n_gain_pairs": len(gain_rows)}

    with open(os.path.join(out_dir, "w1_gain_rows.json"), "w") as f:
        json.dump(gain_rows, f, indent=1)
    with open(os.path.join(out_dir, "w1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", default="tmp/warp/w1_gain")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = []
    for (n, m) in NM_CELLS:
        for layout in LAYOUTS:
            for arch, k in ARCHS:
                for seed in range(args.seeds):
                    for mask in (False, True):
                        jobs.append((n, m, layout, arch, k, seed, mask))
    print(f"W1 sweep: {len(jobs)} episodes "
          f"({len(NM_CELLS)} NM × {len(LAYOUTS)} layouts × "
          f"{len(ARCHS)} archs × {args.seeds} seeds × 2 masks)")

    rows: List[Dict[str, Any]] = []
    raw_path = os.path.join(args.out_dir, "w1_rows.jsonl")
    with open(raw_path, "w") as raw, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one_job, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            rows.append(r)
            raw.write(json.dumps(r, default=str) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} done")

    summary = analyze(rows, args.out_dir)

    print("\n── H-W1 strata (scarcity, full runs) ──────────────────")
    for name, s in summary["strata_H_W1"].items():
        pw = s["p_C_given_W_soft"]
        po = s["p_C_given_M_own"]
        print(f"  {name:<12} P(C*|W*soft)={pw[0]:.3f} [{pw[1]:.3f},{pw[2]:.3f}] "
              f"(n={s['n_warp_soft']})  vs  "
              f"P(C*|M*,own)={po[0]:.3f} [{po[1]:.3f},{po[2]:.3f}] "
              f"(n={s['n_events'] - s['n_warp_soft']})")

    print("\n── H-W2 sign map (wg_time>0 → warp helps) ─────────────")
    for key, v in sorted(summary["sign_map_H_W2"].items()):
        print(f"  {key:<38} rho={v['rho']:<5} WG={v['wg_time_mean']:+7.2f} "
              f"[{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] sign={v['sign']}")

    print("\n── H-W5 ────────────────────────────────────────────────")
    print(f"  {summary['H_W5']}")
    print(f"\nSaved: {args.out_dir}/w1_summary.json")


if __name__ == "__main__":
    main()
