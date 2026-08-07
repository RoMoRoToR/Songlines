"""Package A / Part 2 — direct contention interventions on the base sweep.

Reviewer claim: "identifying target contention as the principal cost
requires a DIRECT intervention on contention, not only correlational
cadence sweeps."  We intervene on the two axes separately — evidence
quality and coordination — a 2x2 plus a scarcity release, on the small
grid (N ∈ {3,5} under scarcity), peer architecture, K ∈ {1,4}:

  mode                 evidence            coordination        scarcity
  ------------------   -----------------   -----------------   --------
  baseline             normal peer memory  none                N > T
  unique_oracle    (a) oracle (agent is    perfect (oracle     N > T
                       told its target)    unique assignment)
  no_scarcity      (b) normal peer memory  none                T = N
  oracle_R_no_coord(c) oracle R (query =   none                N > T
                       all GT waters)
  unique_noisy     (d) normal peer memory, perfect (oracle     N > T
                       filtered to the     unique assignment)
                       assigned target

Unique assignment: an oracle greedily assigns distinct waters to the
min(N,T) nearest agents (by spawn distance, deterministic).  Unassigned
agents receive an empty candidate set — exclusivity IS the intervention;
the structural success ceiling is min(N,T)/N and is reported alongside.

═══════════════════════════════════════════════════════════════════════
REGISTERED EXPECTATIONS — formulated BEFORE the first smoke run
(2026-08-07).  Identical seeds across all modes.

  F1  unique_oracle reaches its structural ceiling (success ≈ min(N,T)/N,
      p(C*|M*) ≈ 1) with duplicate-commitment ticks ≈ 0 — upper bound.

  F2  no_scarcity recovers p(C*|M*) to near 1 even at K=1 WITHOUT any
      coordination: if contention is the principal cost, removing
      scarcity removes the collapse.  Success rises toward 1 (bounded
      by discovery, not contention).

  F3  oracle_R_no_coord does NOT fix the collapse: perfect evidence with
      no coordination leaves p(C*|M*) at/below baseline under scarcity
      (everyone materialises instantly toward the same nearest waters →
      duplicate ticks HIGHER than baseline).  Its mean_t_succ for the
      winners drops (faster discovery), but success_rate stays near the
      scarcity-limited baseline level.

  F4  unique_noisy recovers p(C*|M*) toward the assigned ceiling
      DESPITE unchanged evidence accuracy: coordination alone (with the
      same noisy memory) restores foreign-evidence completion —
      p(C*|W*_soft) for assigned agents rises vs baseline at K=1.
      Together F3+F4 dissociate the axes: coordination is binding,
      evidence is not.
═══════════════════════════════════════════════════════════════════════

Usage::

    PYTHONPATH=. .venv/bin/python experiments/coordination/exp_contention_interventions.py \\
        --seeds 3 --workers 8 --out_dir tmp/contention_interv_smoke   # smoke
    PYTHONPATH=. .venv/bin/python experiments/coordination/exp_contention_interventions.py \\
        --seeds 20 --workers 16 --out_dir tmp/cluster/contention_interv  # full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.big_experiment.env_factory import build_env
from experiments.big_experiment.memory_factory import build_memory
from experiments.big_experiment.runner import RunConfig
from experiments.warp.warp_runner import run_warp_episode

from experiments.coordination.coord_protocols import NullProtocol, _close
from experiments.coordination.coord_runner import t_censored

MODES = ["baseline", "unique_oracle", "no_scarcity",
         "oracle_R_no_coord", "unique_noisy"]
NM_CELLS = [(3, 2), (5, 3)]              # small grid, scarcity (N in {3,5})
LAYOUTS = ["random", "asymmetric"]
KS = [1, 4]
HAZARD = 0.05
STEP_LIMIT = 120

CSV_FIELDS = [
    "mode", "n_agents", "n_waters", "n_waters_effective", "layout",
    "broadcast_every_k", "seed", "ceiling",
    "success_rate", "n_succeeded", "mean_t_succ", "t_cens", "ticks_played",
    "m_star_rate", "c_star_rate", "p_C_given_M",
    "warp_share_soft", "p_C_given_W_soft", "n_success_no_lock",
    "dup_lock_ticks", "dup_lock_events",
    "scarcity", "hazard_density",
]


# ─────────────────────────────────────── oracle unique assignment


def unique_assignment(agent_ids: List[str],
                      spawn_xy: Dict[str, Tuple[int, int]],
                      waters: List[Tuple[int, int]]
                      ) -> Dict[str, Optional[Tuple[int, int]]]:
    """Greedy oracle assignment: sorted (distance, agent, water) pairs,
    each agent and each water used at most once.  Deterministic."""
    pairs = sorted(
        (abs(p[0] - w[0]) + abs(p[1] - w[1]), aid, w)
        for aid, p in spawn_xy.items() for w in waters)
    assigned: Dict[str, Optional[Tuple[int, int]]] = {
        aid: None for aid in agent_ids}
    used = set()
    for _d, aid, w in pairs:
        if assigned[aid] is not None or w in used:
            continue
        assigned[aid] = w
        used.add(w)
    return assigned


# ─────────────────────────────────────── one episode per mode


def run_intervention_episode(mode: str, n: int, m: int, layout: str,
                             k: int, seed: int) -> Dict[str, Any]:
    m_eff = n if mode == "no_scarcity" else m
    cfg = RunConfig(
        n_agents=n, n_waters=m_eff, layout=layout, architecture="peer",
        broadcast_every_k=k, hazard_density=HAZARD, seed=seed,
        step_limit=STEP_LIMIT,
    )
    built = build_env(
        n_agents=cfg.n_agents, n_waters=cfg.n_waters, layout=cfg.layout,
        hazard_density=cfg.hazard_density, seed=cfg.seed,
        step_limit=cfg.step_limit,
    )
    env_id = f"interv_{mode}_{cfg.as_tag()}"
    memory = build_memory("peer", built.agent_ids, env_id,
                          broadcast_every_k=k)
    waters = built.water_positions

    # ── intervention: wrap the memory adapter's query (instance-level,
    #    nothing in memory_factory is modified) ─────────────────────
    base_query = memory.query
    if mode in ("unique_oracle", "unique_noisy"):
        spawn_xy = {aid: (built.env.agents[aid].x, built.env.agents[aid].y)
                    for aid in built.agent_ids}
        assigned = unique_assignment(built.agent_ids, spawn_xy, waters)
        if mode == "unique_oracle":
            # perfect coordination + agent is told its target's location
            def wrapped(aid, _a=assigned):
                t = _a.get(aid)
                return [(float(t[0]), float(t[1]))] if t is not None else []
        else:
            # perfect coordination, unchanged evidence: the agent must
            # still DISCOVER its assigned target through normal memory
            def wrapped(aid, _a=assigned, _q=base_query):
                t = _a.get(aid)
                if t is None:
                    return []
                return [c for c in _q(aid) if _close(c, t)]
        memory.query = wrapped
    elif mode == "oracle_R_no_coord":
        # perfect evidence, zero coordination
        def wrapped(aid, _w=waters):
            return [(float(w[0]), float(w[1])) for w in _w]
        memory.query = wrapped
    # baseline / no_scarcity: untouched pipeline

    proto = NullProtocol(built.agent_ids, k)   # measurement only
    metrics, log = run_warp_episode(
        built.env, built.agent_ids, waters, memory,
        step_limit=cfg.step_limit, variant_tag="peer", warp_drive=proto,
    )
    ceiling = min(n, m_eff) / n
    metrics.update({
        "mode": mode,
        "n_agents": n, "n_waters": m, "n_waters_effective": m_eff,
        "layout": layout, "broadcast_every_k": k,
        "hazard_density": HAZARD, "seed": seed,
        "scarcity": n / m_eff, "ceiling": ceiling,
        "t_cens": t_censored(log, built.agent_ids, cfg.step_limit),
        # successes with no lock ever held on the reached cell — explains
        # p_C_given_M > 1 under heavy query filtering (exploration walk-ins)
        "n_success_no_lock": sum(log.success_without_lock.values()),
    })
    return metrics


def _one_job(job: Tuple) -> Dict[str, Any]:
    return run_intervention_episode(*job)


# ─────────────────────────────────────── aggregation / table


def _mean_ci(vals: List[float], n_boot: int = 2000) -> Tuple[float, float, float]:
    vals = [v for v in vals
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return (float("nan"),) * 3
    rng = np.random.default_rng(7)
    arr = np.array(vals, dtype=float)
    boots = [float(np.mean(rng.choice(arr, len(arr)))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in KS:
        for mode in MODES:
            cell = [r for r in rows
                    if r["broadcast_every_k"] == k and r["mode"] == mode]
            if not cell:
                continue
            out[f"k{k}/{mode}"] = {
                "n": len(cell),
                "success": _mean_ci([r["success_rate"] for r in cell]),
                "ceiling": statistics.mean(r["ceiling"] for r in cell),
                "t_cens": _mean_ci([r["t_cens"] for r in cell]),
                "p_C_given_M": _mean_ci([r["p_C_given_M"] for r in cell]),
                "p_C_given_W_soft": _mean_ci(
                    [r["p_C_given_W_soft"] for r in cell]),
                "warp_share_soft": _mean_ci(
                    [r["warp_share_soft"] for r in cell]),
                "dup_lock_ticks": _mean_ci([r["dup_lock_ticks"] for r in cell]),
            }
    return out


def print_table(summary: Dict[str, Any]) -> None:
    hdr = (f"{'K':>2} {'mode':<18} {'succ':>6} {'ceil':>5} {'t_cens':>7} "
           f"{'p(C|M)':>7} {'p(C|W)':>7} {'Wshare':>7} {'dup_tk':>7}")
    print(hdr)
    print("-" * len(hdr))
    for k in KS:
        for mode in MODES:
            s = summary.get(f"k{k}/{mode}")
            if s is None:
                continue

            def fmt(v, w=7, p=3):
                return f"{'nan':>{w}}" if math.isnan(v) else f"{v:>{w}.{p}f}"

            print(f"{k:>2} {mode:<18} {s['success'][0]:>6.3f} "
                  f"{s['ceiling']:>5.2f} {s['t_cens'][0]:>7.1f} "
                  f"{fmt(s['p_C_given_M'][0])} "
                  f"{fmt(s['p_C_given_W_soft'][0])} "
                  f"{fmt(s['warp_share_soft'][0])} "
                  f"{s['dup_lock_ticks'][0]:>7.1f}")
        print("-" * len(hdr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--out_dir", default="tmp/contention_interv_smoke")
    parser.add_argument("--modes", nargs="*", default=MODES)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = [(mode, n, m, layout, k, seed)
            for (n, m) in NM_CELLS
            for layout in LAYOUTS
            for seed in range(args.seeds)      # identical seeds across modes
            for k in KS
            for mode in args.modes]
    print(f"Contention interventions: {len(jobs)} episodes "
          f"({len(NM_CELLS)} cells x {len(LAYOUTS)} layouts x "
          f"{args.seeds} seeds x K{KS} x {len(args.modes)} modes), "
          f"workers={args.workers}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(jobs)}  elapsed={el:.1f}s")
    elapsed = time.time() - t0
    print(f"done: {len(rows)}/{len(jobs)} episodes in {elapsed:.1f}s "
          f"({elapsed / max(1, len(rows)):.2f}s/ep)\n")

    csv_path = os.path.join(args.out_dir, "interv_runs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})

    summary = summarize(rows)
    print_table(summary)

    with open(os.path.join(args.out_dir, "interv_summary.json"), "w") as f:
        json.dump({"summary": summary, "n_rows": len(rows),
                   "elapsed_s": elapsed, "modes": args.modes,
                   "seeds": args.seeds}, f, indent=2, default=str)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {os.path.join(args.out_dir, 'interv_summary.json')}")


if __name__ == "__main__":
    main()
