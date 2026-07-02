"""W3 — Warp Drive protocol evaluation (design §7/§8, H-W4).

Three parts:

  A. Decentralisation check ('remove the runtime'): the protocol driven
     by direct per-slot calls — no runner, no env — must produce the
     same retraction decision from a slot's own inbox alone.
  B. Misled-A anchor: on the exp_4way_walk scenario the baseline agent-A
     wastes ticks travelling to (3,8) which agent-C claims first.  With
     Warp Drive, C's reservation retracts A's lock within ~K ticks
     (finite, small rollback latency) and A reprograms.
  C. Cadence sweep: peer at K ∈ {1, 2, 4} with and without Warp Drive,
     plus peer at K=8 as the C-collapse reference.  H-W4: the protocol
     recovers ≥ 50% of the fast-share C-collapse at K ∈ {1, 2}.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_drive.py \\
        --seeds 20 --workers 8 --out_dir tmp/warp/w3_drive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.big_experiment.memory_factory import build_memory
from experiments.big_experiment.runner import RunConfig
from experiments.warp.exp_warp_anchor import (
    AGENT_SPEC, BROADCAST_K, WATER_CELLS, build_anchor_env,
)
from experiments.warp.warp_drive import Reservation, WarpDriveProtocol
from experiments.warp.warp_runner import run_one_config_warp, run_warp_episode

NM_CELLS = [(3, 2), (5, 3), (8, 5)]
LAYOUTS = ["random", "asymmetric"]
KS_FAST = [1, 2, 4]
K_REF = 8
HAZARD = 0.05
STEP_LIMIT = 120


# ───────────────────────────────────── A. remove-the-runtime check


def check_decentralised() -> bool:
    """Drive the protocol with direct calls only — no runner, no env."""
    proto = WarpDriveProtocol(["agent-A", "agent-C"], broadcast_every_k=1)
    # C locks (3,8) at distance 2; A locks the same cell at distance 12.
    proto.on_lock("agent-C", (3, 8), tick=7, distance=2)
    proto.on_lock("agent-A", (3, 8), tick=7, distance=12)
    # tick 7 is a broadcast tick for k=1 → reservations cross over;
    # A's slot must retract using only its own inbox.
    retr = proto.on_tick(7, {"agent-A": (3, 8), "agent-C": (3, 8)})
    a_retracted = "agent-A" in retr and "agent-C" not in retr
    # A's next query must suppress (3,8) (backoff + C's reservation)…
    filtered = proto.filter_targets("agent-A", [(3.0, 8.0), (8.0, 7.0)], 8)
    suppression_ok = (3.0, 8.0) not in filtered and (8.0, 7.0) in filtered
    # …and no oscillation: C keeps its lock on the next tick.
    retr2 = proto.on_tick(8, {"agent-A": None, "agent-C": (3, 8)})
    stable = len(retr2) == 0
    return a_retracted and suppression_ok and stable


# ───────────────────────────────────── B. misled-A anchor


def run_anchor(with_wd: bool) -> Dict[str, Any]:
    env = build_anchor_env()
    agent_ids = [s[0] for s in AGENT_SPEC]
    memory = build_memory("peer", agent_ids, "warp-drive-anchor",
                          broadcast_every_k=BROADCAST_K)
    wd = (WarpDriveProtocol(agent_ids, BROADCAST_K) if with_wd else None)
    metrics, log = run_warp_episode(
        env, agent_ids, WATER_CELLS, memory,
        step_limit=60, variant_tag="peer", warp_drive=wd,
    )
    a_events = [e for e in log.events if e.agent_id == "agent-A"]
    a_38 = [e for e in a_events if tuple(e.target_xy) == (3, 8)]
    a_completed = [e for e in a_events if e.completed]
    return {
        "with_wd": with_wd,
        "metrics": {k: v for k, v in metrics.items()
                    if not isinstance(v, (list, dict))},
        "A_lock_38_tick": a_38[0].tick if a_38 else None,
        "A_38_retracted": a_38[0].retracted if a_38 else None,
        "A_38_rollback_latency": a_38[0].rollback_latency if a_38 else None,
        "A_38_dropped_tick": a_38[0].dropped_tick if a_38 else None,
        "A_final_target": (list(a_completed[0].target_xy)
                           if a_completed else None),
        "A_t_succ": log.first_success_tick.get("agent-A"),
        "retraction_log": wd.retraction_log if wd else [],
    }


# ───────────────────────────────────── C. cadence sweep


def _one_job(job: Tuple) -> Dict[str, Any]:
    (n, m, layout, k, seed, with_wd) = job
    cfg = RunConfig(
        n_agents=n, n_waters=m, layout=layout, architecture="peer",
        broadcast_every_k=k, hazard_density=HAZARD, seed=seed,
        step_limit=STEP_LIMIT,
    )
    factory = ((lambda agent_ids, kk: WarpDriveProtocol(agent_ids, kk))
               if with_wd else None)
    metrics, log = run_one_config_warp(cfg, warp_drive_factory=factory)
    metrics["with_wd"] = with_wd
    # Structural ceiling of P(C*|M*) under scarcity: with M waters and
    # m agents holding an M*-lock at most min(m, M) can complete.  The
    # recoverable part of the C-collapse is (ceiling − base), not
    # (K8-reference − base): no protocol conjures extra water.
    m_star_count = round(metrics["m_star_rate"] * n)
    metrics["p_C_given_M_ceiling"] = (
        min(m_star_count, m) / m_star_count if m_star_count > 0
        else float("nan"))
    metrics.pop("events", None)
    return metrics


def analyze_sweep(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def cell(k: int, wd: bool) -> List[Dict]:
        return [r for r in rows
                if r["broadcast_every_k"] == k and r["with_wd"] == wd]

    def mean_ci(vals: List[float], n_boot: int = 4000):
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return (float("nan"),) * 3
        rng = np.random.default_rng(2)
        arr = np.array(vals)
        boots = [np.mean(rng.choice(arr, len(arr))) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(np.mean(arr)), float(lo), float(hi)

    ref = mean_ci([r["p_C_given_M"] for r in cell(K_REF, False)])
    out: Dict[str, Any] = {"p_C_given_M_ref_k8": ref}
    for k in KS_FAST:
        base = mean_ci([r["p_C_given_M"] for r in cell(k, False)])
        wd = mean_ci([r["p_C_given_M"] for r in cell(k, True)])
        ceiling = mean_ci([r["p_C_given_M_ceiling"] for r in cell(k, False)])
        collapse_rec = ceiling[0] - base[0]      # recoverable collapse
        recovered = wd[0] - base[0]
        recovery = (recovered / collapse_rec if collapse_rec > 1e-9
                    else float("nan"))
        lat = [r["wd_mean_rollback_latency"] for r in cell(k, True)
               if r.get("wd_n_retractions", 0) > 0]
        out[f"k{k}"] = {
            "base": base, "warp_drive": wd, "ceiling": ceiling,
            "collapse_vs_k8": round(ref[0] - base[0], 4),
            "recoverable_collapse": round(collapse_rec, 4),
            "recovered": round(recovered, 4),
            "recovery_share": (round(recovery, 3)
                               if not np.isnan(recovery) else None),
            "mean_rollback_latency": (float(np.mean(lat)) if lat
                                      else float("nan")),
            "succ_base": mean_ci([r["success_rate"] for r in cell(k, False)]),
            "succ_wd": mean_ci([r["success_rate"] for r in cell(k, True)]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", default="tmp/warp/w3_drive")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("A. remove-the-runtime decentralisation check")
    decentralised_ok = check_decentralised()
    print(f"   [{'PASS' if decentralised_ok else 'FAIL'}] direct-call protocol"
          " decisions from own inbox only\n")

    print("B. misled-A anchor (peer K=4)")
    anchor = {"baseline": run_anchor(False), "warp_drive": run_anchor(True)}
    for name, a in anchor.items():
        print(f"   {name:<11} A_lock(3,8)@{a['A_lock_38_tick']} "
              f"retracted={a['A_38_retracted']} "
              f"rollback_latency={a['A_38_rollback_latency']} "
              f"dropped@{a['A_38_dropped_tick']} "
              f"final={a['A_final_target']} t_succ={a['A_t_succ']}")
    wd_a = anchor["warp_drive"]
    # The pathology is resolved either way:
    #   cure       — the (3,8) lock is retracted within ≤ 2K ticks, or
    #   prevention — C's reservation arrives before A ever locks (3,8)
    #                (stronger: zero misled travel, latency ≡ 0).
    cured = (wd_a["A_38_retracted"] is True
             and wd_a["A_38_rollback_latency"] is not None
             and wd_a["A_38_rollback_latency"] <= 2 * BROADCAST_K)
    prevented = wd_a["A_lock_38_tick"] is None
    anchor_ok = ((cured or prevented)
                 and wd_a["A_final_target"] is not None
                 and tuple(wd_a["A_final_target"]) != (3, 8)
                 and wd_a["A_t_succ"] is not None)
    mode = "prevented" if prevented else ("cured" if cured else "unresolved")
    print(f"   [{'PASS' if anchor_ok else 'FAIL'}] misled-A {mode}; "
          f"A completes on {wd_a['A_final_target']}\n")

    print(f"C. cadence sweep: K∈{KS_FAST} × {NM_CELLS} × {LAYOUTS} × "
          f"{args.seeds} seeds × WD on/off + K={K_REF} reference")
    jobs = []
    for (n, m) in NM_CELLS:
        for layout in LAYOUTS:
            for seed in range(args.seeds):
                for k in KS_FAST:
                    jobs.append((n, m, layout, k, seed, False))
                    jobs.append((n, m, layout, k, seed, True))
                jobs.append((n, m, layout, K_REF, seed, False))
    print(f"   {len(jobs)} episodes")

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i + 1) % 200 == 0:
                print(f"   {i + 1}/{len(jobs)} done")

    summary = analyze_sweep(rows)
    print("\n── H-W4: C-collapse recovery ───────────────────────────")
    print(f"   P(C*|M*) reference @K8: {summary['p_C_given_M_ref_k8'][0]:.3f}")
    for k in KS_FAST:
        s = summary[f"k{k}"]
        print(f"   K={k}: base={s['base'][0]:.3f} "
              f"[{s['base'][1]:.3f},{s['base'][2]:.3f}]  "
              f"WD={s['warp_drive'][0]:.3f} "
              f"[{s['warp_drive'][1]:.3f},{s['warp_drive'][2]:.3f}]  "
              f"ceiling={s['ceiling'][0]:.3f}  "
              f"recovery={s['recovery_share']}  "
              f"rollback_lat={s['mean_rollback_latency']:.1f}")

    rec12 = [summary[f"k{k}"]["recovery_share"] for k in (1, 2)]
    hw4_ok = all(r is not None and r >= 0.5 for r in rec12)

    verdict = {
        "decentralised_direct_calls": decentralised_ok,
        "anchor_misled_A_resolved": anchor_ok,
        "H_W4_recovery_ge_50pct_at_K12": hw4_ok,
        "recovery_shares": {f"k{k}": summary[f"k{k}"]["recovery_share"]
                            for k in KS_FAST},
    }
    with open(os.path.join(args.out_dir, "w3_results.json"), "w") as f:
        json.dump({"anchor": anchor, "sweep_summary": summary,
                   "verdict": verdict, "rows": rows},
                  f, indent=2, default=str)

    print("\n" + "=" * 60)
    for k, v in verdict.items():
        if isinstance(v, bool):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"Saved: {args.out_dir}/w3_results.json")


if __name__ == "__main__":
    main()
