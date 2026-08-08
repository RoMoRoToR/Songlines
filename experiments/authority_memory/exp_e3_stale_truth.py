"""E3 — Stale Truth -> False Belief: the third critical experiment of
the Memory-as-Authority-Protocol frontier
(``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/04_EXPERIMENTS.md`` §E3).

The world silently changes (a bridge that was open closes) at
``t_invalidated``; no agent is told directly.  A certificate stamped
"bridge open" keeps circulating.  How long does it keep durable
action authority after the world stopped agreeing with it?

04_EXPERIMENTS.md lists seven baseline architectures (raw history /
vector RAG / recency-only RAG / shared graph / trust-only /
staleness-only / full protocol).  For THIS metric (Revocation
Latency, stale-authority area) their differences from each other do
not matter --- only whether they have ANY staleness mechanism at all.
That collapses the seven into three distinct behaviour classes, and
Sprint 4's job is to exercise the REAL revocation mechanism (Theorem
2, ``authority_memory/revocation.py``) against exactly those three,
not to build a seven-arm bake-off:

  1. never_expires    -- raw history / vector RAG / shared graph
                         without a gate. evidence_score is set once
                         and never re-checked; authority persists
                         forever once granted.
  2. decay_only       -- has Theorem 2's age-based exponential decay
                         (``authority_memory.decay``) but no explicit
                         world-version awareness: it WILL eventually
                         expire, but on ITS OWN clock from t=0, with
                         no relationship to when the world actually
                         changed. This is the historical S2 finding
                         (``docs/FRONTIER_UCSM_2026-07-27.md`` §S2):
                         staleness alone is necessary but not
                         sufficient.
  3. full_protocol    -- decay AND an explicit world-version check
                         (``authority_memory.apply_world_version_check``)
                         that fires ``detection_delay`` ticks after
                         the world actually changes (the receiver's
                         next periodic re-observation, not
                         omniscience). Forces EXPIRED immediately once
                         the mismatch is detected.

Swept across several ``t_invalidated`` values (all comfortably before
decay_only's own from-t0 expiry horizon, so its eventual expiry is
never accidentally "correct" by coincidence): the registered claim is
that full_protocol's Revocation Latency is small AND constant
(``detection_delay``, invariant to when the world actually changed),
while decay_only's latency VARIES with ``t_invalidated`` (it is
governed by an unrelated clock), and never_expires never revokes at
all within the simulation horizon.

Registered predictions (written to disk BEFORE any trial is run):
  E3.ordering       -- for every t_invalidated tested:
                       L_R(full) < L_R(decay_only) < horizon (never_expires)
                       AND
                       stale_area(full) < stale_area(decay_only)
                                        < stale_area(never_expires).
  E3.full_invariant -- L_R(full_protocol) == detection_delay exactly,
                       for every t_invalidated tested (responsiveness
                       does not depend on when the change happened).
  E3.decay_consistency -- L_R(decay_only) == expiry_horizon(e0, rate,
                       tau_e) - t_invalidated exactly, for every
                       t_invalidated tested (the closed-form Theorem-2
                       formula must agree with the tick-by-tick
                       simulation, not just the unit tests in
                       authority_memory/tests/test_revocation.py).

Usage::

    PYTHONPATH=. .venv/bin/python \\
        experiments/authority_memory/exp_e3_stale_truth.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from authority_memory.authority_state import AuthorityState
from authority_memory.revocation import (apply_staleness,
                                         apply_world_version_check,
                                         expiry_horizon)
from experiments.authority_memory.corruption_kit import (
    build_staleness_scenario, make_admitted_certificate)

OUT_DIR = "tmp/authority_memory/e3_stale_truth"
T_INVALIDATED_VALUES = (5, 10, 15, 20)
EVIDENCE_AT_T0 = 1.0
DECAY_RATE = 0.02
TAU_E = 0.6                     # same threshold established in Sprint 5
DETECTION_DELAY = 2             # full_protocol's periodic re-check cadence
SIM_HORIZON = 100


def _run_never_expires(t_invalidated: int) -> Dict[str, Any]:
    scenario = build_staleness_scenario(t_invalidated)
    cert = make_admitted_certificate(scenario, "cert-never-expires",
                                     EVIDENCE_AT_T0)
    area = 0.0
    for t in range(t_invalidated, SIM_HORIZON + 1):
        area += cert.evidence_score       # never decays, never checked
    return {"t_revoked": None, "revocation_latency": None,
           "stale_authority_area": area,
           "final_state": cert.authority_state.value}


def _run_decay_only(t_invalidated: int) -> Dict[str, Any]:
    scenario = build_staleness_scenario(t_invalidated)
    cert = make_admitted_certificate(scenario, "cert-decay-only",
                                     EVIDENCE_AT_T0)
    t_revoked = None
    area = 0.0
    for t in range(0, SIM_HORIZON + 1):
        if t >= t_invalidated and cert.authority_state != AuthorityState.EXPIRED:
            area += cert.evidence_score
        fired = apply_staleness(cert, age=t, rate=DECAY_RATE, tau_e=TAU_E,
                                utility_lcb=0.0, reason="age decay",
                                timestamp=t)
        if fired and t_revoked is None:
            t_revoked = t
    latency = None if t_revoked is None else t_revoked - t_invalidated
    return {"t_revoked": t_revoked, "revocation_latency": latency,
           "stale_authority_area": area,
           "final_state": cert.authority_state.value}


def _run_full_protocol(t_invalidated: int) -> Dict[str, Any]:
    scenario = build_staleness_scenario(t_invalidated)
    cert = make_admitted_certificate(scenario, "cert-full-protocol",
                                     EVIDENCE_AT_T0)
    detect_at = t_invalidated + DETECTION_DELAY
    t_revoked = None
    area = 0.0
    for t in range(0, SIM_HORIZON + 1):
        if t >= t_invalidated and cert.authority_state != AuthorityState.EXPIRED:
            area += cert.evidence_score
        # Decay runs continuously in the background (the mechanism is
        # ALWAYS present); the version check only fires at the one
        # scheduled "visit" tick, modelling detection delay rather
        # than omniscience.
        fired = apply_staleness(cert, age=t, rate=DECAY_RATE, tau_e=TAU_E,
                                utility_lcb=0.0, reason="age decay",
                                timestamp=t)
        if not fired and t == detect_at:
            fired = apply_world_version_check(
                cert, current_world_version=scenario.world_version_after,
                utility_lcb=0.0, reason="world-version mismatch detected",
                timestamp=t)
        if fired and t_revoked is None:
            t_revoked = t
    latency = None if t_revoked is None else t_revoked - t_invalidated
    return {"t_revoked": t_revoked, "revocation_latency": latency,
           "stale_authority_area": area,
           "final_state": cert.authority_state.value}


ARMS = {
    "never_expires": _run_never_expires,
    "decay_only": _run_decay_only,
    "full_protocol": _run_full_protocol,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    theoretical_decay_horizon = expiry_horizon(EVIDENCE_AT_T0, DECAY_RATE,
                                               TAU_E)
    with open(os.path.join(args.out, "e3_registered.json"), "w") as f:
        json.dump({
            "E3.ordering": "L_R(full) < L_R(decay_only) < horizon "
                          "(never_expires); same order for "
                          "stale_authority_area, for every "
                          "t_invalidated tested",
            "E3.full_invariant": {"revocation_latency": DETECTION_DELAY},
            "E3.decay_consistency": {
                "expiry_horizon_from_t0": theoretical_decay_horizon},
            "t_invalidated_values": list(T_INVALIDATED_VALUES),
            "decay_rate": DECAY_RATE, "tau_e": TAU_E,
            "detection_delay": DETECTION_DELAY,
        }, f, indent=2)

    results: Dict[int, Dict[str, Any]] = {}
    for t_inv in T_INVALIDATED_VALUES:
        results[t_inv] = {arm: fn(t_inv) for arm, fn in ARMS.items()}

    ordering_ok = all(
        results[t_inv]["full_protocol"]["revocation_latency"]
        < results[t_inv]["decay_only"]["revocation_latency"]
        and results[t_inv]["decay_only"]["revocation_latency"] < SIM_HORIZON
        and results[t_inv]["never_expires"]["revocation_latency"] is None
        and results[t_inv]["full_protocol"]["stale_authority_area"]
        < results[t_inv]["decay_only"]["stale_authority_area"]
        < results[t_inv]["never_expires"]["stale_authority_area"]
        for t_inv in T_INVALIDATED_VALUES)

    full_invariant_ok = all(
        results[t_inv]["full_protocol"]["revocation_latency"]
        == DETECTION_DELAY
        for t_inv in T_INVALIDATED_VALUES)

    decay_consistency_ok = all(
        results[t_inv]["decay_only"]["revocation_latency"]
        == round(theoretical_decay_horizon) - t_inv
        or abs(results[t_inv]["decay_only"]["revocation_latency"]
              - (theoretical_decay_horizon - t_inv)) < 1.0
        for t_inv in T_INVALIDATED_VALUES)

    verdict = {
        "E3.ordering": ordering_ok,
        "E3.full_invariant": full_invariant_ok,
        "E3.decay_consistency": decay_consistency_ok,
    }
    go_no_go = all(verdict.values())

    summary = {
        "theoretical_decay_horizon_from_t0": theoretical_decay_horizon,
        "results_by_t_invalidated": results,
    }
    with open(os.path.join(args.out, "e3_results.json"), "w") as f:
        json.dump({"summary": summary, "verdict": verdict,
                  "go_no_go": go_no_go}, f, indent=2)

    print(json.dumps(results, indent=2))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"  [{'GO' if go_no_go else 'NO-GO'}] Sprint 4 critical-path "
         f"verdict")
    print(f"Saved: {args.out}/e3_results.json")


if __name__ == "__main__":
    main()
