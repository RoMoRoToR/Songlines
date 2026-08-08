"""E2 — Independent Corroboration: the second critical experiment of
the Memory-as-Authority-Protocol frontier
(``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/04_EXPERIMENTS.md`` §E2).

E1 (Sprint 3) showed that pure retransmission of a false claim never
inflates the provenance-aware arm's authority (Theorem 1).  On its
own, that is indistinguishable from a system that simply never trusts
anything --- a liveness failure (Theorem 3, ``02_THEOREMS.md``).  E2
closes that gap: genuinely independent corroboration of a USEFUL
claim must make authority rise, and it must rise exactly when new
evidence appears, not when noise (relay) happens around it.

Unlike E1 (which compared five architectures via toy scalar
formulas, because four of them have no real FSM to exercise), E2 runs
the REAL Sprint 1 authority-state machine end to end: one receiver
(``agent-F``) holds a single belief certificate about a claim; its
``evidence_score`` is recomputed from the group-level ``n_eff``
(Sprint 2) every time a new message arrives, and
``authority_state.transition()`` (Sprint 1) is invoked whenever that
score crosses tau_E --- exactly the mechanism the doc specifies for
QUARANTINED -> PROVISIONAL (``01_FORMAL_MODEL.md`` §5).  Promotion to
ADMITTED needs the utility/LCB gate, which does not exist until
Sprint 7-8; this experiment deliberately stops at PROVISIONAL, the
one state boundary that is gated by evidence alone.

Scenario per trial (topology randomised; the checkpoints are not):
  A observes X.                                 (n_eff=1, evidence=0.5)
  0-3 relay hops among {A,B,C}.                 (n_eff stays 1 --- E1's
                                                  finding, re-verified
                                                  through the REAL FSM
                                                  this time)
  D independently observes X.                   (n_eff=2, evidence=0.75
                                                  -- crosses tau_E=0.6:
                                                  QUARANTINED->PROVISIONAL)
  0-3 relay hops among {A,B,C,D}.                (n_eff stays 2, state
                                                  stays PROVISIONAL)
  E independently observes X.                   (n_eff=3, evidence=0.875)
  0-3 relay hops among {A,B,C,D,E}.              (n_eff stays 3)

evidence_score(n) reuses E1's provenance_aware EMA rule (rate 0.5,
closed form 1-(1-rate)**n) so E1 and E2 are directly comparable: the
ONLY difference between the two experiments is which events are fed
to this same rule, not the rule itself.

Registered predictions (written to disk BEFORE any trial is run):
  E2.exact_evidence  -- for 100% of trials, evidence_score equals
                        exactly 0.5 / 0.75 / 0.875 at the pre-D /
                        post-D / post-E checkpoints (the mechanism is
                        deterministic given n_eff, so this is an
                        exact-match prediction, not a statistical one).
  E2.strict_increase -- for 100% of trials, evidence strictly rises at
                        each independent corroboration (not merely
                        non-decreasing).
  E2.state_timing    -- for 100% of trials, authority_state is
                        QUARANTINED at every n_eff==1 checkpoint and
                        PROVISIONAL at every n_eff>=2 checkpoint; the
                        FIRST transition happens exactly at the
                        "independent_D" event, never at a "relay"
                        event.
  E2.relay_is_inert  -- for 100% of "relay" trajectory points,
                        evidence_score is unchanged from the
                        immediately preceding point (Theorem 1,
                        re-verified through the real FSM).

Go/no-go point #2 (``07_ROADMAP_SPRINTS.md`` Sprint 5): if authority
does not rise with independent corroboration, tau_E/rate calibration
is too conservative and must be revisited before Sprint 6.

Usage::

    PYTHONPATH=. .venv/bin/python \\
        experiments/authority_memory/exp_e2_independent_corroboration.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from authority_memory.authority_state import AuthorityState, transition
from authority_memory.certificate import Claim, receive
from authority_memory.metrics import n_eff
from authority_memory.provenance_graph import ProvenanceGraph

OUT_DIR = "tmp/authority_memory/e2_independent_corroboration"
N_SEEDS_DEFAULT = 100
RECEIVER_AGENT = "agent-F"
RATE = 0.5           # same EMA rate as E1's provenance_aware arm
TAU_E = 0.6          # crossed by n_eff=2 (0.75) but not n_eff=1 (0.5)
POOL_PRE_D = ("A", "B", "C")
POOL_PRE_E = ("A", "B", "C", "D")
POOL_POST_E = ("A", "B", "C", "D", "E")


def corroborated_claim() -> Claim:
    """A genuinely useful claim under test for liveness (Theorem 3)
    --- unlike E1's fixed-false claim (repetition must not inflate
    authority regardless of truth value), E2 is the positive case:
    independently-witnessed, useful knowledge should eventually earn
    authority.  Truth value still never enters the arithmetic (same
    as E1) --- it only makes the story land: this is the case where
    trusting it is the right call."""
    return Claim(subject="water_source_12", relation="state",
                object="available", conditions={})


def evidence_score_for_n_eff(n: int) -> float:
    """Closed form of E1's provenance_aware EMA rule (each of n unit
    updates from 0 toward 1 at ``RATE``): x_n = 1 - (1-RATE)**n.
    Path-independent, so it can be recomputed directly from the
    current group n_eff rather than replayed step by step."""
    return 1.0 - (1.0 - RATE) ** n


def run_trial(seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    graph = ProvenanceGraph()
    claim = corroborated_claim()
    raw_certs = []          # every observe/relay certificate instance
                             # seen so far --- feeds the group n_eff
    belief, _ = receive(f"belief-seed{seed}", claim, RECEIVER_AGENT,
                        timestamp=0)   # RECEIVED -> QUARANTINED, ev=0.0
    trajectory: List[Dict[str, Any]] = []

    def absorb(cert, t: int, event: str) -> None:
        raw_certs.append(cert)
        n = n_eff(raw_certs)
        score = evidence_score_for_n_eff(n)
        target = (AuthorityState.PROVISIONAL if score >= TAU_E
                 else AuthorityState.QUARANTINED)
        if target != belief.authority_state:
            transition(belief, target, evidence_score=score,
                      utility_lcb=0.0, reason=f"n_eff={n}", timestamp=t)
        else:
            belief.evidence_score = score
        trajectory.append({
            "t": t, "event": event, "n_eff": n,
            "evidence_score": belief.evidence_score,
            "authority_state": belief.authority_state.value,
        })

    def relay_noise(current, holder, pool, t):
        for _ in range(rng.randint(0, 3)):
            receiver = rng.choice([a for a in pool if a != holder])
            current, _ = graph.relay(
                current, sender=holder, receiver=receiver,
                new_certificate_id=f"cert-seed{seed}-r{t}", timestamp=t)
            absorb(current, t, "relay")
            holder = receiver
            t += 1
        return current, holder, t

    origin, _ = graph.observe(f"cert-seed{seed}-origin", claim, "A",
                              f"obs-A-{seed}", world_version=0,
                              observed_at=0)
    absorb(origin, 0, "origin")
    current, holder, t = relay_noise(origin, "A", POOL_PRE_D, 1)

    certD, _ = graph.observe(f"cert-seed{seed}-D", claim, "D",
                             f"obs-D-{seed}", world_version=0,
                             observed_at=t)
    absorb(certD, t, "independent_D")
    t += 1
    current, holder, t = relay_noise(certD, "D", POOL_PRE_E, t)

    certE, _ = graph.observe(f"cert-seed{seed}-E", claim, "E",
                             f"obs-E-{seed}", world_version=0,
                             observed_at=t)
    absorb(certE, t, "independent_E")
    t += 1
    relay_noise(certE, "E", POOL_POST_E, t)

    return trajectory


def checkpoints(trajectory: List[Dict[str, Any]]
               ) -> Dict[str, Dict[str, Any]]:
    """The three points that matter, by event type --- there is
    exactly one 'origin', one 'independent_D', one 'independent_E'
    entry per trial, however many 'relay' hops fall around them."""
    by_event = {row["event"]: row for row in trajectory
               if row["event"] != "relay"}
    return {"pre_d": by_event["origin"], "post_d": by_event["independent_D"],
           "post_e": by_event["independent_E"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e2_registered.json"), "w") as f:
        json.dump({
            "E2.exact_evidence": {"pre_d": 0.5, "post_d": 0.75,
                                  "post_e": 0.875},
            "E2.strict_increase": True,
            "E2.state_timing": {"pre_d": "quarantined",
                                "post_d": "provisional",
                                "post_e": "provisional",
                                "first_transition_event": "independent_D"},
            "E2.relay_is_inert": True,
            "tau_e": TAU_E, "rate": RATE, "n_seeds": args.seeds,
        }, f, indent=2)

    all_checkpoints = []
    relay_violations = 0
    n_relay_points = 0
    for seed in range(args.seeds):
        trajectory = run_trial(seed)
        cps = checkpoints(trajectory)
        all_checkpoints.append(cps)
        prev_score = None
        for row in trajectory:
            if row["event"] == "relay":
                n_relay_points += 1
                if prev_score is not None and row["evidence_score"] != prev_score:
                    relay_violations += 1
            prev_score = row["evidence_score"]

    exact_evidence_ok = all(
        cps["pre_d"]["evidence_score"] == 0.5
        and cps["post_d"]["evidence_score"] == 0.75
        and cps["post_e"]["evidence_score"] == 0.875
        for cps in all_checkpoints)
    strict_increase_ok = all(
        cps["post_d"]["evidence_score"] > cps["pre_d"]["evidence_score"]
        and cps["post_e"]["evidence_score"] > cps["post_d"]["evidence_score"]
        for cps in all_checkpoints)
    state_timing_ok = all(
        cps["pre_d"]["authority_state"] == AuthorityState.QUARANTINED.value
        and cps["post_d"]["authority_state"] == AuthorityState.PROVISIONAL.value
        and cps["post_e"]["authority_state"] == AuthorityState.PROVISIONAL.value
        for cps in all_checkpoints)
    relay_is_inert_ok = (relay_violations == 0)

    verdict = {
        "E2.exact_evidence": exact_evidence_ok,
        "E2.strict_increase": strict_increase_ok,
        "E2.state_timing": state_timing_ok,
        "E2.relay_is_inert": relay_is_inert_ok,
    }
    go_no_go = all(verdict.values())

    summary = {
        "n_seeds": args.seeds, "n_relay_points_checked": n_relay_points,
        "relay_violations": relay_violations,
        "example_trajectory_seed0": run_trial(0),
    }
    with open(os.path.join(args.out, "e2_results.json"), "w") as f:
        json.dump({"summary": summary, "verdict": verdict,
                  "go_no_go": go_no_go}, f, indent=2)

    print(json.dumps(summary["example_trajectory_seed0"], indent=2))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"  [{'GO' if go_no_go else 'NO-GO'}] Sprint 5 critical-path "
         f"verdict")
    print(f"Saved: {args.out}/e2_results.json")


if __name__ == "__main__":
    main()
