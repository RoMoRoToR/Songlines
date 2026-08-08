"""E1 — Social Amplification: the first critical experiment of the
Memory-as-Authority-Protocol frontier
(``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/04_EXPERIMENTS.md`` §E1).

Hypothesis: naive shared/vector memory architectures increase
confidence in a claim under repeated social RETRANSMISSION, even when
no new independent evidence ever appears; a provenance-aware
authority protocol (Sprint 1-2, ``authority_memory/``) does not,
because its bookkeeping distinguishes "this instance was relayed"
(``provenance_parents``) from "this is new evidence"
(``origin_ids``) --- Theorem 1, ``02_THEOREMS.md`` §1.

Scenario (type-C corruption, ``05_BENCHMARK_CORRUPTIONS.md`` §2.C):
one agent ('A') makes a single (evaluator-known-false) observation;
the claim is then relayed through a randomly drawn chain of 5 hops
across a 6-agent pool, so a hop MAY return to an agent already in the
chain (the laundering case) purely by chance.  200 independent random
chains (``--seeds``) give the distribution needed to check whether
each architecture's authority curve grows with hop count.

Five arms, one shared set of ground-truth counters (messages_seen,
distinct_agents_seen, n_eff --- computed once from the REAL
``ProvenanceGraph``/``MemoryCertificate`` machinery, Sprint 1-2), five
different (and independently motivated) ways of turning those
counters into a confidence score:

  1. shared_context      -- authority tracks raw MESSAGE count (every
                             relay is one more mention in context);
                             fast EMA toward certainty --- models the
                             "illusory truth" repetition bias of naive
                             context concatenation.
  2. vector_memory       -- authority tracks the same raw message
                             count, but through a hard top-K retrieval
                             window (linear ramp to certainty at K
                             matching chunks) --- a different surface
                             mechanism, same failure: no source dedup.
  3. naive_graph_merge   -- authority updates only when a NEW
                             DISTINCT AGENT relays the claim (dedup by
                             source identity) --- exactly the bug: a
                             relayer looks identical to an independent
                             witness.
  4. source_count_trust  -- a stateless trust RATIO of the same
                             distinct-agent count (more "sources" =
                             more trust) --- a different formula, same
                             input, same blind spot.
  5. provenance_aware    -- THE SAME update rule as arm 3 (EMA,
                             same rate), gated by n_eff (independent
                             ORIGIN count) instead of distinct-agent
                             count.  Since there is exactly one origin
                             throughout, this arm updates once (at the
                             origin observation) and never again.

Registered predictions (written to disk BEFORE any trial is run):
  E1.core -- arm 5's PAF at hop 5 (mean over trials) in [0.9, 1.1];
             arms 1-3 show a statistically significant rise between
             hop 1 and hop 5 (non-overlapping bootstrap 95% CI).
  E1.diag -- (not gating; reported for completeness) arm 4 is
             expected to rise similarly to arm 3, since both are
             driven by distinct-agent count; 04_EXPERIMENTS.md's E1
             acceptance sentence does not commit to it.
  E1.n_eff_flat -- n_eff must equal 1 at every hop of every trial, by
             construction of Theorem 1's code path (a direct sanity
             check on Sprint 2, independent of the authority-score
             arithmetic above).

This is the go/no-go point of Sprint 3
(``07_ROADMAP_SPRINTS.md``): if arm 5 fails E1.core, the provenance
architecture (not the authority-score arithmetic) must be fixed
before Sprint 4 proceeds.

Usage::

    PYTHONPATH=. .venv/bin/python \\
        experiments/authority_memory/exp_e1_social_amplification.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from authority_memory.metrics import n_eff
from experiments.authority_memory.corruption_kit import build_chain

OUT_DIR = "tmp/authority_memory/e1_social_amplification"
N_HOPS_DEFAULT = 5
N_SEEDS_DEFAULT = 200
CI_ALPHA = 0.05
N_BOOTSTRAP_RESAMPLES = 2000

RATE_CONTEXT = 0.6        # arm 1: fast, undifferentiated repetition bias
RATE_GRAPH_MERGE = 0.5    # arms 3 and 5: SAME rate, different gate
VECTOR_TOP_K = 3          # arm 2: certainty once K chunks are retrieved
TRUST_C = 1.0             # arm 4: trust = d / (d + TRUST_C)


# ── ground-truth counters, computed once from the real machinery ---
def bookkeeping(chain) -> Dict[str, List[int]]:
    n = len(chain.certificates)
    distinct_agents_seen: List[int] = []
    seen_agents = set()
    for k in range(n):
        sender = chain.origin_agent if k == 0 else chain.hops[k - 1].sender
        seen_agents.add(sender)
        distinct_agents_seen.append(len(seen_agents))
    return {
        "messages_seen": list(range(1, n + 1)),
        "distinct_agents_seen": distinct_agents_seen,
        "n_eff": [n_eff(cert) for cert in chain.certificates],
    }


def _new_value_flags(counts: List[int]) -> List[bool]:
    prev = 0
    flags = []
    for c in counts:
        flags.append(c > prev)
        prev = c
    return flags


def _ema_update(prior: float, rate: float) -> float:
    """One EMA step toward full confidence (target=1.0) --- the same
    update FORM already used elsewhere in the project for incremental
    trust accumulation, re-derived locally so this synthetic baseline
    stays self-contained (it deliberately does not import
    ``distributed_memory``: this experiment models what a NAIVE
    system would do, not what the existing trust layer does)."""
    return prior + rate * (1.0 - prior)


# ── the five arms -----------------------------------------------------
def authority_shared_context(counts: Dict[str, List[int]]) -> List[float]:
    authority, out = 0.0, []
    for _ in counts["messages_seen"]:
        authority = _ema_update(authority, RATE_CONTEXT)
        out.append(authority)
    return out


def authority_vector_memory(counts: Dict[str, List[int]]) -> List[float]:
    return [min(1.0, m / VECTOR_TOP_K) for m in counts["messages_seen"]]


def authority_naive_graph_merge(counts: Dict[str, List[int]]) -> List[float]:
    authority, out = 0.0, []
    for flag in _new_value_flags(counts["distinct_agents_seen"]):
        if flag:
            authority = _ema_update(authority, RATE_GRAPH_MERGE)
        out.append(authority)
    return out


def authority_source_count_trust(counts: Dict[str, List[int]]) -> List[float]:
    return [d / (d + TRUST_C) for d in counts["distinct_agents_seen"]]


def authority_provenance_aware(counts: Dict[str, List[int]]) -> List[float]:
    authority, out = 0.0, []
    for flag in _new_value_flags(counts["n_eff"]):
        if flag:
            authority = _ema_update(authority, RATE_GRAPH_MERGE)
        out.append(authority)
    return out


ARMS = {
    "shared_context": authority_shared_context,
    "vector_memory": authority_vector_memory,
    "naive_graph_merge": authority_naive_graph_merge,
    "source_count_trust": authority_source_count_trust,
    "provenance_aware": authority_provenance_aware,
}
CORE_ARMS = ("shared_context", "vector_memory", "naive_graph_merge")


# ── statistics --------------------------------------------------------
def bootstrap_ci(values: List[float], seed: int = 0
                 ) -> Tuple[float, float, float]:
    """Percentile bootstrap over trials --- (mean, lo, hi) for a
    two-sided 95% CI.  A constant array (zero variance, which arms 1
    and 2 produce here: their formulas depend only on hop index, not
    on the random topology) collapses the CI to the point estimate
    rather than an artifact of resampling a constant."""
    arr = np.asarray(values, dtype=float)
    if np.allclose(arr, arr[0]):
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(N_BOOTSTRAP_RESAMPLES)])
    lo, hi = np.percentile(means, [100 * CI_ALPHA / 2,
                                   100 * (1 - CI_ALPHA / 2)])
    return float(arr.mean()), float(lo), float(hi)


def significant_growth(ci_k1: Tuple[float, float, float],
                       ci_k5: Tuple[float, float, float]) -> bool:
    """Non-overlapping CI, growth direction (hi at k1 below lo at
    k5) --- the exact registered acceptance criterion, not merely
    'means differ'."""
    return ci_k1[2] < ci_k5[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    ap.add_argument("--hops", type=int, default=N_HOPS_DEFAULT)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e1_registered.json"), "w") as f:
        json.dump({
            "E1.core": {
                "arm5_paf_at_final_hop_in": [0.9, 1.1],
                "arms_show_significant_growth_hop1_vs_hopN": list(CORE_ARMS),
            },
            "E1.diag": {
                "arm4_expected_to_rise_like_arm3_not_gating": True,
            },
            "E1.n_eff_flat": {
                "n_eff_equals_one_at_every_hop_every_trial": True,
            },
            "n_seeds": args.seeds, "n_hops": args.hops,
        }, f, indent=2)

    trials: Dict[str, List[List[float]]] = {arm: [] for arm in ARMS}
    n_eff_trials: List[List[int]] = []
    for seed in range(args.seeds):
        chain = build_chain(seed, n_hops=args.hops)
        counts = bookkeeping(chain)
        n_eff_trials.append(counts["n_eff"])
        for arm_name, arm_fn in ARMS.items():
            trials[arm_name].append(arm_fn(counts))

    n_eff_flat = all(v == 1 for traj in n_eff_trials for v in traj)

    curves: Dict[str, List[Tuple[float, float, float]]] = {}
    paf: Dict[str, Tuple[float, float, float]] = {}
    for arm_name in ARMS:
        arm_trials = trials[arm_name]          # [seed][hop]
        curves[arm_name] = [
            bootstrap_ci([arm_trials[s][k] for s in range(args.seeds)],
                        seed=k)
            for k in range(args.hops + 1)]
        paf_per_trial = [arm_trials[s][args.hops] / arm_trials[s][0]
                         for s in range(args.seeds)]
        paf[arm_name] = bootstrap_ci(paf_per_trial, seed=999)

    growth = {arm: significant_growth(curves[arm][1], curves[arm][args.hops])
             for arm in ARMS}

    core_paf_pass = 0.9 <= paf["provenance_aware"][0] <= 1.1
    core_growth_pass = all(growth[arm] for arm in CORE_ARMS)
    verdict = {
        "E1.core_paf_arm5_in_band": core_paf_pass,
        "E1.core_growth_arms_1_2_3": core_growth_pass,
        "E1.n_eff_flat": n_eff_flat,
        "E1.diag_arm4_growth": growth["source_count_trust"],
    }
    go_no_go = core_paf_pass and core_growth_pass and n_eff_flat

    summary = {
        "n_seeds": args.seeds, "n_hops": args.hops,
        "curves": curves, "paf": paf, "growth_hop1_vs_hopN": growth,
    }
    with open(os.path.join(args.out, "e1_results.json"), "w") as f:
        json.dump({"summary": summary, "verdict": verdict,
                  "go_no_go": go_no_go}, f, indent=2)

    print(json.dumps({"paf": paf, "growth_hop1_vs_hopN": growth}, indent=2))
    print("=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"  [{'GO' if go_no_go else 'NO-GO'}] Sprint 3 critical-path "
         f"verdict")
    print(f"Saved: {args.out}/e1_results.json")


if __name__ == "__main__":
    main()
