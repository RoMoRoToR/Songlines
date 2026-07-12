"""Paired statistics for the single-agent oracle claim + an
episode-weighted recount of the empty-candidate rate (reviewer
requests: CIs and a paired test for 0.39->0.60; check that the 91%
figure is not inflated by long failing episodes producing more
decision points).

1. Hazard recovery, base vs oracle-R, 10 seeds x 8 episodes: per-seed
   paired deltas, exact sign test, and a paired bootstrap CI of the
   mean delta (resampling seeds).
2. Empty-candidate rate three ways: pooled over calls (the paper's
   91%), seed-weighted, and episode-weighted (mean over episodes of
   the per-episode empty share) from the per-query debug records.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/analyze_oracle_pairing.py
"""

from __future__ import annotations

import glob
import json
import math
from collections import defaultdict

import numpy as np

SEED_ROWS = ("tmp/oracle_stage_interventions_final_20260430/analysis/"
             "oracle_stage_seed_rows.json")
QUERY_GLOB = ("tmp/article_revision_10seeds_20260501/hazard_recovery/"
              "assists_on/MiniGrid-LavaGapS7-v0/seed_*/"
              "milestone_state_conditioned_hazard_recovery_v7/"
              "query_debug.json")
RNG = np.random.default_rng(0)


def main() -> None:
    rows = json.load(open(SEED_ROWS))
    hz = [r for r in rows if r["task_name"] == "hazard_recovery"]
    base = {r["seed"]: r["success_rate"] for r in hz
            if r["regime"] == "base"}
    orr = {r["seed"]: r["success_rate"] for r in hz
           if r["regime"] == "oracle_retrieval"}
    seeds = sorted(base)
    deltas = np.array([orr[s] - base[s] for s in seeds])
    print("per-seed deltas (oracle-R - base):",
          [f"{d:+.3f}" for d in deltas])

    n_pos = int((deltas > 0).sum())
    n_neg = int((deltas < 0).sum())
    n_eff = n_pos + n_neg  # zeros dropped (exact sign test)
    p_two = sum(math.comb(n_eff, k) for k in
                range(min(n_pos, n_neg) + 1)) * 2 / 2 ** n_eff
    p_two = min(1.0, p_two)

    idx = RNG.integers(0, len(deltas), size=(4000, len(deltas)))
    boots = deltas[idx].mean(axis=1)
    print(f"mean delta {deltas.mean():+.3f} "
          f"[{np.percentile(boots, 2.5):+.3f},"
          f"{np.percentile(boots, 97.5):+.3f}] (seed bootstrap)")
    print(f"sign test: {n_pos} positive / {n_neg} negative / "
          f"{len(deltas) - n_eff} zero -> two-sided p = {p_two:.4f}")

    # ── empty-candidate rate, three weightings ─────────────────────
    per_seed, per_episode = {}, defaultdict(lambda: [0, 0])
    tot_calls = tot_empty = 0
    for path in sorted(glob.glob(QUERY_GLOB)):
        seed = path.split("seed_")[1].split("/")[0]
        recs = json.load(open(path))
        if isinstance(recs, dict):
            recs = recs.get("records", list(recs.values())[0])
        emp = calls = 0
        for r in recs:
            empty = not r.get("candidate_node_ids")
            calls += 1
            emp += int(empty)
            ep = (seed, r.get("episode_id"))
            per_episode[ep][0] += int(empty)
            per_episode[ep][1] += 1
        per_seed[seed] = (emp, calls)
        tot_calls += calls
        tot_empty += emp

    pooled = tot_empty / tot_calls
    seedw = np.mean([e / c for e, c in per_seed.values()])
    epw_rates = [e / c for e, c in per_episode.values()]
    epw = float(np.mean(epw_rates))
    print(f"\nempty-candidate rate: pooled {pooled:.3f} "
          f"({tot_empty}/{tot_calls}), seed-weighted {seedw:.3f}, "
          f"episode-weighted {epw:.3f} over {len(epw_rates)} episodes")
    ep_counts = [c for _, c in per_episode.values()]
    print(f"decision points per episode: min {min(ep_counts)}, "
          f"median {int(np.median(ep_counts))}, max {max(ep_counts)}")


if __name__ == "__main__":
    main()
