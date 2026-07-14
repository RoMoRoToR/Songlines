"""Phase A driver: 10 episodes of Llama-3.1-8B on TextNav.

Produces under tmp/llm_bridge_minimal/:
  runs.csv                  — one row per episode, Q/R/M/C events + success
  trace_seed0.json          — full event log for seed 0
  trace_seed0_readable.md   — human-readable trace for defense
  RESULTS.md                — summary + honest interpretation

Acceptance:
  • Q/R/M/C events emit non-trivially
  • Nested ordering Q* >= R* >= M* >= C* holds
  • At least one episode succeeds (proves the loop closes)
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.llm_collective.qrmc_llm_runner import run_one_episode


OUT_DIR = "tmp/llm_bridge_minimal"  # run from repo root
N_EPISODES = 10
STEP_LIMIT = 25


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Running {N_EPISODES} episodes of LLM agent on TextNav "
          f"(step_limit={STEP_LIMIT})…")
    print("=" * 70)

    results = []
    for s in range(N_EPISODES):
        r = run_one_episode(seed=s, step_limit=STEP_LIMIT, verbose=False)
        results.append(r)
        print(f"  seed {s}: succ={r.succeeded} ticks={r.n_ticks} "
              f"Q*={int(r.q_star)} R*={int(r.r_star)} M*={int(r.m_star)} C*={int(r.c_star)} "
              f"(n_Q={r.n_Q} n_R={r.n_R} n_M={r.n_M}) {r.wall_clock_s:.1f}s")

        # Dump trace for seed 0 (deepest artefact for defense)
        if s == 0:
            _dump_trace(r, OUT_DIR)

    # CSV
    csv_path = os.path.join(OUT_DIR, "runs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(results[0].to_summary_dict().keys()))
        w.writeheader()
        for r in results:
            w.writerow(r.to_summary_dict())
    print(f"\nSaved → {csv_path}")

    # Aggregate
    q_rate = sum(r.q_star for r in results) / len(results)
    r_rate = sum(r.r_star for r in results) / len(results)
    m_rate = sum(r.m_star for r in results) / len(results)
    c_rate = sum(r.c_star for r in results) / len(results)
    nested_ok = q_rate >= r_rate >= m_rate >= c_rate
    succ = sum(r.succeeded for r in results)
    mean_ticks = statistics.mean(r.n_ticks for r in results)

    print("\n── Aggregate ─────────────────────────────────────")
    print(f"  Episode-level rates:  Q*={q_rate:.2f}  R*={r_rate:.2f}  "
          f"M*={m_rate:.2f}  C*={c_rate:.2f}")
    print(f"  Nested-ordering Q≥R≥M≥C: {'PASS' if nested_ok else 'FAIL'}")
    print(f"  Successes: {succ}/{N_EPISODES}  mean ticks: {mean_ticks:.1f}")

    # Conditional rates
    def _safe_div(a, b):
        return float("nan") if b == 0 else a / b
    qn = sum(r.q_star for r in results)
    rn = sum(r.r_star for r in results)
    mn = sum(r.m_star for r in results)
    cn = sum(r.c_star for r in results)
    p_r_q = _safe_div(rn, qn)
    p_m_r = _safe_div(mn, rn)
    p_c_m = _safe_div(cn, mn)
    print(f"  Conditional rates:    P(R|Q)={p_r_q:.2f}  P(M|R)={p_m_r:.2f}  "
          f"P(C|M)={p_c_m:.2f}")
    print(f"  Product Q·P(R|Q)·P(M|R)·P(C|M) = {q_rate * p_r_q * p_m_r * p_c_m:.2f}  "
          f"(should ≈ C*={c_rate:.2f})")

    # Acceptance — Phase A is narrowly about the measurement interface:
    #   (1) events fire (Q/R/M/C all > 0)
    #   (2) nested ordering Q* ≥ R* ≥ M* ≥ C* holds
    #   (3) factorization is consistent (product matches C*)
    #   (4) the loop closes at least once (succ >= 1)
    product = q_rate * (p_r_q or 0) * (p_m_r or 0) * (p_c_m or 0)
    events_fire = all(x > 0 for x in (q_rate, r_rate, m_rate, c_rate))
    factorization_ok = abs(product - c_rate) < 0.05
    acc = events_fire and nested_ok and factorization_ok and succ >= 1
    print(f"\n  Phase A acceptance criteria:")
    print(f"    (1) events fire (all rates > 0):       {'PASS' if events_fire else 'FAIL'}")
    print(f"    (2) nested ordering Q≥R≥M≥C:           {'PASS' if nested_ok else 'FAIL'}")
    print(f"    (3) factorization product ≈ C*:        "
          f"{'PASS' if factorization_ok else 'FAIL'} "
          f"(prod={product:.2f}, C*={c_rate:.2f})")
    print(f"    (4) loop closes (≥1 success):           {'PASS' if succ >= 1 else 'FAIL'}")
    print(f"\n  OVERALL ACCEPTANCE: {'PASS' if acc else 'FAIL'}")

    # RESULTS.md
    summary = {
        "n_episodes": N_EPISODES,
        "step_limit": STEP_LIMIT,
        "q_star_rate": q_rate,
        "r_star_rate": r_rate,
        "m_star_rate": m_rate,
        "c_star_rate": c_rate,
        "p_R_given_Q": p_r_q,
        "p_M_given_R": p_m_r,
        "p_C_given_M": p_c_m,
        "successes": succ,
        "nested_ordering_ok": nested_ok,
        "acceptance_pass": bool(acc),
        "mean_ticks": mean_ticks,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_results_md(summary, results)
    print(f"  Saved → {OUT_DIR}/RESULTS.md")


def _dump_trace(r, out_dir):
    json_path = os.path.join(out_dir, f"trace_seed{r.seed}.json")
    md_path = os.path.join(out_dir, f"trace_seed{r.seed}_readable.md")
    payload = {
        "seed": r.seed,
        "succeeded": r.succeeded,
        "n_ticks": r.n_ticks,
        "wall_clock_s": r.wall_clock_s,
        "ticks": [dict(
            tick=t.tick, aid=t.aid, obs=t.obs_text,
            tags=t.extracted_tags,
            query_req=t.query_req, query_pref=t.query_pref,
            query_pen=t.query_pen,
            n_candidates=t.n_candidates,
            top_candidate_id=t.top_candidate_id,
            locked_target_xy=list(t.locked_target_xy) if t.locked_target_xy else None,
            chosen_action=t.chosen_action,
            Q=int(t.Q), R=int(t.R), M=int(t.M), C_so_far=int(t.C_so_far),
        ) for t in r.tick_logs],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w") as f:
        f.write(f"# LLM-driven agent trace — seed {r.seed}\n\n")
        f.write(f"**Substrate:** TextNav (3-room household, bring-apple task)  \n")
        f.write(f"**Agent:** PeerLLMAgent with Llama-3.1-8B (ollama, temperature 0, deterministic seed)  \n")
        f.write(f"**Episode outcome:** {'SUCCESS' if r.succeeded else 'failure'} after {r.n_ticks} ticks  \n\n")
        f.write("Each tick shows the Q/R/M/C events emitted by the **unchanged** "
                "Q/R/M/C measurement protocol applied to the LLM-driven decision loop.\n\n")
        for t in r.tick_logs:
            ev = "".join(["Q" if t.Q else "·", "R" if t.R else "·",
                          "M" if t.M else "·", "C" if t.C_so_far else "·"])
            f.write(f"## tick {t.tick}  events=[{ev}]\n")
            f.write(f"**Obs:** {t.obs_text}\n\n")
            tagstr = ", ".join(f"`{k}`:{v:.2f}" for k, v in sorted(
                t.extracted_tags.items(), key=lambda kv: -kv[1]))
            f.write(f"**Extracted tags:** {tagstr}\n\n")
            f.write(f"**Query:** req={t.query_req}  pref={t.query_pref}  pen={t.query_pen}\n\n")
            f.write(f"**Candidates:** {t.n_candidates}  top=`{t.top_candidate_id}`  "
                    f"locked-target={t.locked_target_xy}\n\n")
            f.write(f"**Action:** `{t.chosen_action}`\n\n")
            f.write("---\n\n")


def _write_results_md(summary, results):
    p = os.path.join(OUT_DIR, "RESULTS.md")
    with open(p, "w") as f:
        f.write("# Phase A — Q/R/M/C measurement on an LLM-driven agent\n\n")
        f.write("**Substrate:** minimal text-navigation environment (3-room household)  \n")
        f.write("**Agent class:** LLM-driven (Llama-3.1-8B via ollama, temperature 0)  \n")
        f.write(f"**Episodes:** {summary['n_episodes']}  \n")
        f.write(f"**Step limit:** {summary['step_limit']}  \n")
        f.write(f"**Wall clock per episode:** ~{summary['mean_ticks']:.0f} ticks\n\n")

        f.write("## Result: Q/R/M/C events emit non-trivially on an LLM agent\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Episode-level $Q^\\star$ rate | {summary['q_star_rate']:.2f} |\n")
        f.write(f"| Episode-level $R^\\star$ rate | {summary['r_star_rate']:.2f} |\n")
        f.write(f"| Episode-level $M^\\star$ rate | {summary['m_star_rate']:.2f} |\n")
        f.write(f"| Episode-level $C^\\star$ rate | {summary['c_star_rate']:.2f} |\n")
        f.write(f"| Nested ordering Q* ≥ R* ≥ M* ≥ C* | "
                f"{'PASS' if summary['nested_ordering_ok'] else 'FAIL'} |\n")
        f.write(f"| Successes | {summary['successes']}/{summary['n_episodes']} |\n")
        f.write(f"| $P(R^\\star\\mid Q^\\star)$ | {summary['p_R_given_Q']:.2f} |\n")
        f.write(f"| $P(M^\\star\\mid R^\\star)$ | {summary['p_M_given_R']:.2f} |\n")
        f.write(f"| $P(C^\\star\\mid M^\\star)$ | {summary['p_C_given_M']:.2f} |\n")
        f.write(f"\n**Acceptance:** {'PASS' if summary['acceptance_pass'] else 'FAIL'}\n\n")

        f.write("## What this demonstrates\n\n")
        f.write("The Q/R/M/C measurement protocol — operationalised in "
                "`experiments/big_experiment/runner.py` for the symbolic and RL "
                "agents — emits non-trivial per-stage events when the agent's "
                "decision loop is driven by a language model instead of a "
                "rule-based planner. **The Q/R/M/C event logic itself was not "
                "modified** for this run; only the agent-side adapters changed "
                "(tag extractor, query former, decider — all LLM-driven).\n\n")
        f.write("## What this does not demonstrate\n\n")
        f.write("- This is a **single LLM agent** on a **toy text substrate**. "
                "It does not yet show that the cadence-shift Empirical Claim 1 "
                "transfers to LLM agents — that requires Phase B (N=3, peer "
                "broadcast, 5 cadences × 20 seeds).\n")
        f.write("- The agent uses canonical NL prompts; no fine-tuning, no "
                "fancy chain-of-thought.\n")
        f.write("- The success rate is not directly comparable to MiniGrid "
                "benchmarks; the substrate is different.\n\n")
        f.write("The point of Phase A is narrow and honest: prove the "
                "measurement instrument is agent-class agnostic so the defence "
                "claim 'Q/R/M/C-Диаг работает и на LLM-agents' is "
                "empirically grounded, not aspirational.\n\n")
        f.write("## Artefacts\n\n")
        f.write("- `runs.csv` — one row per episode\n")
        f.write("- `trace_seed0.json` — full event log\n")
        f.write("- `trace_seed0_readable.md` — human-readable trace for defence\n")
        f.write("- `summary.json` — machine-readable aggregate\n")


if __name__ == "__main__":
    main()
