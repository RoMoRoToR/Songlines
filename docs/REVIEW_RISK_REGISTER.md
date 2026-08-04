# Internal Review & Risk Register (Stage 15)

Three independent internal review passes before submission — method,
experiment, hostile — followed by a risk register. Each critical
finding is logged with severity, evidence, required fix, status.

---

## Reviewer 1 — Method

*Is the algorithm clear? implementable? no hidden oracle? storage vs
consumption separated? baselines honest?*

- **Algorithm clear / implementable:** yes since Stage 5 — Algorithm 1 (18 steps), record type `m=(G,C,E,U,P,T,R,F,A)`, formula box, hyperparameter table with selection method. Verifiable: the invariant suite (`songlines/tests`, 11/11) and one-runtime config registry (`songlines/config.py`).
- **Hidden oracle:** the counterfactual utility is exact (replay) in grid worlds — that IS an oracle. **Addressed:** UE1 shows a runtime estimator (Spearman 0.989) drives the full system within 3.5% of the oracle version; external-env calibration flagged open (claim B19, `We do not claim`).
- **Storage vs consumption separated:** yes — `SonglineAgent.form/receive/_admit` (storage) vs `.targets` (consumption); the U2 v1 failure taught this and it is now structural.
- **Baselines honest:** four cards (`docs/baselines/BASELINE_CARDS.md`), each keeps its own idea's strength, stripped only of the axis under test; offline re-implementation caveat stated.
- **Residual method risk:** graph-matching analogy has no roles/causality (B10) — scoped as future work, not claimed.

## Reviewer 2 — Experiment

*splits, seeds, leakage, tuning, statistics, cost accounting, every
headline number.*

- **Splits/seeds/leakage:** test seeds 100+ never used to tune any threshold (frozen in `SONGLINES_V1_FREEZE.md`); safety calibration on dev 5001+/5101+/200-. No leakage found.
- **Tuning:** all formation constants frozen from U1; distance-law constants from Part II; do-not-retune list enforced.
- **Statistics:** paired seeds, 12/12 sign test p=0.0002 on I1. **Gap closed:** the main benchmark was widened to **30 test seeds (100–129)** at the Stage-8 rerun — BU1 songline 135.1 vs best baseline 164.5 (−17.9%), 30/30 paired, two-sided sign test p=1.9e-9, world-block bootstrap 95% CI on the paired gap [28.1, 30.7] (disjoint from 0). Status: CLOSED.
- **Cost accounting:** full wire codec (cert/prov/time/resv); pure-codec vs full-protocol distinguished (A12/A13). **Gap:** full-protocol payload ratio (A13) not yet computed → Stage 9. Status: IN PROGRESS.
- **Headline numbers:** each traces to a `*_results.json`/jsonl via `CLAIM_EVIDENCE_MATRIX.md` and the long CSV; automated table generation from `artifacts/song_grammar_long.csv` recommended before submission.

## Reviewer 3 — Hostile

*attack novelty, claims, safety, generality, SOTA comparison, term
correctness, necessity of each mechanism.*

- **"Just a bigger vector memory":** refuted — vector_sim and graph_memory baselines lose 1.7–4.5×; ablations localise each mechanism's contribution.
- **"Safety numbers are cherry-picked":** N1v2 fail-open 0.000 across the whole 20%×20% grid; continuous 0.0014 on held-out test — both registered, both with the coverage cost stated.
- **"Provenance/exceptions/immutable are dead weight" (I1-H1b honest FAIL):** conceded in this substrate; their value shown where the substrate stresses them (U1 corruption, S1 gossip, P1 poison). Claim scoped accordingly — not overclaimed.
- **"6.4% is misleading":** fixed — labelled pure song codec; full-protocol ratio is A13 (open).
- **"full SE(2)" / "inherits safety" / "emergent landmarks" / "evolvable grammar":** all corrected to precise terms (Stage 4).
- **"No external validity":** now answered — Stage 13 runs the WHOLE runtime (not the Q/R/M/C logger) on VMAS continuous-physics: songline_safe reaches a frame-free-recovered target in 254.5 vs 543.5 team steps (−53.2%), 12/12 paired seeds p=0.0005, transport fail-open 0.0000. Third substrate after grid and synthetic continuous. Remaining: language substrate (ALFWorld) and real robots — scoped open, not blocking.
- **"Adversary too weak":** P1 poisoner corrupts + launders under original origin; finding is that world entropy dominates the adversary — honest, not a dodge.

---

## Risk Register

| risk | severity | evidence | required fix | status |
|---|---|---|---|---|
| Only 12 test seeds on main table (plan wants ≥30) | med | I1/bench 12-seed shards | 30-seed rerun, bootstrap+Holm | CLOSED — 30-seed rerun (seeds 100–129): BU1 −17.9%, 30/30 paired, sign-test p=1.9e-9, bootstrap 95% CI Δ [28.1,30.7] (bench30/v30_verdict.json) |
| Full-protocol payload ratio (A13) uncomputed | med | claim matrix A13 | Stage 9 cost recompute | IN PROGRESS |
| Counterfactual utility is a grid oracle | med | UE1 mitigates (ρ=0.989) | external-env estimator | scoped open (B19) |
| No external end-to-end substrate | high | grid+continuous only | ALFWorld/VMAS full runtime | CLOSED — V2 on VMAS physics: whole runtime, −53.2% team steps, 12/12 p=0.0005, fail-open 0.0000 (tmp/v2_full/v2_verdict.json) |
| Baselines are re-implementations, not orig repos | med | BASELINE_CARDS caveat; BU1-BU3 done (full −18% vs best, 12/12) | run orig code if net access | documented, benchmarked |
| Graph analogy lacks roles/causality | low | B10 scoped | future work | accepted |
| Provenance value substrate-dependent | low | I1-H1b | shown in U1/S1/P1 | accepted, scoped |
| numpy 2.0.2 (local) vs 2.5.1 (cluster) | low | FREEZE §8 | pin environment.yml at rerun | OPEN (Stage 8) |
| Repo uncommitted; tag not yet cut | med | FREEZE §9 | commit + tag songlines-runtime-v1 | author action |
| Continuous safety needs 3 layers (fragile?) | low | C1.4 | test transfer to other continuous worlds | scoped open |

**Blocking for main method-paper submission (critical path): NONE — the critical path is empty.** 30-seed rerun (Stage 8, done — BU1 p=1.9e-9), A13 cost (Stage 9, done), one external end-to-end (Stage 13, done — VMAS X10) are all closed. Remaining risk-register items are author actions (commit+tag, environment.yml pin) or scoped-open future work (ALFWorld language substrate, causal analogy, real robots), none blocking first submission. **Non-blocking (first-submission-deferrable):** open-ended landmark invention, causal analogy, real robots, large social-LLM benchmark, general categorical theory.
