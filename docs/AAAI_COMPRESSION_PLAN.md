# AAAI-2027 compression plan for `songlines_qrmc_measurement_framework`

**Why this is a plan, not a finished reformat.** AAAI main track uses the
`aaai2027.sty`/`aaai.bst` template (2-column, ~7 body pages + unlimited
references + an optional technical/reproducibility appendix). The current
manuscript is single-column NeurIPS format at ~39 pages. Producing the AAAI PDF
requires (a) the AAAI style files (not in this repo) and (b) editorial calls on
what to cut — both author decisions. This document fixes the target budget so
the reformat is mechanical once the template is dropped in.

## Target: ≤7 body pages (2-column)

### Keep in body (the one clear "win")
1. **Abstract + Intro** — compress the 3 bullet contributions to 2 (framework +
   bottleneck-shift result); drop the "diagnostic use" paragraph to §related.
2. **§2 Q/R/M/C decomposition** — keep the factorization, Assumptions 1–2,
   Definition 1/2, Empirical Claim 1, and the causal-DAG figure (Fig. `causal`).
   **Move** §2.3 order-theoretic (Galois) reading and §2.4 options/H-MDP reading
   to the appendix (they are explicitly interpretive lenses, not contributions —
   §2.4 already says so). Leave a 2-sentence pointer in body.
3. **§3 single-agent** — keep the hero figure and the oracle-intervention result
   (the diagnose→intervene→verify loop) and the memory-accumulation refinement.
   Move the per-task tables to appendix.
4. **§4 multi-agent** — keep: 4 architectures, cadence axis, the bottleneck-shift
   figure, the interior-`t_succ`-optimum figure, and the one-paragraph statement
   of the M↔C result. This is the headline.
5. **§5 limitations + §6 conclusion** — keep, tightened.

### Move to technical appendix (already appendix-shaped)
- Galois connection statement + proof (`app:galois`), options/SMDP reading.
- Effect-size / cluster-robust inference (`app:effect-sizes`), design-decomposition
  table (`app:design`), Y-vs-C\* (`app:y-vs-c`), Assumption-1 stress test
  (`app:assumption1-stress`), θ-robustness (`app:theta-robustness`), oracle-C
  definition (`app:oracle-c-def`), VMAS preliminary (`app:vmas`).
- N=12 scale-up, BabyAI portability, retrained-MLP + CommNet-PPO detail, full
  notation table, SOTA coverage matrix.
- The NeurIPS `checklist.tex` is **dropped** for AAAI (no NeurIPS checklist);
  fold its reproducibility content into the reproducibility appendix.

### Cut or shrink
- LLM TextNav block → one sentence in body + keep the honest "portability only"
  framing; table to appendix (or cut if page-critical).
- Redundant restatements of the Φ-vs-`t_succ` split (currently in intro,
  §2.3, §4.5, and appendix) → state once, reference thereafter.

## Figures budget (body)
Keep 3–4: (1) causal DAG, (2) single-agent hero/oracle waterfall, (3) bottleneck
shift + interior `t_succ` optimum (combine panels), (4) optional architecture
schematic. All others → appendix.

## Nomenclature/consistency (do during reformat)
- Definition 1/2 + Empirical Claim 1 naming is now consistent across text and
  (former) checklist; keep it. Renumber for AAAI once sections are cut.
- Run counts reconcile via `app:design` (35,640 = 3,240 base × [peer×8 + 3]).

## Mechanical steps once `aaai2027.sty` is available
1. Swap `\usepackage{neurips_2026}` → AAAI preamble; 2-column.
2. Split file into `main` (body) + `appendix` (`\appendix` include).
3. Move the blocks listed above; fix `\ref`/`\label` (all already labelled).
4. Drop `\input{checklist.tex}`; add AAAI reproducibility checklist if required.
5. Recompile; verify ≤7 body pages and no undefined refs.
