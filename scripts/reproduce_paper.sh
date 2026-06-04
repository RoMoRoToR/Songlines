#!/bin/bash
#
# One-command reproduction of all paper artifacts.
#
# Wall-clock estimate on an 8-core ARM machine, 32 GB RAM, no GPU:
#   - multi-agent 40-seed sweep (Section 7.5): ~18 min
#   - multi-agent oracle interventions (Section 7.5 oracle block): ~40 s
#   - Q/R/M/C stage analysis + bootstrap CIs: ~20 s
#   - 8-script smoke for the underlying memory layers: ~1 s
#   - single-agent benchmark suite (Sections 7.1-7.4) and BabyAI transfer
#     are reproduced from the separate scripts listed in Appendix G.
#
# Usage:
#   bash scripts/reproduce_paper.sh
#
# Or to skip the long multi-agent sweep and reuse cached results:
#   SKIP_SWEEP=1 bash scripts/reproduce_paper.sh
#
# All outputs land under tmp/.

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

PYTHON="${PYTHON:-.venv/bin/python}"
WORKERS="${WORKERS:-8}"

echo "[1/4] Smoke tests for collective-memory layers (Phase 1-4)..."
$PYTHON scripts/run_all_smokes.py --out_dir tmp/all_smokes

if [ "${SKIP_SWEEP:-0}" != "1" ]; then
  echo "[2/4] Multi-agent 40-seed cadence sweep (12,960 runs, ~18 min)..."
  $PYTHON experiments/big_experiment/exp_cadence_phase.py \
      --mode full --workers "$WORKERS" \
      --out_dir tmp/big_experiment_qrmc \
      --progress_every 5000
else
  echo "[2/4] SKIP_SWEEP=1, reusing tmp/big_experiment_qrmc/runs.csv"
fi

echo "[3/4] Multi-agent oracle interventions (~40 s)..."
$PYTHON experiments/big_experiment/exp_oracle_interventions.py \
    --workers "$WORKERS" --out_dir tmp/big_experiment_oracle

echo "[4/4] Q/R/M/C stage decomposition, bootstrap CIs, paired tests..."
$PYTHON experiments/big_experiment/analyze_qrmc.py \
    --runs_csv tmp/big_experiment_qrmc/runs.csv \
    --out_dir tmp/big_experiment_qrmc

cat <<EOF

==========================================================================
Reproduction complete.

Key files (paths relative to repo root):

  Smoke results:
    tmp/all_smokes/all_smokes_summary.json

  Multi-agent sweep:
    tmp/big_experiment_qrmc/runs.csv               (raw, 12,960 rows)
    tmp/big_experiment_qrmc/qrmc_validation.json   (Spearman + bootstrap)
    tmp/big_experiment_qrmc/bottleneck_product_MC.png
    tmp/big_experiment_qrmc/stage_conditional_rates.png
    tmp/big_experiment_qrmc/stage_profile_pertick.png

  Oracle interventions:
    tmp/big_experiment_oracle/oracle_runs.csv

For the single-agent benchmark scripts (Sections 7.1-7.4) and BabyAI
portability check (Appendix D), see Appendix G of the paper.
==========================================================================
EOF
