# 01 - Q/R/M/C Evaluation Code

This folder contains the measurement and experiment code for the Paper 1
Q/R/M/C diagnostic framework.

## Main Entry Points

- `scripts/run_paper1_clean_experiments.py` - orchestrates the clean Paper 1
  suite.
- `scripts/validate_qrmc_factorization.py` - validates the Q/R/M/C
  factorization estimator.
- `experiments/big_experiment/runner.py` - multi-agent runtime that logs
  Q/R/M/C events.
- `experiments/big_experiment/analyze_qrmc.py` - computes Q/R/M/C conditional
  rates and summaries.
- `experiments/big_experiment/exp_cadence_phase.py` - full cadence sweep.
- `experiments/big_experiment/exp_oracle_interventions.py` - oracle R/M/C
  interventions.
- `experiments/llm_collective/qrmc_llm_runner.py` - Q/R/M/C logger for the
  LLM TextNav diagnostic.
- `experiments/llm_collective/run_model_sweep.py` - local Ollama model sweep.
- `experiments/commnet_baseline/eval_with_qrmc.py` - Q/R/M/C evaluation of a
  learned communication baseline.

## Original Locations

These files were copied from `scripts/` and `experiments/`.

