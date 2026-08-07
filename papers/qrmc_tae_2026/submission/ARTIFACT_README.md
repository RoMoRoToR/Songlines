# Q/R/M/C TAE Reproducibility Artifact

This anonymized artifact contains the fixed episode-level records and
deterministic analysis code used for the TAE submission's blinded
fault-localization and Section 5.3 reliability analysis.

## Contents

- `experiments/qrmc_external/analyze_reliability.py` - recomputes the
  sampling and threshold sensitivity results from existing episode rows.
- `experiments/qrmc_external/diagnostic_baselines.py` - shared scoring
  utilities used by the reliability script.
- `experiments/qrmc_external/memory_house.py` - fault taxonomy and task
  environment definitions.
- `experiments/qrmc_external/run_meta_evaluation.py` - registered
  diagnostic thresholds and blind-study accounting.
- `data/qrmc_external/rows_meta_llama.json` - episode rows for
  `llama3.1:latest`, Ollama model ID `46e0c10c039e`.
- `data/qrmc_external/rows_meta_qwen.json` - episode rows for
  `qwen3:4b`, Ollama model ID `359d7dd4bcda`.
- `data/qrmc_external/reliability_verdict.json` - reproduced Section 5.3
  output, including input SHA-256 hashes.
- `data/qrmc_external/baselines_verdict.json` and
  `data/qrmc_external/repair_verdict.json` - diagnostic and repair
  summaries used by the submission.

## Reproduce Section 5.3

From the artifact root:

```sh
python -m pip install -r requirements.txt
python experiments/qrmc_external/analyze_reliability.py \
  --out data/qrmc_external/reliability_verdict_recomputed.json
```

The script uses 1000 without-replacement subsamples with seed `20260807`
by default. It does not run LLM trajectories.

Expected headline values:

- full-data accuracy `0.873`, macro-F1 `0.835`;
- sampling accuracy means `0.780`, `0.829`, `0.852`, `0.873` for
  `n=5,10,15,20`;
- threshold-scale accuracies `0.889`, `0.873`, `0.873`, `0.825`, `0.825`
  for scales `0.8,0.9,1.0,1.1,1.2`.
