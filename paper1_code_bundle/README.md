# Paper 1 Code Bundle

This folder is a readable copy of the code used for Paper 1:
Q/R/M/C evaluation, semantic/symbolic memory, and peer/distributed memory.

The original source files were not moved or deleted. This bundle is only a
navigation-friendly copy.

## Folders

- `01_qrmc_evaluation/` - Q/R/M/C measurement code, experiment runners,
  analysis scripts, LLM TextNav diagnostics, CommNet Q/R/M/C evaluation.
- `02_semantic_symbolic_memory/` - semantic and symbolic memory modules from
  `songline_drive`, plus the collective semantic memory experiment.
- `03_peer_distributed_memory/` - peer broadcast, distributed consensus,
  independent memory modules, and small example experiments.

## Recommended Reading Order

1. `01_qrmc_evaluation/scripts/run_paper1_clean_experiments.py`
2. `01_qrmc_evaluation/experiments/big_experiment/runner.py`
3. `01_qrmc_evaluation/experiments/big_experiment/analyze_qrmc.py`
4. `02_semantic_symbolic_memory/songline_drive/symbolic_memory.py`
5. `02_semantic_symbolic_memory/songline_drive/graph_memory.py`
6. `03_peer_distributed_memory/peer_memory/peer_runtime.py`
7. `03_peer_distributed_memory/distributed_memory/consensus_layer.py`

## Paper

The corresponding manuscript is:

`docs/Formatting_Instructions_For_NeurIPS_2026/songlines_qrmc_measurement_framework.tex`

