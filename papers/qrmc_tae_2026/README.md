# Q/R/M/C TAE 2026 Version

This directory contains the NeurIPS 2026 TAE workshop version of the
Q/R/M/C paper.

Target:

- Workshop: TAE (Trust-AI-Eval): Can We Trust AI Evaluation?
- Format: NeurIPS 2026 workshop, double-blind.
- Main-paper limit: 8 pages excluding references and appendix.

Main files:

- `qrmc_tae_2026.tex` - main paper source.
- `qrmc_tae_2026.pdf` - compiled PDF.
- `qrmc_tae_2026.bib` - bibliography, including the added TAE framing references.
- `neurips_2026.sty` - official NeurIPS 2026 style file from the provided template.
- `figures/` - main-paper figures, including the restored Q/R/M/C contract and
  oracle-intervention figures from the earlier paper.
- `make_tae_figures.py` - regenerates the compact blinded-benchmark/repair and
  multi-agent mechanism figures from existing JSON results.
- `artifacts/reliability_verdict.json` - sampling and threshold sensitivity results computed from existing benchmark rows; includes input row-file hashes and row counts.
- `submission/qrmc_tae_anonymized_artifact.zip` - minimal anonymized artifact
  for local reproducibility review or a separate artifact field, if allowed.
- `submission/` - submission notes and legacy copied supplement/code files.

Build command from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error qrmc_tae_2026.tex
```

Figure generation from the repository root:

```sh
MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/mpl .venv/bin/python papers/qrmc_tae_2026/make_tae_figures.py
```

Reliability analysis:

```sh
.venv/bin/python experiments/qrmc_external/analyze_reliability.py
```

This reproduces Section 5.3 without new LLM trajectories from:

- `tmp/qrmc_external/rows_meta_llama.json` (`llama3.1:latest`, Ollama ID `46e0c10c039e`)
- `tmp/qrmc_external/rows_meta_qwen.json` (`qwen3:4b`, Ollama ID `359d7dd4bcda`)

The generated `reliability_verdict.json` records the exact input SHA-256
hashes, model IDs, scored-cell count, subsampling seed, and threshold scaling.

Current status:

- Uses `\usepackage[dblblindworkshop]{neurips_2026}`.
- Sets `\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}`.
- Compiled PDF is 9 pages total.
- Main content ends on page 7; references start on page 7 after the conclusion; appendix starts on page 8.
- Main paper includes four figures: evaluation contract, blinded benchmark plus repair utility, oracle interventions, and multi-agent mechanism.
- Final log has no unresolved citations, unresolved references, overfull boxes, or LaTeX errors. Remaining warnings are underfull page-break boxes from compact float placement.
