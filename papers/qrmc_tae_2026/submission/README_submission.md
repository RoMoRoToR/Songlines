# TAE 2026 submission bundle - Q/R/M/C

Target workshop: TAE (Trust-AI-Eval): Can We Trust AI Evaluation?, NeurIPS 2026.

OpenReview upload:

- Submit only `../qrmc_tae_2026.pdf` as the workshop submission PDF.
- Do not upload artifact or supplement files with the initial submission unless
  OpenReview explicitly provides a separate
  artifact/supplement field that is allowed by the workshop.

Local bundle mapping:

| File | Intended upload |
|---|---|
| `../qrmc_tae_2026.pdf` | Main single-PDF submission |
| `qrmc_tae_anonymized_artifact.zip` | Local anonymized reproducibility artifact; use only if a separate artifact field is allowed |
| `TechnicalSupplement_QRMC.pdf` | Local source material only; do not upload as a separate initial-submission file |
| `code_supplement.zip` | Legacy full-code bundle copied from the earlier submission; superseded by `qrmc_tae_anonymized_artifact.zip` for TAE |

Pre-submission checks:

- [x] Main paper uses `\usepackage[dblblindworkshop]{neurips_2026}`.
- [x] Main paper sets `\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}`.
- [x] Confirm the compiled main text fits the 8-page TAE full-paper limit, excluding references and appendix.
- [x] Update `code_supplement.zip` with the reliability analysis script, source row files, and generated verdict JSONs used by this version.
- [x] Build `qrmc_tae_anonymized_artifact.zip` with neutral root, source row files, reproduction script, verdict JSONs, row hashes, and Ollama model IDs.
- [x] Initial OpenReview submission is a single anonymized PDF.
- [ ] Review copied supplement files for stale conference names, dates, anonymity, and artifact consistency.
