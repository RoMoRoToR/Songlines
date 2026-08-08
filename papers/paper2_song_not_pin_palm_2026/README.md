# Paper 2 PALM 2026 Submission Draft

Target workshop: PALM: Personalized, Aligned, Long-Term Memory for AI Systems, NeurIPS 2026.

Workshop page: https://palm-neurips-2026.github.io/

## Files

- `paper2_song_not_pin_palm_2026.tex` - PALM/NeurIPS workshop manuscript.
- `paper2_song_not_pin_palm_2026.pdf` - compiled submission draft.
- `neurips_2026.sty` - local NeurIPS 2026 style file.
- `figures/` - figure assets copied from the source manuscript, preserved as both PDF and PNG where available.

## Format Status

- Uses `\usepackage[dblblindworkshop]{neurips_2026}`.
- Sets `\workshoptitle{PALM: Personalized, Aligned, Long-Term Memory for AI Systems}`.
- Uses anonymous authors.
- Current PDF has 15 pages total.
- Main text concludes on page 11; references begin on page 11; appendix begins on page 12.
- This version is a full translation of the Russian author rewrite and is currently over the PALM full-paper limit of up to 9 pages excluding references and supplementary material.

## Reviewer-Hardening Revision

- The manuscript body has been replaced with an English translation of `/Users/taniyashuba/Desktop/Статьи/Статья_2Songline 2026 ИТМО/The_Song_Not_the_Pin_PALM_RU_author_rewrite.docx`.
- The abstract and introduction center one thesis: the transferable unit is a provenance-conditioned executable route, not a target record alone.
- Exact trust-staleness and rupture matches are framed as internal predictive consistency checks for the specified contract, not as universal memory laws.
- The LLM section is framed as an LLM-over-symbolic-memory portability check rather than broad LLM-memory evidence.
- Related work includes a positioning table over identity, route, provenance, validity, coordination, and counterfactual evaluation.
- Limitations explicitly note the missing reservation-only/rollback-only protocol ablation and the unmeasured coverage-vs-fail-open frontier.
- The introduction now includes a compact action-contract table mapping identity, transport, provenance, validity, and authority to reviewer-readable questions and failure modes.
- The related-work positioning table has been tightened around contemporary memory and agent-system paradigms.
- Appendix material now includes a 2,520-episode decentralized coordination-baseline sweep from `tmp/cluster/coord_baselines`, comparing the linked reservation-plus-rollback mechanism against random priority, nearest-agent-wins, greedy assignment, backoff-only, and soft occupancy alternatives.
- No standalone reservation-only or rollback-only primitive ablation is claimed.

## Restored Figures

Main text:

- `fig_warp_law.pdf`
- `fig_warp_drive.pdf`
- `fig_route_cliff.pdf`

Appendix:

- `fig_warp_strata.pdf`
- `fig_warp_universality.pdf`
- `fig_route_hazard.pdf`
- `fig_route_identity.pdf`

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper2_song_not_pin_palm_2026.tex
```

The current build log has no undefined citations, undefined references, overfull boxes, or LaTeX errors. It only reports the usual float-placement warning for one `[h]` float.
