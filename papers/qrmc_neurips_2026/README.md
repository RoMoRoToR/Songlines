# Q/R/M/C NeurIPS 2026 Version

This directory is a NeurIPS 2026 conversion of the AAAI draft in
`../qrmc_aaai27/`.

Main files:

- `songlines_qrmc_neurips_2026.tex` - NeurIPS-formatted main paper.
- `songlines_qrmc_neurips_2026.pdf` - compiled submission-style PDF.
- `neurips_2026.sty` - official NeurIPS 2026 style file copied from the provided template directory.
- `checklist.tex` - NeurIPS paper checklist, with the template instruction block removed and draft answers filled in.
- `aaai27refs.bib` - bibliography copied from the AAAI version.
- `figures` - symlink to `../figures`.
- `submission/` - copied technical supplement and code supplement from the AAAI submission bundle; review names/content before a NeurIPS upload.

Build command:

```sh
pdflatex -interaction=nonstopmode -halt-on-error songlines_qrmc_neurips_2026.tex
bibtex songlines_qrmc_neurips_2026
pdflatex -interaction=nonstopmode -halt-on-error songlines_qrmc_neurips_2026.tex
pdflatex -interaction=nonstopmode -halt-on-error songlines_qrmc_neurips_2026.tex
```

Current status:

- The main text fits in the NeurIPS 9-page content limit; references begin on page 9 after the main conclusion.
- The full PDF is 17 pages including references and checklist.
- The final LaTeX log has no unresolved citations, unresolved references, or overfull boxes; only two underfull `vbox` warnings remain from float placement.
- Checklist item "Licenses for existing assets" is intentionally answered `No` until the paper explicitly audits license/terms text for MiniGrid, BabyAI, VMAS, agent frameworks, and model/tool assets.
