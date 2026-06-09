#!/usr/bin/env bash
#
# Prepare a deanonymized-safe submission bundle.
#
# Strips the internal codename "songline(s)" from the submitted PDF and
# its sources, so that a Google search for the codename cannot trace
# the paper back to the repo. Produces a self-contained `submission/`
# directory with paper.pdf + supplementary code.
#
# Run from repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/docs/Formatting_Instructions_For_NeurIPS_2026"
OUT_DIR="${REPO_ROOT}/tmp/submission"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/figures"

# Copy the .tex and rename
cp "${SRC_DIR}/songlines_symbolic_memory.tex" "${OUT_DIR}/paper.tex"
cp "${SRC_DIR}/checklist.tex" "${OUT_DIR}/checklist.tex"
cp "${SRC_DIR}/neurips_2026.sty" "${OUT_DIR}/neurips_2026.sty"

# Copy figures into renamed directory
cp "${SRC_DIR}/figures/"*.{pdf,png} "${OUT_DIR}/figures/" 2>/dev/null || true

# Sanitize source: defensive guard — figures dir and scripts have already
# been renamed in the working tree, so these substitutions are no-ops for
# the current source; kept so the script also works on any older revision
# of the .tex still containing the codename.
sed -i.bak \
    -e 's|songlines_symbolic_memory_figures|figures|g' \
    -e 's|compare_songline_|compare_semnav_|g' \
    -e 's|songline|semnav|g' \
    "${OUT_DIR}/paper.tex"
rm "${OUT_DIR}/paper.tex.bak"

# Strip internal TODO/codename comments from the submitted source
OUT_DIR="${OUT_DIR}" python3 - <<'PYEOF'
import os, re
path = os.environ["OUT_DIR"]
src = open(f"{path}/paper.tex").read()
out_lines = []
skip_block = False
for line in src.splitlines():
    if re.match(r"^\s*% TODO\(deanonymization\)", line):
        skip_block = True
        continue
    if skip_block:
        if line.startswith("%"):
            continue
        skip_block = False
    line = re.sub(r"\s*%\s*TODO\(deanonymization\).*$", "", line)
    out_lines.append(line)
open(f"{path}/paper.tex", "w").write("\n".join(out_lines) + "\n")
PYEOF

# Compile sanitized version twice (refs)
cd "${OUT_DIR}"
pdflatex -interaction=nonstopmode paper.tex > /dev/null
pdflatex -interaction=nonstopmode paper.tex > /dev/null
echo "→ ${OUT_DIR}/paper.pdf"
ls -la paper.pdf

# Audit: confirm no 'songline' left in compiled PDF or source
echo "--- audit: 'songline' occurrences in submission bundle ---"
grep -rn "songline" "${OUT_DIR}" || echo "(none — clean)"
echo "--- PDF metadata Author/Creator ---"
mdls -name kMDItemAuthors -name kMDItemCreator "${OUT_DIR}/paper.pdf" 2>/dev/null \
    || pdfinfo "${OUT_DIR}/paper.pdf" 2>/dev/null | grep -iE "author|creator" \
    || echo "(no metadata extractor available)"
