#!/usr/bin/env bash
# One-time environment setup ON SPHINX. Wiki-blessed path: user-installed Miniconda
# in scratch (shared via NFS with all compute nodes). No sudo needed.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
MC=/mnt/tank/scratch/rzamotaev/miniconda3
PY=$MC/bin/python

if [ ! -x "$PY" ]; then
  echo "installing miniconda -> $MC"
  cd /mnt/tank/scratch/rzamotaev
  curl -sL -o mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash mc.sh -b -p "$MC"
  rm -f mc.sh
fi
"$MC/bin/pip" install -q numpy scipy pandas matplotlib
"$PY" - <<'EOF'
import numpy, scipy, pandas, matplotlib
print("deps OK:", numpy.__version__, scipy.__version__, pandas.__version__, matplotlib.__version__)
EOF
echo "Smoke test:"
cd "$ROOT"
PYTHONPATH=. "$PY" experiments/place_identity_demo/coordinate_free_matching.py --T 4 --N 3 --seeds 2 | tail -3
echo "SETUP DONE  (python: $PY)"
