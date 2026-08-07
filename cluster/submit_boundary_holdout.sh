#!/usr/bin/env bash
# Package B: parameter-disjoint hold-out of the admissibility/rupture
# boundary (exp_boundary_holdout_v2).  75-cell registered-prediction
# grid: all 54 positive combos over unseen trusts/alpha/tau/conf/d,
# never-cells, pruning-binding cells (alpha=0.03) and a message-delay
# arm.  Predictions are written to disk before the first episode.
# Run ON SPHINX. CPU, deterministic, single process.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs"
DRY=${1:-}
JOB=$ROOT/cluster/jobs/boundary_holdout_v2.sbatch
cat > "$JOB" <<EOS
#!/bin/bash
#SBATCH --job-name=sg_boundary_holdout_v2
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=$LOGS/boundary_holdout_v2.%j.out
set -euo pipefail
export TMPDIR=/tmp
cd $ROOT
export PYTHONPATH=.
$PYBIN experiments/warp/exp_boundary_holdout_v2.py \
    --out tmp/cluster/warp/boundary_holdout_v2
echo "JOB_DONE boundary_holdout_v2"
EOS
if [ "$DRY" = "--dry" ]; then echo "[dry] would submit boundary_holdout_v2"; else sbatch "$JOB"; fi
