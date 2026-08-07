#!/usr/bin/env bash
# Package F: private-frame correspondence stress surface (exp_frame_stress).
# Dense level grids (--full): tag FN/FP noise curves, missing-landmark
# curves, repeated-constellation fail-closed checks (C=2..4 interior +
# @border), tag-misalignment arms (permutation / synonyms with partial
# translator / missing concepts / one-to-many coarsening).
# Run ON SPHINX. CPU, deterministic, single process.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs"
DRY=${1:-}
JOB=$ROOT/cluster/jobs/frame_stress.sbatch
cat > "$JOB" <<EOS
#!/bin/bash
#SBATCH --job-name=sg_frame_stress
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=$LOGS/frame_stress.%j.out
set -euo pipefail
export TMPDIR=/tmp
cd $ROOT
export PYTHONPATH=.
$PYBIN experiments/warp/exp_frame_stress.py --full --seeds 20 \
    --out tmp/cluster/warp/frame_stress_full
echo "JOB_DONE frame_stress"
EOS
if [ "$DRY" = "--dry" ]; then echo "[dry] would submit frame_stress"; else sbatch "$JOB"; fi
