#!/usr/bin/env bash
# Two figure sweeps: (K x trust) success grid + MAPPO convergence curve.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs; mkdir -p "$LOGS" "$ROOT/cluster/jobs"
DRY=${1:-}
# --- job A: K x trust sweep (CPU, minutes) ---
cat > $ROOT/cluster/jobs/ktrust.sbatch <<EOS
#!/bin/bash
#SBATCH --job-name=sg_ktrust
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=$LOGS/ktrust.%j.out
set -euo pipefail
export TMPDIR=/tmp; cd $ROOT; export PYTHONPATH=.
$PYBIN experiments/collective_semantic_memory/exp_ktrust_sweep.py \
    --seeds 8 --out tmp/cluster/song_grammar/ktrust
echo "JOB_DONE ktrust"
EOS
# --- job B: MAPPO convergence curve (CPU, long) ---
cat > $ROOT/cluster/jobs/mappo.sbatch <<EOS
#!/bin/bash
#SBATCH --job-name=sg_mappo
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --output=$LOGS/mappo.%j.out
set -euo pipefail
export TMPDIR=/tmp; cd $ROOT; export PYTHONPATH=.
$PYBIN experiments/mappo_baseline/train_mappo.py \
    --total_updates 300 --rollouts_per_update 64 --out_dir tmp/cluster/mappo
echo "JOB_DONE mappo"
EOS
if [ "$DRY" = "--dry" ]; then echo "[dry] ktrust + mappo"; else sbatch $ROOT/cluster/jobs/ktrust.sbatch; sbatch $ROOT/cluster/jobs/mappo.sbatch; fi
