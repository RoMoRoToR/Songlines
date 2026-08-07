#!/usr/bin/env bash
# Package A: coordination baselines (part 1) + contention interventions
# (part 2), full sweeps. Run ON SPHINX:
#   bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_coord_baselines.sh [--dry]
#
# Full designs:
#   coord_baselines : 3 cells {(3,2),(5,3),(8,5)} x 2 layouts x 20 seeds
#                     x K{1,2,4} x 7 arms                 = 2520 episodes
#   contention_interv: 2 cells {(3,2),(5,3)} x 2 layouts x 20 seeds
#                     x K{1,4} x 5 modes                  =  800 episodes
# Local smoke rate ~0.04 s/episode at 8 workers -> minutes-scale jobs;
# generous walltime for cluster-node variance.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
SCRATCH=/mnt/tank/scratch/rzamotaev
PYBIN=$SCRATCH/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs"
DRY=${1:-}

submit() { # name cpus mem time cmd
  local NAME=$1 CPUS=$2 MEM=$3 TIME=$4 CMD=$5
  local JOB=$ROOT/cluster/jobs/$NAME.sbatch
  cat > "$JOB" <<EOS
#!/bin/bash
#SBATCH --job-name=sl_$NAME
#SBATCH --partition=main
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOGS/$NAME.%j.out
set -euo pipefail
PYBIN=$PYBIN
cd $ROOT
mkdir -p tmp/cluster
export PYTHONPATH=. MPLBACKEND=Agg
export TMPDIR=/tmp   # node-local: NFS tmpdirs leave .nfsXXXX turds on rmtree
$CMD
echo "JOB_DONE $NAME"
EOS
  if [ "$DRY" = "--dry" ]; then echo "[dry] $NAME -> $JOB"; else sbatch "$JOB"; fi
}

submit coord_baselines   16 16G 04:00:00 '$PYBIN experiments/coordination/exp_coord_baselines.py --seeds 20 --workers 16 --out_dir tmp/cluster/coord_baselines > tmp/cluster/coord_baselines.txt'
submit contention_interv 16 16G 02:00:00 '$PYBIN experiments/coordination/exp_contention_interventions.py --seeds 20 --workers 16 --out_dir tmp/cluster/contention_interv > tmp/cluster/contention_interv.txt'

echo "Package A submitted. Monitor: squeue -u \$USER"
