#!/usr/bin/env bash
# Package J: neural baseline with EXPLICIT Q/R/M/C interfaces -- full training.
# Run ON SPHINX:  bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_neural_explicit.sh [--dry]
#
# 3 seeds x 400 PPO updates x 64 rollouts/update on the N=3/M=2 scarcity
# scenario (same env + obs encoding as mappo_baseline), plus one
# lock-shaping variant (disclosed potential-based shaping, seed 0).
# The trainer dumps neural_explicit_curve.json EVERY --dump_every=10 updates
# (not only at the end), so partial curves survive preemption/timeout.
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
export TMPDIR=/tmp   # node-local: NFS tmpdirs are a known footgun
$CMD
echo "JOB_DONE $NAME"
EOS
  if [ "$DRY" = "--dry" ]; then echo "[dry] $NAME"; else sbatch "$JOB"; fi
}

# ~25-40 s/update locally at 64 rollouts => 400 updates ~ 3-5 h; 24 h is safe.
for SEED in 0 1 2; do
  submit neuralexp_s$SEED 8 16G 24:00:00 \
    "\$PYBIN experiments/neural_explicit_baseline/train_neural_explicit.py \
       --total_updates 400 --rollouts_per_update 64 --seed $SEED \
       --dump_every 10 --dump_events_episodes 10 \
       --out_dir tmp/cluster/neural_explicit_s$SEED \
       > tmp/cluster/neural_explicit_s$SEED.txt"
done

# Disclosed potential-based lock-shaping variant (ablation, seed 0).
submit neuralexp_shaped 8 16G 24:00:00 \
  "\$PYBIN experiments/neural_explicit_baseline/train_neural_explicit.py \
     --total_updates 400 --rollouts_per_update 64 --seed 0 --lock_shaping 0.05 \
     --dump_every 10 --dump_events_episodes 10 \
     --out_dir tmp/cluster/neural_explicit_shaped \
     > tmp/cluster/neural_explicit_shaped.txt"

echo "neural_explicit jobs submitted. Monitor: squeue -u \$USER"
