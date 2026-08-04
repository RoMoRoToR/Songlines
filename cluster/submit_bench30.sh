#!/usr/bin/env bash
# Stage 8: 30-seed rerun of the main unified equal-budget benchmark
# (exp_b_unified). Run ON SPHINX.
# Usage: bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_bench30.sh [--dry]
# 6 policies x 30 test seeds (100-130) x e300 x 6 agents, seed-sharded
# 3x10. CPU-only, deterministic; test seeds 100+ never used for tuning.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
OUT=tmp/cluster/song_grammar/bench30
mkdir -p "$LOGS" "$ROOT/cluster/jobs" "$ROOT/$OUT"

DRY=${1:-}
CPUS=2; MEM=8G; TIME=03:00:00

POLICIES="independent songline_full decision_centric execution_path graph_memory learned_formation"
SHARDS="100:110 110:120 120:130"

for POL in $POLICIES; do
  for SH in $SHARDS; do
    A=${SH%:*}; B=${SH#*:}
    NAME="b30_${POL}_s${A}"
    JOB=$ROOT/cluster/jobs/$NAME.sbatch
    cat > "$JOB" <<EOS
#!/bin/bash
#SBATCH --job-name=sg_$NAME
#SBATCH --partition=main
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOGS/$NAME.%j.out
set -euo pipefail
export TMPDIR=/tmp
cd $ROOT
export PYTHONPATH=.
$PYBIN experiments/song_grammar/exp_b_unified.py \
  --policy $POL --seeds $A $B --episodes 300 --agents 6 --out $OUT
echo "JOB_DONE $NAME"
EOS
    if [ "$DRY" = "--dry" ]; then
      echo "[dry] would submit $NAME ($CPUS cpu, $MEM, $TIME): $POL seeds $A-$B"
    else
      sbatch "$JOB"
    fi
  done
done
