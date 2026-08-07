#!/usr/bin/env bash
# Package I: full resource accounting on the B-unified equal-budget
# benchmark (exp_resource_accounting wraps exp_b_unified.run_cell with
# metering; utility numbers stay byte-identical to bench30).
# Run ON SPHINX:
#   bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_resource_accounting.sh [--dry]
# 6 policies x 30 test seeds (100-130) x e300 x 6 agents, seed-sharded
# 3x10 -- same grid as submit_bench30.sh so the resource columns pair
# 1:1 with the published utility table. CPU-only, deterministic.
# IMPORTANT: 1 run per process at a time (cpu_time_s is a process
# delta), which the sharding already guarantees; do not co-schedule
# other work in the same python process.
# Afterwards (login node is fine, seconds):
#   PYTHONPATH=. $PYBIN scripts/analyze_resource_accounting.py \
#       --dir tmp/cluster/song_grammar/resource_accounting
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
OUT=tmp/cluster/song_grammar/resource_accounting
mkdir -p "$LOGS" "$ROOT/cluster/jobs" "$ROOT/$OUT"

DRY=${1:-}
# wall-clock timing wants an unshared core: 2 cpus like bench30 but
# generous time (metering + pickle snapshots add <10% on smoke).
CPUS=2; MEM=8G; TIME=04:00:00

POLICIES="independent songline_full decision_centric execution_path graph_memory learned_formation"
SHARDS="100:110 110:120 120:130"

for POL in $POLICIES; do
  for SH in $SHARDS; do
    A=${SH%:*}; B=${SH#*:}
    NAME="ra_${POL}_s${A}"
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
$PYBIN experiments/song_grammar/exp_resource_accounting.py \
  --policy $POL --seeds $A $B --episodes 300 --agents 6 \
  --snap-every 25 --out $OUT
echo "JOB_DONE $NAME"
EOS
    if [ "$DRY" = "--dry" ]; then
      echo "[dry] would submit $NAME ($CPUS cpu, $MEM, $TIME): $POL seeds $A-$B"
    else
      sbatch "$JOB"
    fi
  done
done
