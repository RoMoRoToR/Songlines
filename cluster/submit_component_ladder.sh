#!/usr/bin/env bash
# Package G: staged component ladder (CL) + clock alternatives (CA)
# of Songline Memory Runtime v1 (run ON SPHINX).
# Usage: bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_component_ladder.sh [--dry]
# CPU-only, deterministic, seed-sharded; paired assignment streams
# per seed; 12 seeds x 300 episodes x 6 agents (S1-S3 protocol).
#   CL: experiments/song_grammar/exp_component_ladder.py
#       7 ladder steps + 5 leave-one-component-out, per-seed shards.
#   CA: experiments/song_grammar/exp_clock_alternatives.py
#       7 clock arms on the S2 drift setting, per-seed shards.
# Registered predictions live in the experiment docstrings and are
# re-emitted as {cl,ca}_registered.json next to the shards.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs" \
  "$ROOT/tmp/cluster/song_grammar/cl" \
  "$ROOT/tmp/cluster/song_grammar/ca"
DRY=${1:-}

# name | cpus | mem | time | command
JOBS=$(
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
  e=$((s + 1))
  echo "cl_e300_s$s|2|8G|12:00:00|\$PYBIN experiments/song_grammar/exp_component_ladder.py --seeds $s $e --episodes 300 --agents 6 --out tmp/cluster/song_grammar/cl"
  echo "ca_e300_s$s|2|16G|24:00:00|\$PYBIN experiments/song_grammar/exp_clock_alternatives.py --seeds $s $e --episodes 300 --agents 6 --out tmp/cluster/song_grammar/ca"
done
)

echo "$JOBS" | while IFS='|' read -r NAME CPUS MEM TIME CMD; do
  [ -z "$NAME" ] && continue
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
PYBIN=$PYBIN
export TMPDIR=/tmp
cd $ROOT
export PYTHONPATH=.
$CMD
echo "JOB_DONE $NAME"
EOS
  if [ "$DRY" = "--dry" ]; then
    echo "[dry] would submit $NAME ($CPUS cpu, $MEM, $TIME)"
  else
    sbatch "$JOB"
  fi
done
echo "Submitted. Monitor: squeue -u \$USER | grep sg_"
