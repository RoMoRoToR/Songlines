#!/usr/bin/env bash
# Package E — operation ablations of the five formation ops (run ON SPHINX).
# Usage: bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_operation_ablations.sh [--dry]
# CPU-only, deterministic. Seed blocks 400/800/1200 are DISJOINT from the
# local smoke (seeds 0-21, incl. the +100..260 novel-world and +200
# goal-world offsets), so the v2-registered E.6/E.7 verdicts are
# confirmed on unseen worlds. 16 valid worlds per shard = 48 total.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs" "$ROOT/tmp/cluster/operation_ablations"
DRY=${1:-}

# name | cpus | mem | time | command
JOBS=$(cat <<'EOF'
opabl_s400|2|8G|04:00:00|$PYBIN experiments/song_grammar/exp_operation_ablations.py --seed-start 400 --seed-scan 300 --n-worlds 16 --out tmp/cluster/operation_ablations/shard400
opabl_s800|2|8G|04:00:00|$PYBIN experiments/song_grammar/exp_operation_ablations.py --seed-start 800 --seed-scan 300 --n-worlds 16 --out tmp/cluster/operation_ablations/shard800
opabl_s1200|2|8G|04:00:00|$PYBIN experiments/song_grammar/exp_operation_ablations.py --seed-start 1200 --seed-scan 300 --n-worlds 16 --out tmp/cluster/operation_ablations/shard1200
EOF
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
echo "Submitted. Monitor: squeue -u \$USER | grep sg_opabl"
