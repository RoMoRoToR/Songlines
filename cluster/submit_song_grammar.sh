#!/usr/bin/env bash
# UCSM stages 2-3 + seven-arm experiment (run ON SPHINX).
# Usage: bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_song_grammar.sh [--dry]
# CPU-only, deterministic, seed-sharded (see docs/FRONTIER_UCSM_2026-07-27.md §6).
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs" "$ROOT/tmp/cluster/song_grammar"
DRY=${1:-}

# name | cpus | mem | time | command
JOBS=$(cat <<'EOF'
u7_e10|2|8G|04:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 0 12 --episodes 10 --out tmp/cluster/song_grammar/u7
u7_e100_s0|2|8G|24:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 0 3 --episodes 100 --out tmp/cluster/song_grammar/u7
u7_e100_s3|2|8G|24:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 3 6 --episodes 100 --out tmp/cluster/song_grammar/u7
u7_e100_s6|2|8G|24:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 6 9 --episodes 100 --out tmp/cluster/song_grammar/u7
u7_e100_s9|2|8G|24:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 9 12 --episodes 100 --out tmp/cluster/song_grammar/u7
u7_e1000_s0|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 0 1 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s1|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 1 2 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s2|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 2 3 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s3|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 3 4 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s4|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 4 5 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s5|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 5 6 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s6|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 6 7 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s7|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 7 8 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s8|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 8 9 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s9|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 9 10 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s10|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 10 11 --episodes 1000 --out tmp/cluster/song_grammar/u7
u7_e1000_s11|2|16G|48:00:00|$PYBIN experiments/song_grammar/exp_u7_seven_arms.py --seeds 11 12 --episodes 1000 --out tmp/cluster/song_grammar/u7
u2_bandit|2|8G|48:00:00|$PYBIN experiments/song_grammar/exp_u2_bandit.py --train-seeds 0 24 --eval-seeds 100 110 --episodes 40 --out tmp/cluster/song_grammar/u2
u3_evolution|2|8G|72:00:00|$PYBIN experiments/song_grammar/exp_u3_evolution.py --generations 30 --pop 16 --episodes 25 --train-seeds 0 6 --holdout-seeds 200 208 --out tmp/cluster/song_grammar/u3
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
echo "Submitted. Monitor: squeue -u \$USER | grep sg_"
