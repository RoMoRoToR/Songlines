#!/usr/bin/env bash
# UE1-EVAL — full utility-estimator evaluation (run ON SPHINX).
# Usage: bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_utility_eval.sh [--dry]
# CPU-only, deterministic. Frozen estimator (features + lambda=1.0 from
# exp_ue1_utility_estimator.py); test seeds 100+/200+ are eval-only.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs" "$ROOT/tmp/cluster/song_grammar"
DRY=${1:-}

# name | cpus | mem | time | command
JOBS=$(cat <<'EOF'
ue1ev_paper|2|8G|04:00:00|$PYBIN experiments/song_grammar/exp_utility_estimator_eval.py --selfcheck --train-seeds 0 20 --test-seeds 100 110 --episodes 40 --horizon-episodes 120 --out tmp/cluster/song_grammar/ue1_eval_paper
ue1ev_wide|2|8G|12:00:00|$PYBIN experiments/song_grammar/exp_utility_estimator_eval.py --train-seeds 0 40 --test-seeds 100 130 --episodes 40 --horizon-episodes 200 --out tmp/cluster/song_grammar/ue1_eval_wide
ue1ev_fresh200|2|8G|12:00:00|$PYBIN experiments/song_grammar/exp_utility_estimator_eval.py --train-seeds 0 40 --test-seeds 200 230 --episodes 40 --horizon-episodes 200 --out tmp/cluster/song_grammar/ue1_eval_fresh200
ue1_degrade_wide|2|8G|48:00:00|$PYBIN experiments/song_grammar/exp_ue1_utility_estimator.py --train-seeds 0 20 --test-seeds 100 110 --degradation-seeds 100 120 --episodes 100 --out tmp/cluster/song_grammar/ue1_degrade_wide
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
echo "Submitted. Monitor: squeue -u \$USER | grep sg_ue1"
