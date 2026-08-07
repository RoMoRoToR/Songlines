#!/usr/bin/env bash
# Figure-1 real baseline: N1 fail-open noise sweep at consensus in {1,2,3}.
# consensus=1 = no consensus/provenance fix (climbs); >=2 = the fix (flat).
# Run ON SPHINX. CPU, deterministic, small.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs"
DRY=${1:-}
JOB=$ROOT/cluster/jobs/n1sweep.sbatch
cat > "$JOB" <<EOS
#!/bin/bash
#SBATCH --job-name=sg_n1sweep
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=$LOGS/n1sweep.%j.out
set -euo pipefail
export TMPDIR=/tmp
cd $ROOT
export PYTHONPATH=.
for C in 1 2 3; do
  $PYBIN experiments/song_grammar/exp_n1_noise.py --seeds 24 --consensus \$C \
      --out tmp/cluster/song_grammar/n1_c\$C
done
echo "JOB_DONE n1sweep"
EOS
if [ "$DRY" = "--dry" ]; then echo "[dry] would submit n1sweep"; else sbatch "$JOB"; fi
