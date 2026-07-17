#!/usr/bin/env bash
# Generate + submit the full CPU experiment suite via SLURM (run ON SPHINX).
# Usage: bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_all.sh [--dry]
# Wave 1: pure numpy/scipy/pandas/matplotlib experiments, scaled-up seeds.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
PYBIN=/mnt/tank/scratch/rzamotaev/miniconda3/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs"
DRY=${1:-}

# name | cpus | mem | time | command (run from $ROOT, venv active, PYTHONPATH=.)
JOBS=$(cat <<'EOF'
cadence_full|32|64G|12:00:00|$PYBIN experiments/big_experiment/exp_cadence_phase.py --mode full --workers 32 --out_dir tmp/cluster/cadence_full
extra_K|16|32G|06:00:00|$PYBIN experiments/big_experiment/exp_extra_K.py --workers 16 --out_dir tmp/cluster/extra_K
scale_N12|32|64G|08:00:00|$PYBIN experiments/big_experiment/exp_scale_N12.py --workers 32 --out_dir tmp/cluster/scale_N12
cadence_robustness|16|32G|12:00:00|$PYBIN experiments/big_experiment/exp_cadence_robustness.py --seeds 100 --workers 16 > tmp/cluster/cadence_robustness.txt
eps_sensitivity|8|16G|12:00:00|$PYBIN experiments/big_experiment/exp_eps_sensitivity.py --seeds 50 > tmp/cluster/eps_sensitivity.txt
oracle_interventions|16|32G|06:00:00|$PYBIN experiments/big_experiment/exp_oracle_interventions.py --out_dir tmp/cluster/oracle --workers 16
assumption1_stress|2|8G|01:00:00|$PYBIN experiments/big_experiment/assumption1_stress_test.py > tmp/cluster/assumption1.txt
route_r0|4|8G|04:00:00|$PYBIN experiments/warp/exp_route_warp_r0.py > tmp/cluster/route_r0.txt
route_r1|8|16G|08:00:00|$PYBIN experiments/warp/exp_route_warp_r1.py --seeds 50 > tmp/cluster/route_r1.txt
route_r2|8|16G|08:00:00|$PYBIN experiments/warp/exp_route_warp_r2.py --seeds 10 > tmp/cluster/route_r2.txt
route_r3|8|16G|08:00:00|$PYBIN experiments/warp/exp_route_warp_r3.py --seeds 50 > tmp/cluster/route_r3.txt
w7_semantic_identity|4|8G|04:00:00|$PYBIN experiments/warp/exp_warp_semantic_identity.py > tmp/cluster/w7.txt
w8_semantic_stack|8|16G|08:00:00|$PYBIN experiments/warp/exp_warp_semantic_stack.py --seeds 50 > tmp/cluster/w8.txt
alignment_defect|4|8G|02:00:00|$PYBIN experiments/warp/alignment_defect.py > tmp/cluster/alignment_defect.txt
semantic_cadence|8|16G|08:00:00|$PYBIN experiments/warp/exp_semantic_cadence_qrmc.py --N 6 --T 2 --seeds 40 > tmp/cluster/semantic_cadence.txt
symbol_alignment|4|8G|02:00:00|$PYBIN experiments/warp/symbol_alignment.py --seeds 100 > tmp/cluster/symbol_alignment.txt
coord_free_matching|4|8G|04:00:00|$PYBIN experiments/place_identity_demo/coordinate_free_matching.py --T 6 --N 4 --seeds 100 > tmp/cluster/coord_free.txt
phase_diagram|4|8G|02:00:00|$PYBIN experiments/collective_semantic_memory/phase_diagram_trust_cadence.py --seeds 50 --fig tmp/cluster/fig_phase_diagram.pdf > tmp/cluster/phase_diagram.txt
csm_benchmark|16|32G|12:00:00|$PYBIN -m experiments.collective_semantic_memory.run_csm_vs_peer --out_dir tmp/cluster/csm_benchmark
EOF
)

echo "$JOBS" | while IFS='|' read -r NAME CPUS MEM TIME CMD; do
  [ -z "$NAME" ] && continue
  JOB=$ROOT/cluster/jobs/$NAME.sbatch
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
echo "All jobs generated in cluster/jobs/. Monitor: squeue -u \$USER"
