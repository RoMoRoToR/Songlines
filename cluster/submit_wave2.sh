#!/usr/bin/env bash
# Wave 1b (CPU: torch/minigrid deps) + Wave 2 (GPU: LLM) suite. Run ON SPHINX.
# Usage: bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_wave2.sh [--dry]
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
SCRATCH=/mnt/tank/scratch/rzamotaev
PYBIN=$SCRATCH/miniconda3/bin/python
PYGPU=$SCRATCH/miniconda3/envs/gpu/bin/python
LOGS=$ROOT/cluster/logs
mkdir -p "$LOGS" "$ROOT/cluster/jobs"
DRY=${1:-}

submit() { # name partition gres cpus mem time cmd
  local NAME=$1 PART=$2 GRES=$3 CPUS=$4 MEM=$5 TIME=$6 CMD=$7
  local JOB=$ROOT/cluster/jobs/$NAME.sbatch
  cat > "$JOB" <<EOS
#!/bin/bash
#SBATCH --job-name=sl_$NAME
#SBATCH --partition=$PART
${GRES:+#SBATCH --gres=$GRES}
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOGS/$NAME.%j.out
set -euo pipefail
PYBIN=$PYBIN
PYGPU=$PYGPU
cd $ROOT
mkdir -p tmp/cluster
export PYTHONPATH=. MPLBACKEND=Agg
export TMPDIR=/tmp   # node-local: NFS tmpdirs break textworld rmtree (.nfsXXXX)
export HF_HOME=$SCRATCH/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ALFWORLD_DATA=$SCRATCH/alfworld_data ALFWORLD_CONFIG=$SCRATCH/alfworld_config.yaml
$CMD
echo "JOB_DONE $NAME"
EOS
  if [ "$DRY" = "--dry" ]; then echo "[dry] $NAME"; else sbatch "$JOB"; fi
}

# ---- Wave 1b (CPU, torch/minigrid) ----
submit vmas_full        main ""            8  16G 12:00:00 '$PYBIN experiments/vmas_portability/run_vmas_full.py --seeds 40 --out_dir tmp/cluster/vmas_full > tmp/cluster/vmas_full.txt'
submit mappo            main ""            16 32G 24:00:00 '$PYBIN experiments/mappo_baseline/train_mappo.py --total_updates 300 --out_dir tmp/cluster/mappo > tmp/cluster/mappo_train.txt && $PYBIN experiments/commnet_baseline/eval_with_qrmc.py --policy_path tmp/cluster/mappo/mappo_policy.pt --n_episodes 50 --out_dir tmp/cluster/mappo_eval > tmp/cluster/mappo_eval.txt'
submit singleagent_30   main ""            8  16G 24:00:00 '$PYBIN scripts/benchmark_symbolic_memory_article.py --tasks water rest goal_region hazard_recovery --num_seeds 30 --episodes 8 --assist_modes off on --out_dir tmp/cluster/singleagent_30'
submit theta_full       main ""            4  8G  12:00:00 '$PYBIN experiments/big_experiment/theta_ablation.py > tmp/cluster/theta_full.txt'

# ---- Wave 2 (GPU: real LLM) ----
submit alfworld_qwen3b  gpu  gpu:1         8  32G 12:00:00 '$PYGPU experiments/alfworld_qrmc/run_alfworld_qrmc.py --model Qwen/Qwen2.5-3B-Instruct --episodes 25 --out_dir tmp/cluster/alfworld_qwen3b'
submit alfworld_qwen7b  gpu  gpu:1         8  48G 16:00:00 '$PYGPU experiments/alfworld_qrmc/run_alfworld_qrmc.py --model Qwen/Qwen2.5-7B-Instruct --episodes 25 --out_dir tmp/cluster/alfworld_qwen7b'
submit textnav_hf_3b    gpu  gpu:1         8  32G 12:00:00 'for SL in 6 8 12 25; do $PYGPU -m experiments.llm_collective.run_model_sweep --backend hf --models Qwen/Qwen2.5-3B-Instruct --episodes 25 --step_limit $SL --out_dir tmp/cluster/textnav_hf_sl$SL; done'
submit textnav_hf_7b    gpu  gpu:1         8  48G 16:00:00 'for SL in 6 12; do $PYGPU -m experiments.llm_collective.run_model_sweep --backend hf --models Qwen/Qwen2.5-7B-Instruct --episodes 25 --step_limit $SL --out_dir tmp/cluster/textnav7b_sl$SL; done'

echo "Wave1b+2 submitted. Monitor: squeue -u \$USER"
