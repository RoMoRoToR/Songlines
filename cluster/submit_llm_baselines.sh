#!/usr/bin/env bash
# H -- LLM context baselines at a fixed token budget (run ON SPHINX).
# Usage: bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_llm_baselines.sh [--dry]
#
# Answers the reviewer's "raw baseline too weak / compare at fixed
# token budget" claim: 6 arms (raw / raw_instr / rolling summary /
# vector retrieval / programmatic table / Songlines graph) + none
# control, same model, same evidence, same budget, same questions.
#
# Models as in the article's L1 numbers:
#   * Qwen2.5-3B-Instruct -- here, hf backend, GPU nodes (weights are
#     pre-staged offline in $SCRATCH/hf_home; HF_HUB_OFFLINE=1).
#   * llama3.1 -- LOCAL ollama (cluster nodes have no ollama daemon;
#     the published llama3.1 L1 numbers were produced locally too):
#       PYTHONPATH=. python experiments/llm_context_baselines/run_baselines.py \
#         --config experiments/llm_context_baselines/configs/full_grid.json \
#         --run llama31_long   # and llama31_short
#
# Known cluster pitfalls baked in (see memory/cluster_access.md and
# submit_wave2.sh): TMPDIR=/tmp (NFS tmpdirs break rmtree), gpu env
# python (base torch cu130 does not see the GPU; envs/gpu is cu124),
# HF_HOME on scratch + offline flags, 7B fp16 ONLY on nike/kali
# (24 GB cards) -- the optional 7B job below is pinned with -w nike.
set -euo pipefail
ROOT=/mnt/tank/scratch/rzamotaev/songlines
SCRATCH=/mnt/tank/scratch/rzamotaev
PYBIN=$SCRATCH/miniconda3/bin/python
PYGPU=$SCRATCH/miniconda3/envs/gpu/bin/python
LOGS=$ROOT/cluster/logs
CFG=experiments/llm_context_baselines/configs/full_grid.json
mkdir -p "$LOGS" "$ROOT/cluster/jobs" "$ROOT/tmp/cluster/llm_context_baselines"
DRY=${1:-}

submit() { # name partition extra_sbatch cpus mem time cmd
  local NAME=$1 PART=$2 EXTRA=$3 CPUS=$4 MEM=$5 TIME=$6 CMD=$7
  local JOB=$ROOT/cluster/jobs/$NAME.sbatch
  cat > "$JOB" <<EOS
#!/bin/bash
#SBATCH --job-name=hx_$NAME
#SBATCH --partition=$PART
${EXTRA:+$EXTRA}
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOGS/$NAME.%j.out
set -euo pipefail
cd $ROOT
export PYTHONPATH=. MPLBACKEND=Agg
export TMPDIR=/tmp
export HF_HOME=$SCRATCH/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$CMD
echo "JOB_DONE $NAME"
EOS
  if [ "$DRY" = "--dry" ]; then
    echo "[dry] would submit $NAME ($PART, $CPUS cpu, $MEM, $TIME)"
  else
    sbatch "$JOB"
  fi
}

# 1) CPU plumbing check (stub backend, no LLM): partition main, base python.
submit ctxbase_stub main "" 2 8G 01:00:00 \
  "$PYBIN experiments/llm_context_baselines/run_baselines.py --backend stub --mode long --layouts 6 --budgets 0 128 512 --out tmp/cluster/llm_context_baselines/stub_check"

# 2) Qwen2.5-3B, short + long histories (the article's model; 3B fits
#    any gpu node).  Prompts stay <2k tokens, far below the sm75 cap.
submit ctxbase_qwen3b_short gpu "#SBATCH --gres=gpu:1" 8 32G 12:00:00 \
  "$PYGPU experiments/llm_context_baselines/run_baselines.py --config $CFG --run qwen3b_short"
submit ctxbase_qwen3b_long gpu "#SBATCH --gres=gpu:1" 8 32G 16:00:00 \
  "$PYGPU experiments/llm_context_baselines/run_baselines.py --config $CFG --run qwen3b_long"

# 3) OPTIONAL robustness point: Qwen2.5-7B (fp16 needs the 24 GB
#    cards -> nike/kali ONLY). Uncomment to include.
# submit ctxbase_qwen7b_long gpu "#SBATCH --gres=gpu:1
# #SBATCH -w nike" 8 48G 24:00:00 \
#   "$PYGPU experiments/llm_context_baselines/run_baselines.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --mode long --layouts 12 --budgets 0 96 192 384 768 --out tmp/cluster/llm_context_baselines/qwen7b_long"

echo "Submitted. Monitor: squeue -u \$USER | grep hx_"
echo "Collect:  tmp/cluster/llm_context_baselines/*/results.json"
