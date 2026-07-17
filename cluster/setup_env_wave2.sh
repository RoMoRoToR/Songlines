#!/usr/bin/env bash
# Wave 1b/2 environment ON SPHINX: torch+transformers+alfworld+vmas+minigrid,
# plus offline staging (ALFWorld data, HF weights, alfworld config) to scratch.
# GPU compute nodes may lack internet -> everything is pre-staged here.
set -euo pipefail
SCRATCH=/mnt/tank/scratch/rzamotaev
MC=$SCRATCH/miniconda3
PIP=$MC/bin/pip
PY=$MC/bin/python

echo "== pip deps (this downloads torch ~2.5GB once) =="
$PIP install -q torch transformers accelerate huggingface_hub vmas gymnasium minigrid pyyaml alfworld

echo "== alfworld data -> scratch =="
export ALFWORLD_DATA=$SCRATCH/alfworld_data
mkdir -p "$ALFWORLD_DATA"
if [ ! -d "$ALFWORLD_DATA/json_2.1.1" ]; then
  $MC/bin/alfworld-download || $PY -m alfworld.scripts.alfworld_download
fi
ls "$ALFWORLD_DATA" | head -5

echo "== official base config =="
CFG=$SCRATCH/alfworld_config.yaml
[ -s "$CFG" ] || curl -sL -o "$CFG" \
  https://raw.githubusercontent.com/alfworld/alfworld/master/configs/base_config.yaml
head -3 "$CFG"

echo "== HF weights -> scratch (Qwen2.5 3B + 7B Instruct) =="
export HF_HOME=$SCRATCH/hf_home
$PY - <<'EOF'
import os
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
    p = snapshot_download(repo)
    print("staged:", repo, "->", p)
EOF

echo "== sanity: imports =="
$PY - <<'EOF'
import torch, transformers, vmas, gymnasium, minigrid, yaml
import alfworld
print("torch", torch.__version__, "cuda_available(login)=", torch.cuda.is_available())
print("transformers", transformers.__version__)
EOF
echo "WAVE2 SETUP DONE"
