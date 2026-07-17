#!/usr/bin/env bash
# Separate conda env with torch cu124 (cluster driver = CUDA 12.4; base env's
# torch cu130 is too new). Isolated so running CPU jobs on base are untouched.
set -euo pipefail
MC=/mnt/tank/scratch/rzamotaev/miniconda3
ENVP=$MC/envs/gpu

[ -x "$ENVP/bin/python" ] || $MC/bin/conda create -y -n gpu python=3.11
$ENVP/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu124
$ENVP/bin/pip install -q transformers accelerate huggingface_hub pyyaml alfworld numpy
$ENVP/bin/python - <<'EOF'
import torch, transformers
print("gpu-env torch", torch.__version__, "| transformers", transformers.__version__)
from alfworld.agents.environment import get_environment
print("alfworld OK:", get_environment("AlfredTWEnv").__name__)
EOF
echo "GPU ENV DONE ($ENVP/bin/python)"
