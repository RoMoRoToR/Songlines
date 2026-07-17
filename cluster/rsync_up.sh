#!/usr/bin/env bash
# Push the project to the KT cluster scratch (run FROM your Mac).
# Usage: bash cluster/rsync_up.sh
# Assumes ~/.ssh/config has Host ctlab (ctlab.itmo.ru) and Host sphinx (ProxyJump ctlab).
set -euo pipefail
DEST="sphinx:/mnt/tank/scratch/rzamotaev/songlines/"
SRC="$(cd "$(dirname "$0")/.." && pwd)/"
rsync -av --delete \
  --exclude 'cluster/logs' --exclude 'cluster/jobs' \
  --exclude '.git' --exclude '.venv' --exclude '.idea' --exclude '__pycache__' \
  --exclude 'wheelhouse*' --exclude 'wheelenv' --exclude 'mnt' \
  --exclude 'tmp/' --exclude '*.pdf' --exclude '*.gif' --exclude '*.pptx' --exclude '*.docx' \
  --exclude 'docs/' --exclude 'paper1_code_bundle' \
  "$SRC" "$DEST"
echo "Pushed to $DEST"
