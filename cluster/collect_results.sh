#!/usr/bin/env bash
# Pull cluster results back to the Mac (run FROM your Mac).
# Usage: bash cluster/collect_results.sh
set -euo pipefail
SRC="sphinx:/mnt/tank/scratch/rzamotaev/songlines/tmp/cluster/"
DST="$(cd "$(dirname "$0")/.." && pwd)/tmp/cluster_results/"
mkdir -p "$DST"
rsync -av "$SRC" "$DST"
echo "Results in $DST"
rsync -av "sphinx:/mnt/tank/scratch/rzamotaev/songlines/cluster/logs/" "$DST/logs/" || true
