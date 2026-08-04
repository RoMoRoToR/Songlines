#!/usr/bin/env bash
# Stage 8 --- reproduce the song-grammar CORE from a clean checkout of
# the songlines/ package (fast, CPU-only, minutes). Verifies the
# package reproduces the paper's headline verdicts, then aggregates
# every artifact into one long-format CSV.
#
# Heavier campaigns (U7 e1000, social S1-S3, integration I1 x17,
# continuous C1, unified benchmark) run on the cluster via
# cluster/submit_song_grammar.sh + submit_song_grammar drivers; this
# script covers what runs locally in one sitting.
set -euo pipefail
PY=${PY:-.venv/bin/python}
export PYTHONPATH=.

echo "== invariants =="
$PY -m songlines.tests.test_invariants

echo "== S0 song anatomy =="
$PY experiments/song_grammar/exp_s0_song_smoke.py | tail -5
echo "== U1 two-axis matrix =="
$PY experiments/song_grammar/exp_u1_ucsm_smoke.py | tail -5
echo "== W10 vocabulary ablation =="
$PY experiments/warp/exp_warp_landmark_ablation.py | tail -5
echo "== R1 single-run integration =="
$PY experiments/song_grammar/exp_r1_runtime_checklist.py | tail -3
echo "== E1 landmark emergence =="
$PY experiments/song_grammar/exp_e1_landmark_emergence.py | tail -3
echo "== N1v2 fail-closed under noise (consensus) =="
$PY experiments/song_grammar/exp_n1_noise.py --seeds 12 --consensus 2 \
    --out tmp/song_grammar/repro_n1 | tail -3

echo "== aggregate all artifacts -> long CSV =="
$PY scripts/aggregate_song_grammar.py \
    --roots tmp/song_grammar tmp/cluster/song_grammar \
    --out artifacts/song_grammar_long.csv

echo "CORE REPRODUCTION DONE"
