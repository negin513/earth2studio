#!/bin/bash
# Benchmark 4 matrix, COLD cache by default (each variant downloads model weights
# and the six GOES frames into its own fresh EARTH2STUDIO_CACHE dir).
# Uses the forks/earth2studio venv (NATTEN 0.21.7 + icechunk 2.1.0, torch 2.12.1)
# instead of $PY from env.sh: the shared main-bench venv has no natten.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
PY4=${PY4:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/forks/earth2studio/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
STEPS=${STEPS:-12}
CACHE=${CACHE:-cold}
START=${START:-}   # e.g. START=2026-10-28T16:50 pins the init across variants (recommended)
ST=${START:+--start-time $START}
cd "$HERE/.."
LOG=$BENCH_RESULTS/04_stormscope_goes/logs; mkdir -p "$LOG"
run() { echo "### $(date -u +%T) $*"; $PY4 04_stormscope_goes/bench.py --steps "$STEPS" --cache "$CACHE" $ST "$@" 2>&1 | grep -v 'Warning\|DEBUG\|Downloading' | tee "$LOG/${CACHE}_$(echo "$*" | tr ' /-' '___').log" | grep 'bench4\]\|extra\.\|total wall\|Error\|error'; }
run --io zarr
run --io icechunk --commit-mode per_step
run --io icechunk --commit-mode per_init
run --io icechunk --commit-mode per_init --compile
if [ "${BASELINE:-0}" = 1 ]; then run --io icechunk --commit-mode per_init --variant 3km_10min; fi
$PY common/make_table.py
echo "### $(date -u +%T) ALL_DONE"
