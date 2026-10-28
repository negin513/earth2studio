#!/bin/bash
# Benchmark 6 matrix, all cold cache. Run from benchmarks/workflows after `source env.sh`.
# GPU: bench.py picks the least-loaded GPU unless CUDA_VISIBLE_DEVICES is set.
set -u
cd "$(dirname "$0")/.."
source env.sh
STEPS=${STEPS:-12}
START=${START_TIME:+--start-time $START_TIME}
LOG=$BENCH_RESULTS/06_stormscope_eu; mkdir -p "$LOG"
run() { echo "=== $*"; $PY 06_stormscope_eu/bench.py --steps "$STEPS" $START "$@" 2>&1 | tee -a "$LOG/run_all.log" | grep -v -i warning; }
run --io zarr
run --io icechunk --commit-mode per_step
run --io icechunk --commit-mode per_init
run --io icechunk --commit-mode per_init --compile
[ "${WARM:-0}" = 1 ] && run --io icechunk --commit-mode per_init --cache warm
$PY common/make_table.py
