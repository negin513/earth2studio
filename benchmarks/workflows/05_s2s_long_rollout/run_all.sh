#!/bin/bash
# Benchmark 5: DLESyM S2S 45-day rollout, members 1 and 4, cold cache (default).
# Usage: bash 05_s2s_long_rollout/run_all.sh [--cache warm]   (from benchmarks/workflows/)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
cd "$HERE/.."
# DLESyM needs earth2grid and a newer physicsnemo than the shared venv has.
# Both live in an overlay dir (never installed into the venv), see README.
export PYTHONPATH=$BENCH_SCRATCH/pyoverlay/05:$PYTHONPATH
# Pick the GPU with the least memory in use (other benchmarks own GPUs 0..3).
pick_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
EXTRA="$@"
for M in 1 4; do
  export CUDA_VISIBLE_DEVICES=$(pick_gpu)
  echo "== members=$M on GPU $CUDA_VISIBLE_DEVICES ($EXTRA)"
  $PY 05_s2s_long_rollout/bench.py --days 45 --members $M --weekly-stats $EXTRA 2>&1 | grep -v -i 'warning\|pynvml\|Downloading'
done
$PY common/make_table.py
