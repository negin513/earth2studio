#!/bin/bash
# Full matrix for benchmark 2, cold cache (each variant downloads models + GFS IC
# into its own fresh cache dir). Pass "warm" as $1 for a warm-cache pass.
# Combinations that OOM/fail are recorded in results/02_ensemble_hens_lite/failures.log.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
MODE=${1:-cold}
# 5 GB SFNO checkpoint from NGC exceeds the 300 s default fsspec timeout on a busy link
export EARTH2STUDIO_PACKAGE_TIMEOUT=${EARTH2STUDIO_PACKAGE_TIMEOUT:-7200}
STEPS=${STEPS:-20}
LOG=$BENCH_RESULTS/02_ensemble_hens_lite; mkdir -p "$LOG"
cd "$HERE/.."
for pert in spherical_gaussian bred_vector; do
  for members in 16 32; do
    for bs in 8 16; do
      tag="${pert}_m${members}_b${bs}_${MODE}"
      echo "=== $tag  $(date -u +%FT%TZ)"
      if ! $PY 02_ensemble_hens_lite/bench.py --perturbation $pert --members $members \
            --batch-size $bs --steps $STEPS --cache $MODE > "$LOG/$tag.log" 2>&1; then
        echo "FAILED $tag" | tee -a "$LOG/failures.log"
        grep -m1 -E 'OutOfMemoryError|Error' "$LOG/$tag.log" | tee -a "$LOG/failures.log"
      fi
    done
  done
done
$PY common/make_table.py
