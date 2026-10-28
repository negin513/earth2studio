#!/bin/bash
# Runs the 4-variant matrix for benchmark 1, cold cache (default). Each variant
# gets its own fresh EARTH2STUDIO_CACHE dir (see common.harness.setup_cache), so
# model weights and the GFS IC are downloaded from scratch every time.
#   ./run_all.sh                 # cold matrix, 40 steps
#   ./run_all.sh --cache warm    # warm pass (reuses ~/.cache/earth2studio + local IC)
#   ./run_all.sh --steps 2       # quick smoke
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$HERE/.."
for io in zarr async_zarr; do
  for dt in fp32 bf16; do
    echo "### io=$io dtype=$dt $*"
    $PY 01_deterministic_fcn3/bench.py --io $io --dtype $dt "$@" 2>&1 | grep -v -i warning
  done
done
$PY common/make_table.py
