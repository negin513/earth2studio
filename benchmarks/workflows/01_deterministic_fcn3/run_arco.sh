#!/usr/bin/env bash
# Benchmark 1b: FCN3 10-day forecast initialized from ARCO ERA5 (Zarr on GCS) instead of
# GFS (GRIB2 on S3), written with AsyncZarrBackend. Same model, same output path, so the
# only difference from the fastest benchmark-1 variant is the input data format and store.
#   bash 01_deterministic_fcn3/run_arco.sh                # cold, fp32, 40 steps
#   CACHE=warm STEPS=2 bash 01_deterministic_fcn3/run_arco.sh   # quick check
set -u
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../env.sh"; cd "$HERE/.."
CACHE=${CACHE:-cold}; STEPS=${STEPS:-40}; DTYPE=${DTYPE:-fp32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)}
LOG=$BENCH_RESULTS/01_deterministic_fcn3; mkdir -p "$LOG"
"${PY1:-$PY}" 01_deterministic_fcn3/bench.py --source arco --io async_zarr --dtype "$DTYPE" --steps "$STEPS" --cache "$CACHE" "$@" 2>&1 \
  | grep -v -i warning | tee "$LOG/arco_async_zarr_${DTYPE}_${CACHE}.log"
"$PY" common/make_table.py >/dev/null
