#!/usr/bin/env bash
# Run every workflow benchmark and save all timing outputs.
#
#   bash run_all_benchmarks.sh                 # sequential, GPU 0, cold cache
#   bash run_all_benchmarks.sh --parallel      # one benchmark per GPU (round-robin over visible GPUs)
#   bash run_all_benchmarks.sh --only 01,04    # subset
#   bash run_all_benchmarks.sh --warm          # also run each benchmark's warm-cache pass
#
# Outputs (all under $BENCH_RESULTS, default <repo>/results):
#   <benchmark>/*.json         one JSON per run (phase timings, peak memory, extra metrics)
#   logs/<benchmark>.log       full stdout/stderr of that benchmark's run_all.sh
#   run_manifest.json          start/end/exit code per benchmark for this invocation
#   RESULTS.md, ANALYSIS.md    tables produced by common/make_table.py and common/analyze.py
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
source env.sh

PARALLEL=0; WARM=0; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel) PARALLEL=1 ;;
    --warm) WARM=1 ;;
    --only) ONLY="$2"; shift ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done

ALL=(01_deterministic_fcn3 02_ensemble_hens_lite 03_corrdiff_taiwan 04_stormscope_goes 05_s2s_long_rollout 06_stormscope_eu)
BENCHES=()
for b in "${ALL[@]}"; do
  [[ -f "$b/run_all.sh" ]] || { echo "skip $b (no run_all.sh)"; continue; }
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",${b%%_*},"* ]] || continue; fi
  BENCHES+=("$b")
done

mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
NGPU=${#GPUS[@]}
LOGDIR=$BENCH_RESULTS/logs; mkdir -p "$LOGDIR"
MANIFEST=$BENCH_RESULTS/run_manifest.json
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)

# Per-benchmark warm-pass convention differs; pass through what each script understands.
warm_args() {
  case "$1" in
    02_*) echo "warm" ;;                       # positional MODE
    03_*) echo "--warm" ;;
    *) echo "" ;;                              # 04/06 use env WARM/CACHE, handled below
  esac
}

run_one() {  # bench gpu
  local b=$1 gpu=$2 t0 t1 rc
  t0=$(date -u +%FT%TZ)
  echo "### $t0 start $b on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu bash "$b/run_all.sh" > "$LOGDIR/$b.log" 2>&1; rc=$?
  if [[ $WARM == 1 && $rc == 0 ]]; then
    case "$b" in
      04_*) CUDA_VISIBLE_DEVICES=$gpu CACHE=warm bash "$b/run_all.sh" >> "$LOGDIR/$b.log" 2>&1 || rc=$? ;;
      06_*) CUDA_VISIBLE_DEVICES=$gpu WARM=1 bash "$b/run_all.sh" >> "$LOGDIR/$b.log" 2>&1 || rc=$? ;;
      *) wa=$(warm_args "$b"); [[ -n "$wa" ]] && { CUDA_VISIBLE_DEVICES=$gpu bash "$b/run_all.sh" $wa >> "$LOGDIR/$b.log" 2>&1 || rc=$?; } ;;
    esac
  fi
  t1=$(date -u +%FT%TZ)
  echo "### $t1 done  $b rc=$rc"
  printf '{"run_id":"%s","benchmark":"%s","gpu":%s,"start":"%s","end":"%s","exit_code":%d}\n' \
    "$RUN_ID" "$b" "$gpu" "$t0" "$t1" "$rc" >> "$MANIFEST.jsonl"
}

if [[ $PARALLEL == 1 ]]; then
  i=0
  for b in "${BENCHES[@]}"; do
    run_one "$b" "${GPUS[$((i % NGPU))]}" &
    i=$((i+1))
  done
  wait
else
  for b in "${BENCHES[@]}"; do run_one "$b" "${CUDA_VISIBLE_DEVICES:-${GPUS[0]}}"; done
fi

# consolidate manifest lines into a JSON array
$PY - "$MANIFEST" <<'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); lines = (p.with_suffix(".json.jsonl")).read_text().splitlines()
p.write_text(json.dumps([json.loads(l) for l in lines if l.strip()], indent=2))
PYEOF

$PY common/make_table.py
$PY common/analyze.py
echo "### ALL_DONE  results in $BENCH_RESULTS"
