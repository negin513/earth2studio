#!/bin/bash
# Full 3x3 sweep (samples x diffusion steps) at 4 forcing times on GPU 2, COLD cache:
# every sweep point downloads the CorrDiff package and the GFS forcing from scratch.
# Pass --warm as first arg to also run a warm-cache pass afterwards.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
cd "$HERE/.."
WARM=0; if [[ "${1:-}" == "--warm" ]]; then WARM=1; shift; fi
$PY 03_corrdiff_taiwan/bench.py --cache cold --times 4 --samples 1,4,8 --diffusion-steps 8,18,36 "$@"
if [[ $WARM == 1 ]]; then
  $PY 03_corrdiff_taiwan/bench.py --cache warm --times 4 --samples 1,4,8 --diffusion-steps 8,18,36 "$@"
fi
$PY common/make_table.py
