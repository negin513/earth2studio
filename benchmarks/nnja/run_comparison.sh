#!/usr/bin/env bash
# End-to-end old-vs-new comparison for the NNJAObsSat IR-decode optimization (#1059).
#
# Sets up two worktrees (pre-#1059 and current), a shared venv, downloads one
# real IASI aggregate file once, and times decode_ir_sounder against it with
# both versions of the code. Meant for a machine with real bandwidth/CPU
# (a login/compute node), not a sandbox.
#
# Usage:
#   ./run_comparison.sh [workdir]
#   WORKER_COUNTS="1 4 16" ./run_comparison.sh [workdir]
#
# By default runs at decode_workers=1 and decode_workers=<nproc>; override
# with the WORKER_COUNTS env var (space-separated) to test other points,
# e.g. matching a GPU node's CPU core count.
#
# Requires: git, uv, curl. ~2 GB disk (repo x2 + venv + BUFR file), several
# GB RAM, and enough CPU/time to fully decode a ~200M-row aggregate at each
# worker count, for both old and new code.

set -euo pipefail

WORKDIR="${1:-$(pwd)/nnja-bench-workdir}"
REPO_URL="https://github.com/NVIDIA/earth2studio.git"
PERF_COMMIT="ab47c427"       # perf(data): vectorize NNJA IR decode (#1059)
PARENT_COMMIT="${PERF_COMMIT}~1"
BUFR_URL="https://noaa-reanalyses-pds.s3.amazonaws.com/observations/reanalysis/iasi/mtiasi/2019/01/bufr/gdas.20190101.t00z.mtiasi.tm00.bufr_d"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "==> Cloning earth2studio (new code, at ${PERF_COMMIT})"
if [ ! -d new ]; then
    git clone "$REPO_URL" new
fi
git -C new fetch origin "$PERF_COMMIT" 2>/dev/null || true
git -C new checkout "$PERF_COMMIT"

echo "==> Adding worktree for old code (at ${PARENT_COMMIT})"
if [ ! -d old ]; then
    git -C new worktree add "$WORKDIR/old" "$PARENT_COMMIT"
fi

echo "==> Setting up shared venv (installed against new code; same deps as old)"
if [ ! -d .venv ]; then
    uv venv --python 3.11 .venv
fi
source .venv/bin/activate
uv pip install -e ./new[data]

echo "==> Downloading benchmark file (${BUFR_URL})"
if [ ! -f iasi.bufr_d ]; then
    curl -sS -o iasi.bufr_d "$BUFR_URL"
fi
ls -la iasi.bufr_d

# Worker counts to benchmark. decode_workers=1 isolates the per-message
# vectorization change; decode_workers>1 also exercises the #1059 batch-level
# change (Arrow tables crossing the process boundary instead of pickled
# per-row dicts), which is where a many-core node pays off most.
NPROC="$(python3 -c 'import os; print(os.cpu_count())')"
WORKER_COUNTS="${WORKER_COUNTS:-1 ${NPROC}}"

> comparison.log
for W in $WORKER_COUNTS; do
    echo "==> Running NEW-code decode benchmark (decode_workers=${W})"
    ( cd new && python "$WORKDIR/new/benchmarks/nnja/bench_nnja_decode.py" "$WORKDIR/iasi.bufr_d" "$W" ) | tee "new_result_w${W}.log"

    echo "==> Running OLD-code decode benchmark (decode_workers=${W})"
    ( cd old && python "$WORKDIR/new/benchmarks/nnja/bench_nnja_decode.py" "$WORKDIR/iasi.bufr_d" "$W" ) | tee "old_result_w${W}.log"

    OLD_S=$(grep -oE 'seconds=[0-9.]+' "old_result_w${W}.log" | cut -d= -f2)
    NEW_S=$(grep -oE 'seconds=[0-9.]+' "new_result_w${W}.log" | cut -d= -f2)
    echo "decode_workers=${W} old=${OLD_S}s new=${NEW_S}s" >> comparison.log
done

echo ""
echo "==> Comparison (decode_workers: ${WORKER_COUNTS})"
python3 -c "
import sys
header = ('workers', 'old(s)', 'new(s)', 'speedup')
print(f'{header[0]:>8} {header[1]:>10} {header[2]:>10} {header[3]:>10}')
with open('comparison.log') as f:
    for line in f:
        parts = dict(p.split('=') for p in line.split())
        w, old, new = parts['decode_workers'], float(parts['old'].rstrip('s')), float(parts['new'].rstrip('s'))
        print(f'{w:>8} {old:>10.2f} {new:>10.2f} {old / new:>9.2f}x')
"
