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
#
# Requires: git, uv, curl. ~2 GB disk (repo x2 + venv + BUFR file), several
# GB RAM, and enough CPU/time to fully decode a ~200M-row aggregate twice.

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

echo "==> Running NEW-code decode benchmark"
( cd new && python "$WORKDIR/new/benchmarks/nnja/bench_nnja_decode.py" "$WORKDIR/iasi.bufr_d" 1 ) | tee new_result.log

echo "==> Running OLD-code decode benchmark"
( cd old && python "$WORKDIR/new/benchmarks/nnja/bench_nnja_decode.py" "$WORKDIR/iasi.bufr_d" 1 ) | tee old_result.log

echo ""
echo "==> Comparison"
OLD_S=$(grep -oE 'seconds=[0-9.]+' old_result.log | cut -d= -f2)
NEW_S=$(grep -oE 'seconds=[0-9.]+' new_result.log | cut -d= -f2)
python3 -c "
old, new = ${OLD_S}, ${NEW_S}
print(f'old (pre-#1059):  {old:.2f}s')
print(f'new (post-#1059): {new:.2f}s')
print(f'speedup: {old / new:.2f}x')
"
