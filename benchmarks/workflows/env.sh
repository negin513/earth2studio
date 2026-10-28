# Source from Bash; paths follow this checkout and can be overridden.
_BENCH_WORKFLOWS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BENCH_ROOT=${BENCH_ROOT:-$(cd "$_BENCH_WORKFLOWS/../.." && pwd)}
export PY=${PY:-$BENCH_ROOT/.venv/bin/python}
export PYTHONPATH="$BENCH_ROOT:$BENCH_ROOT/benchmarks/workflows${PYTHONPATH:+:$PYTHONPATH}"
export EARTH2STUDIO_CACHE=${EARTH2STUDIO_CACHE:-$HOME/.cache/earth2studio}
export BENCH_RESULTS=${BENCH_RESULTS:-$BENCH_ROOT/results}
export BENCH_SCRATCH=${BENCH_SCRATCH:-${TMPDIR:-/tmp}/earth2studio-benchmarks}
unset _BENCH_WORKFLOWS
