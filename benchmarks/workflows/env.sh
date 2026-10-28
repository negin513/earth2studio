# Source this before running any workflow benchmark.
# Reuses the earth2studio-main-bench venv (torch 2.13 cu130) and overlays this
# fresh clone on PYTHONPATH. Run scripts from benchmarks/workflows/<name>/, not
# from the clone root (cwd package would shadow nothing here, but keep it clean).
FORKS=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/forks
export BENCH_ROOT=$FORKS/earth2studio-bench
export PYTHONPATH=$BENCH_ROOT:$BENCH_ROOT/benchmarks/workflows${PYTHONPATH:+:$PYTHONPATH}
export PY=$FORKS/earth2studio-main-bench/.venv/bin/python
export EARTH2STUDIO_CACHE=${EARTH2STUDIO_CACHE:-/root/.cache/earth2studio}
export BENCH_RESULTS=$BENCH_ROOT/results
export BENCH_SCRATCH=${BENCH_SCRATCH:-/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/forks/e2s_bench_cache/workflows}
mkdir -p "$BENCH_RESULTS" "$BENCH_SCRATCH"
