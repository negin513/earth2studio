#!/usr/bin/env bash
# One entry point for all six benchmark matrices. See --help.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
if [[ ! -x "$PY" ]]; then
  echo "Python environment missing: $PY" >&2
  echo "Set PY=/absolute/path/to/your/venv/bin/python (see benchmarks/workflows/README.md)." >&2
  exit 2
fi
exec "$PY" "$HERE/common/run_suite.py" "$@"
