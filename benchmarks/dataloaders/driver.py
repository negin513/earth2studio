#!/usr/bin/env python3
"""Drive the cold-cache before/after benchmark across both earth2studio venvs.

Runs each (source, version, trial) in a fresh subprocess with a wiped cache
directory. Runs are sequential so trials never compete for network bandwidth.

Usage:
    python driver.py \
        --python-before /path/to/earth2studio-0.14.0/.venv/bin/python \
        --python-after  /path/to/earth2studio-main/.venv/bin/python \
        --vars-dir vars/ --results-dir results/ --cache-dir /tmp/bench_cache \
        [--trials 2] [--sources GFS HRRR ...] [--skip-existing]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# Cheap sources first so failures surface early; heavy fetches last.
# Open-PR rows (UFSObsConv_PR914, JPSS, PC_IFS, PC_IFS_5var, PC_OISST) are
# deliberately excluded: each needs --python-after pointed at its PR-branch
# venv — pass them explicitly via --sources (see README).
DEFAULT_ORDER = [
    "MRMS",
    "NClimGridDaily",
    "ISD",
    "GOES",
    "UFSObsConv",
    "WB2ERA5",
    "GEFS_FX",
    "GFS",
    "NCAR_ERA5",
    "HRRR",
    "ARCO",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-before", required=True)
    ap.add_argument("--python-after", required=True)
    ap.add_argument("--vars-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--sources", nargs="*", default=DEFAULT_ORDER)
    ap.add_argument("--timeout", type=int, default=7200, help="per-run wall limit (s)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    pythons = {"before": args.python_before, "after": args.python_after}

    for trial in range(args.trials):
        for source in args.sources:
            for version, py in pythons.items():
                out = args.results_dir / f"{source}__{version}__trial{trial}.json"
                if args.skip_existing and out.exists():
                    print(f"skip {out.name}")
                    continue
                cmd = [
                    py,
                    str(HERE / "bench_source.py"),
                    "--source", source,
                    "--version", version,
                    "--variables-file", str(args.vars_dir / f"{source}.json"),
                    "--cache-dir", str(args.cache_dir),
                    "--output", str(out),
                ]
                print(f"=== {source} [{version}] trial {trial} ===", flush=True)
                t0 = time.time()
                try:
                    subprocess.run(cmd, timeout=args.timeout, check=False)
                except subprocess.TimeoutExpired:
                    print(f"TIMEOUT after {args.timeout}s", flush=True)
                    out.write_text(json.dumps({
                        "source": source, "version": version, "n_variables": None,
                        "seconds": None, "error": f"driver timeout after {args.timeout}s",
                    }))
                if not out.exists():
                    print(f"    WARN: no result file written for {out.name}", flush=True)
                print(f"    wall {time.time() - t0:.1f}s", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
