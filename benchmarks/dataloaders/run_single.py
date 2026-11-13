#!/usr/bin/env python3
"""Run the cold-cache data loader benchmarks against ONE earth2studio install.

No before/after comparison: each (source, trial) is run in a fresh subprocess
with a wiped cache, using whichever earth2studio is installed in --python
(intended: PyPI earth2studio==0.20.0). Prints a per-source summary at the end.

Usage:
    python run_single.py --python /path/to/venv/bin/python \
        --vars-dir vars/ --results-dir results_0.20.0/ \
        --cache-dir /tmp/e2s_bench_cache [--trials 2] [--sources GFS HRRR ...]
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent

# Same order/set as driver.py: cheap first, heavy last. Open-PR rows excluded.
DEFAULT_ORDER = [
    "MRMS", "NClimGridDaily", "ISD", "GOES", "UFSObsConv", "WB2ERA5",
    "GEFS_FX", "GFS", "NCAR_ERA5", "HRRR", "ARCO",
]


def summarize(results_dir: Path, sources: list[str]) -> None:
    print(f"\n{'source':<16}{'#vars':>7}{'median s':>10}{'ctor s':>9}{'call s':>9}{'MB':>9}  note")
    for s in sources:
        runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob(f"{s}__single__trial*.json"))]
        ok = [r for r in runs if r.get("seconds") is not None and not r.get("error")]
        if not ok:
            err = next((r["error"] for r in runs if r.get("error")), "no result")
            print(f"{s:<16}{'-':>7}{'-':>10}{'-':>9}{'-':>9}{'-':>9}  FAILED: {err.strip().splitlines()[-1][:80]}")
            continue
        med = lambda k: statistics.median(r[k] for r in ok)
        mb = statistics.median(r.get("cache_bytes", 0) for r in ok) / 1e6
        print(f"{s:<16}{ok[0]['n_variables']:>7}{med('seconds'):>10.1f}{med('ctor_seconds'):>9.1f}"
              f"{med('call_seconds'):>9.1f}{mb:>9.0f}  {len(ok)}/{len(runs)} ok, v{ok[0].get('earth2studio_version')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--vars-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--sources", nargs="*", default=DEFAULT_ORDER)
    ap.add_argument("--timeout", type=int, default=7200, help="per-run wall limit (s)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for trial in range(args.trials):
        for source in args.sources:
            out = args.results_dir / f"{source}__single__trial{trial}.json"
            if args.skip_existing and out.exists():
                print(f"skip {out.name}")
                continue
            cmd = [
                args.python, str(HERE / "bench_source.py"),
                "--source", source, "--version", "single",
                "--variables-file", str(args.vars_dir / f"{source}.json"),
                "--cache-dir", str(args.cache_dir), "--output", str(out),
            ]
            print(f"=== {source} trial {trial} ===", flush=True)
            t0 = time.time()
            try:
                subprocess.run(cmd, timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired:
                out.write_text(json.dumps({
                    "source": source, "version": "single", "n_variables": None,
                    "seconds": None, "error": f"driver timeout after {args.timeout}s",
                }))
            print(f"    wall {time.time() - t0:.1f}s", flush=True)

    summarize(args.results_dir, args.sources)


if __name__ == "__main__":
    main()
