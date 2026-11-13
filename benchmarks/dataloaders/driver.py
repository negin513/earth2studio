#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the cold-cache dataloader benchmark for every spec, sequentially.

Each (source, trial) runs in a fresh subprocess with a wiped cache dir.

Usage:
    python driver.py [--sources GFS HRRR] [--trials 3] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

# Default set: the sources that models in this repository read from, plus a few
# widely used ones. Run with --preset all for every enabled spec.
TOP20 = [
    "GFS", "GFS_FX", "HRRR", "GEFS_FX", "CFS_FX", "MRMS", "GOES", "GOESGLMGrid",
    "ARCO", "WB2ERA5", "PC_IFS", "ACE2", "IFS", "NCAR_ERA5", "UFSObsConv",
    "UFSObsSat", "NNJAObsConv", "NNJAObsSatwnd", "GHCNHourly", "ISD",
]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs-dir", type=Path, default=HERE / "specs")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--preset", choices=["top20", "all"], default="top20")
    ap.add_argument("--sources", nargs="*", default=None, help="explicit source names")
    ap.add_argument("--timeout", type=int, default=900, help="per-run limit (s)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs = sorted(args.specs_dir.glob("*.json"))
    if args.sources:
        # naming a source explicitly overrides its "enabled": false flag
        specs = [s for s in specs if s.stem in set(args.sources)]
    elif args.preset == "top20":
        specs = [s for s in specs if s.stem in set(TOP20)]
    else:
        kept = []
        for s in specs:
            meta = json.loads(s.read_text())
            if meta.get("enabled", True):
                kept.append(s)
            else:
                print(f"skip {s.stem}: {meta.get('disabled_reason', 'disabled')}")
        specs = kept
    if not specs:
        sys.exit("no specs selected")
    cache = args.cache_dir or Path(tempfile.gettempdir()) / "e2s_dl_bench_cache"
    raw = args.results_dir / "raw"
    trials = 1 if args.dry_run else args.trials

    failures = 0
    for spec in specs:
        for trial in range(trials):
            out = raw / f"{spec.stem}__trial{trial}.json"
            cmd = [
                args.python,
                str(HERE / "bench_source.py"),
                "--spec", str(spec),
                "--cache-dir", str(cache),
                "--output", str(out),
            ]  # fmt: skip
            if args.dry_run:
                cmd.append("--dry-run")
            print(f"=== {spec.stem} trial {trial} ===", flush=True)
            try:
                rc = subprocess.run(cmd, timeout=args.timeout).returncode  # noqa: S603
            except subprocess.TimeoutExpired:
                rc = -1
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(
                        {"source": spec.stem, "error": "TIMEOUT", "seconds": None}
                    )
                )
            failures += rc != 0

    if args.dry_run:
        print(f"dry run done, {failures} failures")
        sys.exit(1 if failures else 0)

    specs_by_name = {s.stem: json.loads(s.read_text()) for s in specs}
    runs = []
    for stem, sp in specs_by_name.items():
        # only this run's trials, so stale files from earlier runs are ignored
        for trial in range(trials):
            r = json.loads((raw / f"{stem}__trial{trial}.json").read_text())
            r["source"] = stem
            for key in ("winning_model", "format", "origin"):
                r[key] = sp.get(key)
            runs.append(r)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results = args.results_dir / "results.json"
    results.write_text(json.dumps(runs, indent=2))
    print(f"wrote {results} ({len(runs)} runs)")

    table = subprocess.run(  # noqa: S603
        [args.python, str(HERE / "make_table.py"), "--results", str(results)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (args.results_dir / "table.md").write_text(table)
    print(f"wrote {args.results_dir / 'table.md'}\n\n{table}")


if __name__ == "__main__":
    main()
