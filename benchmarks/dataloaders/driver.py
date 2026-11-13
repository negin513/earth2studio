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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs-dir", type=Path, default=HERE / "specs")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--sources", nargs="*", default=None, help="filter by source name")
    ap.add_argument("--timeout", type=int, default=7200, help="per-run limit (s)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs = sorted(args.specs_dir.glob("*.json"))
    if args.sources:
        # naming a source explicitly overrides its "enabled": false flag
        specs = [s for s in specs if s.stem in set(args.sources)]
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

    runs = [json.loads(p.read_text()) for p in sorted(raw.glob("*__trial*.json"))]
    specs_by_name = {s.stem: json.loads(s.read_text()) for s in specs}
    for r in runs:
        sp = specs_by_name.get(r.get("source"), {})
        for key in ("winning_model", "format", "origin"):
            r[key] = sp.get(key)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "results.json").write_text(json.dumps(runs, indent=2))
    print(f"wrote {args.results_dir / 'results.json'} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
