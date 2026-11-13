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

"""Render results/results.json as a markdown table."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "results.json")
    args = ap.parse_args()

    groups = defaultdict(list)
    for r in json.loads(args.results.read_text()):
        groups[r["source"]].append(r)

    def mb(runs: list, key: str) -> str:
        """Median of a byte-count field across runs, in MB."""
        vals = [r[key] for r in runs if r.get(key) is not None]
        return f"{statistics.median(vals) / 1e6:.1f}" if vals else "-"

    def spec_field(runs: list, src: str, key: str) -> str:
        """Field from the results, falling back to the spec file."""
        val = runs[0].get(key)
        spec = HERE / "specs" / f"{src}.json"
        if val is None and spec.exists():
            val = json.loads(spec.read_text()).get(key)
        return val or "-"

    def shape(runs: list) -> str:
        """Returned array shape, labelled with dimension names when recorded."""
        for r in runs:
            if r.get("result_shape"):
                dims = r.get("result_dims")
                out = " x ".join(str(n) for n in r["result_shape"])
                return f"{out} ({', '.join(dims)})" if dims else out
        return "-"

    def zarr_version(runs: list, src: str) -> str:
        """Zarr format version: detected at run time, else from the spec."""
        for r in runs:
            if r.get("zarr_format"):
                return f"v{r['zarr_format']}"
        val = spec_field(runs, src, "zarr_format")
        return f"v{val}" if val != "-" else "-"

    print(
        "| Source | Model | earth2studio version | Format | Zarr version "
        "| Pulled from | Variables requested | Variables returned | Shape "
        "| Median time (s) | Cache (MB) | Data in memory (MB) | Status |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for src in sorted(groups):
        runs = groups[src]
        fmt = spec_field(runs, src, "format")
        origin = spec_field(runs, src, "origin")
        ok = [r for r in runs if not r.get("error") and r.get("seconds") is not None]
        errs = [r for r in runs if r.get("error")]
        model = runs[0].get("winning_model") or "-"
        req = runs[0].get("n_variables_requested")
        ret = ok[0].get("n_variables_returned") if ok else None
        med = f"{statistics.median(r['seconds'] for r in ok):.1f}" if ok else "-"
        if errs:
            last = errs[-1]["error"].strip().splitlines()[-1][:60].replace("|", "/")
            status = f"{len(ok)}/{len(runs)} ok; {last}"
        else:
            status = f"ok ({len(ok)}/{len(runs)})"
        empty = max((len(r.get("variables_all_nan") or []) for r in ok), default=0)
        if empty:
            status += f"; {empty} variables all NaN"
        print(
            f"| {src} | {model} | {runs[0].get('earth2studio_version') or '-'} "
            f"| {fmt} | {zarr_version(runs, src)} | {origin} | {req} "
            f"| {ret if ret is not None else '-'} | {shape(ok)} | {med} "
            f"| {mb(ok, 'cache_bytes')} | {mb(ok, 'result_bytes')} | {status} |"
        )


if __name__ == "__main__":
    main()
