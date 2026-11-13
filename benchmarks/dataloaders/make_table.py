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

    print("| Source | Model | #vars (req/ret) | Median time (s) | Status |")
    print("|---|---|---|---|---|")
    for src in sorted(groups):
        runs = groups[src]
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
        print(
            f"| {src} | {model} | {req}/{ret if ret is not None else '-'} | {med} | {status} |"
        )


if __name__ == "__main__":
    main()
