# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collate results/<benchmark>/*.json into results/RESULTS.md."""

import argparse
import json
import os
from pathlib import Path

from rich.console import Console

try:
    from .presentation import latest_runs, print_comparison, summary_markdown
except ImportError:  # Direct invocation: python common/make_table.py
    from presentation import latest_runs, print_comparison, summary_markdown

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--results", type=Path, default=Path(os.environ.get("BENCH_RESULTS", "results"))
)
parser.add_argument(
    "--latest-only",
    action="store_true",
    help="Keep the newest run per benchmark/variant",
)
args = parser.parse_args()
root = args.results
root.mkdir(parents=True, exist_ok=True)
all_rows = []
out = []
for bdir in sorted(p for p in root.iterdir() if p.is_dir()):
    files = sorted(bdir.glob("*.json"))
    if not files:
        continue
    rows = [json.loads(f.read_text()) for f in files]
    if args.latest_only:
        rows = latest_runs(rows)
    all_rows.extend(rows)
    out.append(f"\n## {bdir.name}\n")
    env = rows[-1]["env"]
    out.append(
        f"GPU: {env.get('gpu')}  torch {env['torch']}  earth2studio {env.get('earth2studio_sha')}\n"
    )
    phases = sorted({p for r in rows for p in r["phases"]})
    vkeys = sorted({k for r in rows for k in r["variant"]})
    ekeys = sorted(
        {k for r in rows for k in r["extra"] if isinstance(r["extra"][k], (int, float))}
    )
    hdr = vkeys + ["total_s", "peak_GiB"] + [f"{p} (mean s)" for p in phases] + ekeys
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for r in rows:
        cells = [str(r["variant"].get(k, "")) for k in vkeys]
        cells += [str(r["total_wall_s"]), str(r["peak_gpu_mem_gib"])]
        for p in phases:
            ph = r["phases"].get(p)
            cells.append(
                ""
                if ph is None
                else f"{ph.get('mean_steady_s', ph['mean_s']):.3f} (n={ph['count']})"
            )
        cells += [f"{r['extra'][k]:.4g}" if k in r["extra"] else "" for k in ekeys]
        out.append("| " + " | ".join(cells) + " |")
(root / "RESULTS.md").write_text(
    "# Workflow benchmark results\n" + "\n".join(out) + "\n"
)
(root / "SUMMARY.md").write_text(summary_markdown(all_rows))
console = Console(record=True, width=max(160, Console().width))
print_comparison(all_rows, console)
console.save_text(str(root / "SUMMARY.txt"))
print(root / "RESULTS.md")
