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
"""Rich terminal views for saved workflow results; no CUDA imports required."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

HEADLINE = {
    "01_deterministic_fcn3": ("steps_per_s", "steps/s"),
    "02_ensemble_hens_lite": ("member_steps_per_s", "member-steps/s"),
    "03_corrdiff_taiwan": ("seconds_per_sample", "s/sample"),
    "04_stormscope_goes": ("step_seconds_steady", "s/step"),
    "05_s2s_long_rollout": ("member_days_per_gpu_minute", "member-days/GPU-min"),
    "06_stormscope_eu": ("step_seconds_steady", "s/step"),
}


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def variant_label(variant: Mapping[str, Any]) -> str:
    return (
        ", ".join(f"{key}={value}" for key, value in sorted(variant.items()))
        or "default"
    )


def latest_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest timestamp per complete benchmark/variant combination."""
    keep: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        key = (run["benchmark"], json.dumps(run["variant"], sort_keys=True))
        if key not in keep or run["started_utc"] > keep[key]["started_utc"]:
            keep[key] = run
    return list(keep.values())


def print_run(run: Mapping[str, Any], console: Console | None = None) -> None:
    """Show one run's phase samples and extra metrics without truncating keys."""
    console = console or Console()
    console.rule(run["benchmark"], style="cyan")
    console.print(variant_label(run["variant"]), markup=False)
    console.print(
        f"Wall: {format_value(run['total_wall_s'])} s   "
        f"Peak GPU allocation: {format_value(run['peak_gpu_mem_gib'])} GiB",
        markup=False,
    )
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan")
    for name in ("Phase", "Count", "Total s", "Mean s", "Steady s"):
        table.add_column(name, justify="left" if name == "Phase" else "right")
    for name, phase in run["phases"].items():
        table.add_row(
            name,
            *(
                format_value(phase.get(key))
                for key in ("count", "total_s", "mean_s", "mean_steady_s")
            ),
        )
    console.print(table)
    if run.get("extra"):
        metrics = Table(box=box.SIMPLE_HEAD, header_style="bold cyan")
        metrics.add_column("Metric", style="cyan")
        metrics.add_column("Value", overflow="fold")
        for key, value in run["extra"].items():
            metrics.add_row(key, format_value(value))
        console.print(metrics)


def print_comparison(
    runs: Sequence[dict[str, Any]], console: Console | None = None
) -> None:
    """Compare variants per benchmark using a compact set of headline columns."""
    console = console or Console()
    if not runs:
        console.print("No benchmark runs found.")
        return
    for benchmark in sorted({run["benchmark"] for run in runs}):
        group = [run for run in runs if run["benchmark"] == benchmark]
        metric_key, metric_label = HEADLINE.get(benchmark, ("", "Metric"))
        console.rule(benchmark, style="cyan")
        table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan")
        table.add_column("Variant", overflow="fold", ratio=3)
        table.add_column("Started UTC", overflow="fold")
        for label in ("Wall s", "Peak GiB", metric_label):
            table.add_column(label, justify="right")
        for run in group:
            table.add_row(
                variant_label(run["variant"]),
                run.get("started_utc", "—"),
                format_value(run.get("total_wall_s")),
                format_value(run.get("peak_gpu_mem_gib")),
                format_value(run.get("extra", {}).get(metric_key)),
            )
        console.print(table)
