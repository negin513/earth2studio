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


SUMMARY_COLUMNS = (
    "Workflow",
    "Variant",
    "Model download s",
    "Model load s",
    "Input fetch s",
    "Inference total s",
    "Inference s/step",
    "Total wall s",
    "Peak GPU GiB",
)
WORKFLOWS = {
    "01_deterministic_fcn3": "Weather: FCN3 (6 h)",
    "02_ensemble_hens_lite": "Ensemble: SFNO (6 h)",
    "03_corrdiff_taiwan": "Downscaling: CorrDiff (forcing time)",
    "04_stormscope_goes": "Nowcasting: GOES (10 min)",
    "05_s2s_long_rollout": "S2S: DLESyM (96 h)",
    "06_stormscope_eu": "Nowcasting: EU (10 min)",
}
SUMMARY_NOTE = (
    "Seconds per step use steady inference timing (first call excluded when available). "
    "Ensembles are per batch step; CorrDiff is per forcing time for all requested samples, "
    "not per diffusion iteration. Input fetch includes download/decode/preparation. "
    "Model load includes DLESyM's lat/lon wrapper. Total wall includes IO, compilation "
    "and other overhead; phase columns need not sum to it. Peak GPU is allocated memory."
)


def comparison_rows(runs: Sequence[dict[str, Any]]) -> list[list[str]]:
    """Extract the same phase totals and step metrics for terminal and Markdown."""
    rows = []
    for run in sorted(
        runs,
        key=lambda r: (
            r["benchmark"],
            variant_label(r["variant"]),
            r.get("started_utc", ""),
        ),
    ):
        phases = run.get("phases", {})
        inference = phases.get("inference_step", phases.get("downscale_step", {}))
        extra = run.get("extra", {})
        per_step = extra.get(
            "step_seconds_steady", extra.get("inference_s_per_step_steady")
        )
        if per_step is None:
            per_step = inference.get("mean_steady_s", inference.get("mean_s"))
        loads = [
            phase["total_s"]
            for name, phase in phases.items()
            if name.startswith("model_load")
        ]
        rows.append(
            [
                WORKFLOWS.get(run["benchmark"], run["benchmark"]),
                variant_label(run["variant"]),
                format_value(phases.get("model_download", {}).get("total_s")),
                format_value(sum(loads) if loads else None),
                format_value(phases.get("ic_fetch", {}).get("total_s")),
                format_value(inference.get("total_s")),
                format_value(per_step),
                format_value(run.get("total_wall_s")),
                format_value(run.get("peak_gpu_mem_gib")),
            ]
        )
    return rows


def summary_markdown(runs: Sequence[dict[str, Any]]) -> str:
    """Export the combined table with the same values as the Rich output."""

    def escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Workflow benchmark summary",
        "",
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "|" + "---|" * len(SUMMARY_COLUMNS),
    ]
    lines.extend(
        "| " + " | ".join(escape(cell) for cell in row) + " |"
        for row in comparison_rows(runs)
    )
    return "\n".join(lines) + "\n\n" + SUMMARY_NOTE + "\n"


def print_comparison(
    runs: Sequence[dict[str, Any]], console: Console | None = None
) -> None:
    """Show all workflows in one table of phase totals, step time and GPU memory."""
    console = console or Console()
    if not runs:
        console.print("No benchmark runs found.")
        return
    table = Table(
        title="Workflow benchmark summary",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
    )
    for index, label in enumerate(SUMMARY_COLUMNS):
        table.add_column(
            label,
            justify="left" if index < 2 else "right",
            overflow="fold",
            min_width=7 if index >= 2 else 12,
        )
    for row in comparison_rows(runs):
        table.add_row(*row)
    console.print(table)
    console.print(SUMMARY_NOTE, markup=False)
