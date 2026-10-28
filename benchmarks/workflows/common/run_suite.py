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
"""Run workflow matrices, queue one benchmark per GPU, and collate reports."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

HERE = Path(__file__).resolve().parents[1]
BENCHMARKS = (
    "01_deterministic_fcn3",
    "02_ensemble_hens_lite",
    "03_corrdiff_taiwan",
    "04_stormscope_goes",
    "05_s2s_long_rollout",
    "06_stormscope_eu",
)


def variants(benchmark: str, smoke: bool) -> list[list[str]]:
    """Full matrices; smoke keeps one short configuration per benchmark."""
    number = benchmark[:2]
    if smoke:
        return {
            "01": [["--steps", "2"]],
            "02": [["--members", "2", "--batch-size", "2", "--steps", "2"]],
            "03": [["--times", "1", "--samples", "1", "--diffusion-steps", "8"]],
            "04": [["--steps", "2"]],
            "05": [["--steps", "1", "--members", "1", "--weekly-stats"]],
            "06": [["--steps", "2", "--io", "zarr"]],
        }[number]
    if number == "01":
        return [
            ["--io", io, "--dtype", dtype]
            for io, dtype in itertools.product(("zarr", "async_zarr"), ("fp32", "bf16"))
        ]
    if number == "02":
        return [
            [
                "--perturbation",
                perturb,
                "--members",
                str(members),
                "--batch-size",
                str(batch),
            ]
            for perturb, members, batch in itertools.product(
                ("spherical_gaussian", "bred_vector"), (16, 32), (8, 16)
            )
        ]
    if number == "03":
        return [["--times", "4", "--samples", "1,4,8", "--diffusion-steps", "8,18,36"]]
    if number == "05":
        return [["--days", "45", "--members", str(m), "--weekly-stats"] for m in (1, 4)]
    matrix = [
        ["--io", "zarr"],
        ["--io", "icechunk", "--commit-mode", "per_step"],
        ["--io", "icechunk", "--commit-mode", "per_init"],
        ["--io", "icechunk", "--commit-mode", "per_init", "--compile"],
    ]
    if number == "04" and os.environ.get("BASELINE") == "1":
        matrix.append(
            ["--io", "icechunk", "--commit-mode", "per_init", "--variant", "3km_10min"]
        )
    return matrix


def gpu_ids(explicit: str | None, dry_run: bool) -> list[str]:
    visible = (
        explicit if explicit is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if visible is None and dry_run:
        visible = "0"
    if visible is None:
        try:
            executable = shutil.which("nvidia-smi")
            if executable is None:
                raise FileNotFoundError("nvidia-smi")
            visible = subprocess.check_output(  # noqa: S603 - resolved nvidia-smi, fixed arguments
                [executable, "--query-gpu=index", "--format=csv,noheader"], text=True
            ).replace(
                "\n", ","
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                "No NVIDIA GPUs detected; use --dry-run to preview commands."
            ) from exc
    ids = [value.strip() for value in visible.split(",") if value.strip()]
    if (
        not ids
        or any(not value.isdigit() for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(
            "Select at least one distinct GPU with --gpus 0,1 or CUDA_VISIBLE_DEVICES."
        )
    return ids


def commands(benchmark: str, args: argparse.Namespace) -> list[list[str]]:
    interpreter = os.environ.get(f"PY{int(benchmark[:2])}", sys.executable)
    modes = ("cold", "warm") if args.cache == "both" else (args.cache,)
    extra: list[str] = []
    if benchmark.startswith("04"):
        extra = ["--start-time", args.goes_time]
    elif benchmark.startswith("06"):
        extra = ["--start-time", args.eu_time]
    return [
        [
            interpreter,
            str(HERE / benchmark / "bench.py"),
            *variant,
            *extra,
            "--cache",
            mode,
        ]
        for variant in variants(benchmark, args.smoke)
        for mode in modes
    ]


def run_benchmark(
    benchmark: str, gpu: str, args: argparse.Namespace, console: Console
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        BENCH_RESULTS=str(args.results),
        BENCH_SCRATCH=str(args.scratch),
    )
    env.setdefault("EARTH2STUDIO_PACKAGE_TIMEOUT", "7200")
    overlay = os.environ.get(f"BENCH_OVERLAY_{benchmark[:2]}")
    if overlay:
        env["PYTHONPATH"] = overlay + os.pathsep + env.get("PYTHONPATH", "")
    log = args.results / "logs" / f"{benchmark}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    records = []
    console.print(f"Starting {benchmark} on GPU {gpu}; log: {log}", markup=False)
    with log.open("w") as stream:
        for index, command in enumerate(commands(benchmark, args)):
            child_env = dict(env)
            if args.cache in ("cold", "both"):
                group = index // 2 if args.cache == "both" else index
                cache_group = (
                    args.scratch / "suite_cache" / args.run_id / benchmark / str(group)
                )
                cold_root = cache_group / "model"
                child_env["BENCH_COLD_CACHE_ROOT"] = str(cold_root)
                child_env["EARTH2STUDIO_DATA_CACHE"] = str(cache_group / "data")
                if command[-1] == "warm" and cold_root.exists():
                    caches = sorted(
                        path for path in cold_root.iterdir() if path.is_dir()
                    )
                    if caches:
                        child_env["EARTH2STUDIO_CACHE"] = str(caches[-1])
            stream.write("$ " + shlex.join(command) + "\n")
            stream.flush()
            try:
                result = subprocess.run(  # noqa: S603 - internal argv, no shell
                    command,
                    cwd=HERE,
                    env=child_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                code = result.returncode
            except OSError as exc:
                stream.write(f"Could not launch: {exc}\n")
                code = 127
            records.append({"command": command, "exit_code": code})
    failed = sum(record["exit_code"] != 0 for record in records)
    console.print(
        f"{'FAILED' if failed else 'DONE'} {benchmark}: {failed}/{len(records)} commands failed",
        style="red" if failed else "green",
        markup=False,
    )
    return {
        "benchmark": benchmark,
        "gpu": gpu,
        "start": started,
        "end": datetime.now(timezone.utc).isoformat(),
        "log": str(log),
        "exit_code": 1 if failed else 0,
        "commands": records,
    }


def run_queue(
    benchmarks: list[str], gpu: str, args: argparse.Namespace, console: Console
) -> list[dict[str, Any]]:
    return [run_benchmark(benchmark, gpu, args, console) for benchmark in benchmarks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", help="Comma-separated IDs, e.g. 01,04 (default: all six)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="One queue per GPU; never share a GPU between benchmarks",
    )
    parser.add_argument(
        "--gpus", help="GPU indices; defaults to CUDA_VISIBLE_DEVICES, then nvidia-smi"
    )
    parser.add_argument("--cache", choices=("cold", "warm", "both"), default="cold")
    parser.add_argument("--warm", action="store_true", help="Alias for --cache both")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One short variant per benchmark; still downloads models/data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without downloads, GPU use or output writes",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("BENCH_RESULTS", HERE.parents[1] / "results")),
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(
            os.environ.get(
                "BENCH_SCRATCH",
                str(Path(tempfile.gettempdir()) / "earth2studio-benchmarks"),
            )
        ),
    )
    args = parser.parse_args()
    if args.warm:
        args.cache = "both"
    selected = (
        args.only.split(",") if args.only is not None else [b[:2] for b in BENCHMARKS]
    )
    if not selected or any(s not in {b[:2] for b in BENCHMARKS} for s in selected):
        parser.error("--only must contain benchmark IDs 01 through 06")
    benchmarks = [b for b in BENCHMARKS if b[:2] in selected]
    try:
        gpus = gpu_ids(args.gpus, args.dry_run)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.parallel:
        gpus = gpus[:1]
    args.results = args.results.expanduser().resolve()
    args.scratch = args.scratch.expanduser().resolve()
    now = datetime.now(timezone.utc)
    args.run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    args.results = args.results / args.run_id
    # Fix satellite initial conditions once, so variants remain comparable across the run.
    goes = now - timedelta(hours=3)
    eu = now - timedelta(minutes=90)
    args.goes_time = os.environ.get("START") or goes.replace(
        minute=goes.minute // 10 * 10, second=0, microsecond=0
    ).strftime("%Y-%m-%dT%H:%M")
    args.eu_time = os.environ.get("START_TIME") or eu.replace(
        minute=eu.minute // 10 * 10, second=0, microsecond=0
    ).strftime("%Y-%m-%dT%H:%M")
    console = Console()
    console.print(
        f"Benchmarks: {len(benchmarks)} | GPUs: {', '.join(gpus)} | cache: {args.cache} | smoke: {args.smoke}",
        markup=False,
    )
    if args.dry_run:
        for index, benchmark in enumerate(benchmarks):
            for command in commands(benchmark, args):
                console.print(
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(gpus[index % len(gpus)])} "
                    + shlex.join(command),
                    markup=False,
                    soft_wrap=True,
                )
        return 0
    args.results.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(run_queue, benchmarks[index :: len(gpus)], gpu, args, console)
            for index, gpu in enumerate(gpus)
        ]
        records = [record for future in futures for record in future.result()]
    manifest = args.results / "run_manifest.json"
    payload = {"run_id": args.run_id, "benchmarks": records}
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    table = Table(title="Suite status")
    for label in ("Benchmark", "GPU", "Status", "Failed / commands"):
        table.add_column(label)
    for record in records:
        table.add_row(
            record["benchmark"],
            record["gpu"],
            "FAILED" if record["exit_code"] else "OK",
            f"{sum(c['exit_code'] != 0 for c in record['commands'])} / {len(record['commands'])}",
        )
    console.print(table)
    failed = any(record["exit_code"] for record in records)
    if any(args.results.glob("[0-9][0-9]*/*.json")):
        for script in ("make_table.py", "analyze.py"):
            result = subprocess.run(  # noqa: S603 - internal report scripts, no shell
                [
                    sys.executable,
                    str(HERE / "common" / script),
                    "--results",
                    str(args.results),
                    "--latest-only",
                ],
                check=False,
            )
            failed |= result.returncode != 0
    else:
        console.print("No result JSON produced; check the benchmark logs.", style="red")
        failed = True
    console.print(f"Results: {args.results}\nManifest: {manifest}", markup=False)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
