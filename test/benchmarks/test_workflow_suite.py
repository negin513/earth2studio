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

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from rich.console import Console


@pytest.fixture
def suite(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2] / "benchmarks/workflows/common/run_suite.py"
    )
    spec = importlib.util.spec_from_file_location("workflow_run_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("BASELINE", raising=False)
    for number in range(1, 7):
        monkeypatch.delenv(f"PY{number}", raising=False)
        monkeypatch.delenv(f"BENCH_OVERLAY_{number:02}", raising=False)
    monkeypatch.setattr(
        module, "Console", lambda: Console(width=300, color_system=None)
    )
    return module


@pytest.mark.parametrize("cache,multiplier", [("cold", 1), ("warm", 1), ("both", 2)])
def test_full_matrices_and_cache_modes(
    suite: ModuleType, cache: str, multiplier: int
) -> None:
    args = argparse.Namespace(
        cache=cache,
        smoke=False,
        goes_time="2026-10-01T12:00",
        eu_time="2026-10-01T13:00",
    )
    for benchmark, count in zip(suite.BENCHMARKS, (4, 8, 1, 4, 2, 4)):
        commands = suite.commands(benchmark, args)
        assert len(commands) == count * multiplier
        for command in commands:
            mode = command[command.index("--cache") + 1]
            assert mode in (("cold", "warm") if cache == "both" else (cache,))
        if benchmark.startswith(("04", "06")):
            expected = args.goes_time if benchmark.startswith("04") else args.eu_time
            assert all(c[c.index("--start-time") + 1] == expected for c in commands)
        if cache == "both":
            for cold, warm in zip(commands[::2], commands[1::2]):
                assert cold[:-1] == warm[:-1]
                assert (cold[-1], warm[-1]) == ("cold", "warm")


def test_dry_run_previews_all_smoke_commands_without_side_effects(
    suite: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results, scratch = tmp_path / "results", tmp_path / "scratch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_suite.py",
            "--dry-run",
            "--smoke",
            "--warm",
            "--parallel",
            "--gpus",
            "2,3",
            "--results",
            str(results),
            "--scratch",
            str(scratch),
        ],
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Dry run must not launch a subprocess or query a GPU")

    monkeypatch.setattr(suite.subprocess, "run", forbidden)
    monkeypatch.setattr(suite.subprocess, "check_output", forbidden)
    assert suite.main() == 0
    output = capsys.readouterr().out
    assert output.count("--cache cold") == 6
    assert output.count("--cache warm") == 6
    assert output.count("CUDA_VISIBLE_DEVICES=2") == 6
    assert output.count("CUDA_VISIBLE_DEVICES=3") == 6
    assert not results.exists()
    assert not scratch.exists()


def test_gpu_selection_uses_visible_devices_and_explicit_override(
    suite: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert suite.gpu_ids(None, True) == ["0"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 2, 4 ")
    assert suite.gpu_ids(None, False) == ["2", "4"]
    assert suite.gpu_ids("1", False) == ["1"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(suite.shutil, "which", lambda name: "/test/nvidia-smi")
    monkeypatch.setattr(suite.subprocess, "check_output", lambda *a, **kw: "0\n3\n")
    assert suite.gpu_ids(None, False) == ["0", "3"]


@pytest.mark.parametrize("visible", ["", "-1", "0,0", "banana", "GPU-abc"])
def test_invalid_gpu_selection(suite: ModuleType, visible: str) -> None:
    with pytest.raises(ValueError):
        suite.gpu_ids(visible, False)


def test_missing_gpu_explains_dry_run(
    suite: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(suite.subprocess, "check_output", missing)
    with pytest.raises(ValueError, match="dry-run"):
        suite.gpu_ids(None, False)


@pytest.mark.parametrize("selection", ["00", "01,07", ""])
def test_invalid_benchmark_selection(
    suite: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_suite.py", "--dry-run", "--only", selection])
    with pytest.raises(SystemExit) as error:
        suite.main()
    assert error.value.code == 2


@pytest.fixture
def fake_workflows(
    suite: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    workflows = tmp_path / "workflows"
    for benchmark in suite.BENCHMARKS:
        directory = workflows / benchmark
        directory.mkdir(parents=True)
        (directory / "bench.py").write_text("""
import json
import os
from pathlib import Path
import sys
import time

benchmark = Path(__file__).parent.name
start = time.monotonic()
time.sleep(0.05)
end = time.monotonic()
mode = sys.argv[sys.argv.index("--cache") + 1]
if mode == "cold" and os.environ.get("BENCH_COLD_CACHE_ROOT"):
    model_cache = Path(os.environ["BENCH_COLD_CACHE_ROOT"]) / "20261028T120000000000"
    model_cache.mkdir(parents=True, exist_ok=True)
output = Path(os.environ["BENCH_RESULTS"]) / benchmark
output.mkdir(parents=True, exist_ok=True)
(output / f"{mode}.json").write_text(json.dumps({
    "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "start": start, "end": end,
    "scratch": os.environ["BENCH_SCRATCH"], "args": sys.argv[1:],
    "cold_root": os.environ.get("BENCH_COLD_CACHE_ROOT"),
    "model_cache": os.environ.get("EARTH2STUDIO_CACHE"),
    "data_cache": os.environ.get("EARTH2STUDIO_DATA_CACHE"),
}))
raise SystemExit(7 if benchmark.startswith("02") and mode == "cold" else 0)
""")
    common = workflows / "common"
    common.mkdir()
    for report in ("make_table.py", "analyze.py"):
        (common / report).write_text("""
from pathlib import Path
import os
import sys

results = Path(sys.argv[sys.argv.index("--results") + 1])
with (results / "reports.txt").open("a") as stream:
    stream.write(Path(__file__).name + " " + " ".join(sys.argv[1:]) + "\\n")
raise SystemExit(int(os.environ.get("TEST_REPORT_EXIT", "0")))
""")
    monkeypatch.setattr(suite, "HERE", workflows)
    return workflows


def test_suite_queues_per_gpu_continues_failures_and_writes_reports(
    suite: ModuleType,
    fake_workflows: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, scratch = tmp_path / "results", tmp_path / "scratch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_suite.py",
            "--smoke",
            "--cache",
            "both",
            "--parallel",
            "--gpus",
            "2,3",
            "--results",
            str(results),
            "--scratch",
            str(scratch),
        ],
    )
    assert suite.main() == 1
    manifest_path = next(results.glob("*/run_manifest.json"))
    results = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    records = manifest["benchmarks"]
    assert {r["benchmark"] for r in records} == set(suite.BENCHMARKS)
    assert sum(r["exit_code"] != 0 for r in records) == 1
    for record in records:
        assert len(record["commands"]) == 2
        assert Path(record["log"]).is_file()
        expected_gpu = (
            "2" if suite.BENCHMARKS.index(record["benchmark"]) % 2 == 0 else "3"
        )
        assert record["gpu"] == expected_gpu
    failed = next(r for r in records if r["benchmark"].startswith("02"))
    assert [c["exit_code"] for c in failed["commands"]] == [7, 0]
    intervals: dict[str, list[tuple[float, float]]] = {"2": [], "3": []}
    for result in results.glob("[0-9][0-9]*/*.json"):
        data = json.loads(result.read_text())
        assert data["scratch"] == str(scratch)
        intervals[data["gpu"]].append((data["start"], data["end"]))
    for gpu_intervals in intervals.values():
        ordered = sorted(gpu_intervals)
        assert len(ordered) == 6
        assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))
    assert results.name == manifest["run_id"]
    for benchmark in suite.BENCHMARKS:
        cold = json.loads((results / benchmark / "cold.json").read_text())
        warm = json.loads((results / benchmark / "warm.json").read_text())
        assert cold["data_cache"] == warm["data_cache"]
        assert Path(cold["data_cache"]).parent.parent.name == benchmark
        assert Path(warm["model_cache"]).parent == Path(cold["cold_root"])
        assert Path(warm["model_cache"]).is_dir()
    reports = (results / "reports.txt").read_text().splitlines()
    assert [r.split()[0] for r in reports] == ["make_table.py", "analyze.py"]
    assert all("--latest-only" in line for line in reports)


@pytest.mark.parametrize("report_exit,expected", [(0, 0), (9, 1)])
def test_report_failures_propagate_to_suite_status(
    suite: ModuleType,
    fake_workflows: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_exit: int,
    expected: int,
) -> None:
    monkeypatch.setenv("TEST_REPORT_EXIT", str(report_exit))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_suite.py",
            "--smoke",
            "--only",
            "01",
            "--gpus",
            "0",
            "--results",
            str(tmp_path / "results"),
            "--scratch",
            str(tmp_path / "scratch"),
        ],
    )
    assert suite.main() == expected
    report = next((tmp_path / "results").glob("*/reports.txt"))
    assert len(report.read_text().splitlines()) == 2


def test_missing_benchmark_interpreter_is_recorded_and_other_commands_continue(
    suite: ModuleType,
    fake_workflows: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PY1", str(tmp_path / "missing-python"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_suite.py",
            "--smoke",
            "--only",
            "01,03",
            "--gpus",
            "0",
            "--results",
            str(tmp_path / "results"),
            "--scratch",
            str(tmp_path / "scratch"),
        ],
    )
    assert suite.main() == 1
    manifest_path = next((tmp_path / "results").glob("*/run_manifest.json"))
    records = json.loads(manifest_path.read_text())["benchmarks"]
    assert records[0]["commands"][0]["exit_code"] == 127
    assert records[1]["exit_code"] == 0


def test_suite_without_result_files_fails(
    suite: ModuleType,
    fake_workflows: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_suite.py",
            "--smoke",
            "--only",
            "01",
            "--gpus",
            "0",
            "--results",
            str(tmp_path / "results"),
            "--scratch",
            str(tmp_path / "scratch"),
        ],
    )
    stale = tmp_path / "results/older-suite/01_deterministic_fcn3"
    stale.mkdir(parents=True)
    (stale / "old.json").write_text("{}")
    monkeypatch.setattr(
        suite.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0)
    )
    assert suite.main() == 1
    assert not list((tmp_path / "results").rglob("reports.txt"))
