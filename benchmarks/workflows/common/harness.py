"""Shared timing/recording harness for the earth2studio workflow benchmarks.

Usage pattern::

    rec = BenchRecorder(benchmark="01_deterministic_fcn3", variant={"io": "zarr", "dtype": "fp32"})
    with rec.phase("ic_fetch"):
        ...
    with rec.phase("model_load"):
        ...
    for step in ...:
        with rec.phase("inference_step"):   # repeated phases accumulate a list
            ...
    rec.finish(extra={"steps": 40})
    rec.write()  # -> $BENCH_RESULTS/<benchmark>/<variant-slug>__<timestamp>.json

Every phase records wall time; GPU phases also synchronize CUDA before start/stop
so the number is the real device time. Peak GPU memory is tracked across the run.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    import earth2studio

    root = Path(earth2studio.__file__).resolve().parents[1]
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "earth2studio_path": str(root),
        "earth2studio_sha": _git_sha(root),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gib"] = round(p.total_memory / 2**30, 1)
        info["gpu_count"] = torch.cuda.device_count()
    return info


class BenchRecorder:
    def __init__(
        self,
        benchmark: str,
        variant: dict[str, Any],
        results_dir: str | Path | None = None,
        device: str = "cuda",
    ):
        self.benchmark = benchmark
        self.variant = dict(variant)
        self.device = device
        self.results_dir = Path(
            results_dir or os.environ.get("BENCH_RESULTS", "results")
        ) / benchmark
        self.phases: dict[str, list[float]] = {}
        self.extra: dict[str, Any] = {}
        self.started = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        self._total: float | None = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------ phases
    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def phase(self, name: str, sync: bool = True):
        if sync:
            self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                self._sync()
            self.phases.setdefault(name, []).append(time.perf_counter() - t)

    def record(self, name: str, seconds: float) -> None:
        self.phases.setdefault(name, []).append(seconds)

    # ----------------------------------------------------------------- results
    def finish(self, extra: dict[str, Any] | None = None) -> None:
        self._sync()
        self._total = time.perf_counter() - self._t0
        if extra:
            self.extra.update(extra)

    @property
    def peak_gpu_gib(self) -> float | None:
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 2**30, 3)

    def summary(self) -> dict[str, Any]:
        ph = {}
        for k, v in self.phases.items():
            ph[k] = {
                "count": len(v),
                "total_s": round(sum(v), 4),
                "mean_s": round(sum(v) / len(v), 4),
                "min_s": round(min(v), 4),
                "max_s": round(max(v), 4),
            }
            if len(v) > 2:  # steady state: drop first (warmup) sample
                ph[k]["mean_steady_s"] = round(sum(v[1:]) / (len(v) - 1), 4)
        return {
            "benchmark": self.benchmark,
            "variant": self.variant,
            "started_utc": self.started.isoformat(),
            "total_wall_s": None if self._total is None else round(self._total, 3),
            "peak_gpu_mem_gib": self.peak_gpu_gib,
            "phases": ph,
            "phase_samples": {k: [round(x, 4) for x in v] for k, v in self.phases.items()},
            "extra": self.extra,
            "env": environment_info(),
            "argv": sys.argv,
        }

    def slug(self) -> str:
        parts = [f"{k}-{str(v).replace('/', '_')}" for k, v in sorted(self.variant.items())]
        return "__".join(parts) if parts else "default"

    def write(self) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        ts = self.started.strftime("%Y%m%dT%H%M%SZ")
        out = self.results_dir / f"{self.slug()}__{ts}.json"
        out.write_text(json.dumps(self.summary(), indent=2, default=str))
        print(f"[harness] wrote {out}")
        return out

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n=== {self.benchmark} {self.variant} ===")
        print(f"total wall: {s['total_wall_s']} s   peak GPU: {s['peak_gpu_mem_gib']} GiB")
        for k, v in s["phases"].items():
            line = f"  {k:24s} n={v['count']:<4d} total={v['total_s']:9.3f}s mean={v['mean_s']:8.3f}s"
            if "mean_steady_s" in v:
                line += f" steady={v['mean_steady_s']:8.3f}s"
            print(line)
        for k, v in s["extra"].items():
            print(f"  extra.{k}: {v}")


def pick_recent_init_time(
    hours_back: int = 24, step_hours: int = 6
) -> datetime:
    """Return a UTC datetime on the step_hours grid at least hours_back ago.

    Keeps IC fetches reproducible across variants within a run day and safely
    behind the data source publication latency.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    t = now - __import__("datetime").timedelta(hours=hours_back)
    t = t.replace(hour=(t.hour // step_hours) * step_hours)
    return t.replace(tzinfo=None)


def setup_cache(mode: str, benchmark: str) -> str:
    """Point EARTH2STUDIO_CACHE at a fresh empty dir for a cold run.

    earth2studio resolves the cache dir from os.environ at call time (see
    earth2studio/models/auto/package.py and earth2studio/data/utils.py), so
    setting it before the first Package/DataSource call makes both model weights
    and data-source caches cold. Call this before importing any model/data class
    that might resolve the cache at import time, and record the mode in
    `extra["cache_mode"]`. Returns the cache path in use.

    mode: "cold" -> new empty dir under $BENCH_SCRATCH/cold_cache/<benchmark>/<ts>
          "warm" -> leave EARTH2STUDIO_CACHE as is
    """
    if mode == "warm":
        return os.environ.get("EARTH2STUDIO_CACHE", os.path.expanduser("~/.cache/earth2studio"))
    if mode != "cold":
        raise ValueError(f"cache mode must be cold|warm, got {mode}")
    base = Path(os.environ.get("BENCH_SCRATCH", "/tmp/e2s_bench")) / "cold_cache" / benchmark
    path = base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path.mkdir(parents=True, exist_ok=False)
    os.environ["EARTH2STUDIO_CACHE"] = str(path)
    print(f"[harness] cold cache at {path}")
    return str(path)


def cache_size_bytes(path: str) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total
