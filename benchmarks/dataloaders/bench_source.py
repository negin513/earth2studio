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

"""Cold-cache benchmark of one earth2studio data source (single version).

Runs in a fresh process against the earth2studio installed in the active
environment. The cache dir is wiped before timing, so every run is cold.

Usage:
    python bench_source.py --spec specs/GFS.json --cache-dir /tmp/bench_cache \
        --output results/raw/GFS__trial0.json [--dry-run]
"""

import argparse
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


def load_spec(path: Path) -> dict:
    """Load and validate a spec file."""
    spec = json.loads(path.read_text())
    missing = [
        k
        for k in ("source", "data_class", "variables", "ctor_kwargs", "time")
        if k not in spec
    ]
    if missing:
        raise ValueError(f"{path.name}: missing keys {missing}")
    if not spec["variables"]:
        raise ValueError(f"{path.name}: empty variables")
    return spec


def build_kwargs(ctor_kwargs: dict) -> dict:
    """Translate JSON-friendly time_tolerance_* keys into timedelta objects."""
    import numpy as np

    kw = dict(ctor_kwargs)
    if "time_tolerance_hours" in kw:
        kw["time_tolerance"] = timedelta(hours=kw.pop("time_tolerance_hours"))
    if "time_tolerance_td_days" in kw:
        kw["time_tolerance"] = timedelta(days=kw.pop("time_tolerance_td_days"))
    if "time_tolerance_np_minutes" in kw:
        kw["time_tolerance"] = np.timedelta64(kw.pop("time_tolerance_np_minutes"), "m")
    return kw


def resolve_class(name: str):
    from earth2studio import data as e2s_data

    return getattr(e2s_data, name)


def parse_times(spec: dict):
    import numpy as np

    raw = spec["time"]
    raw = raw if isinstance(raw, list) else [raw]
    if spec.get("tz_aware"):
        return [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in raw]
    return [np.datetime64(t) for t in raw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate/import only")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    result = {
        "source": None,
        "n_variables_requested": None,
        "n_variables_returned": None,
        "ctor_seconds": None,
        "call_seconds": None,
        "seconds": None,
        "result_shape": None,
        "result_bytes": None,
        "cache_bytes": None,
        "error": None,
    }
    try:
        spec = load_spec(Path(args.spec))
        result["source"] = spec["source"]
        variables = list(spec["variables"])
        result["n_variables_requested"] = len(variables)

        if not args.dry_run:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["EARTH2STUDIO_CACHE"] = str(cache_dir)

        import earth2studio

        result["earth2studio_version"] = getattr(earth2studio, "__version__", "?")
        cls = resolve_class(spec["data_class"])
        kwargs = build_kwargs(spec["ctor_kwargs"])
        times = parse_times(spec)
        if args.dry_run:
            result["dry_run"] = "ok"
            return

        t0 = time.perf_counter()
        ds = cls(**kwargs)  # constructor may do network init; counted
        t1 = time.perf_counter()

        call_args = [times if len(times) > 1 else times[0]]
        if "lead_time_hours" in spec:
            call_args.append([timedelta(hours=spec["lead_time_hours"])])
        call_args.append(variables)
        t2 = time.perf_counter()
        out = ds(*call_args)
        t3 = time.perf_counter()

        result["ctor_seconds"] = t1 - t0
        result["call_seconds"] = t3 - t2
        result["seconds"] = t3 - t0
        if hasattr(out, "shape"):  # xarray DataArray / DataFrame
            result["result_shape"] = list(out.shape)
        if hasattr(out, "nbytes"):
            result["result_bytes"] = int(out.nbytes)
        elif hasattr(out, "memory_usage"):
            result["result_bytes"] = int(out.memory_usage(deep=True).sum())
        # returned variable count
        if hasattr(out, "coords") and "variable" in out.coords:
            result["n_variables_returned"] = int(out.coords["variable"].size)
        elif hasattr(out, "columns") and "variable" in out.columns:
            result["n_variables_returned"] = int(out["variable"].nunique())
        result["cache_bytes"] = sum(
            f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if not args.dry_run:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2))
        brief = {k: result.get(k) for k in ("source", "seconds", "dry_run")}
        brief["error"] = (result["error"] or "")[-300:] or None
        print(json.dumps(brief))
        if args.dry_run and result["error"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
