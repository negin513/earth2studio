#!/usr/bin/env python3
"""Cold-cache benchmark of a single earth2studio data source.

Runs in a fresh process against whichever earth2studio version is installed
in the active environment. The cache is redirected to --cache-dir, which is
wiped before the timed fetch, so every run measures a fully cold cache.

The per-source specs below only use constructor kwargs that are valid in BOTH
the baseline (0.14.0) and optimized (main) versions.

Usage:
    python bench_source.py --source GFS --version before \
        --variables-file vars/GFS.json --cache-dir /tmp/bench_cache \
        --output results/GFS__before__trial0.json
"""

import argparse
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

# async_timeout raised well above the 600 s default: the max-variable ARCO and
# HRRR fetches on the pre-obstore baseline can exceed it.
TIMEOUT = 7200

SOURCES = {
    "ARCO": {
        "class_name": "ARCO",
        "ctor_kwargs": {"verbose": False, "async_timeout": TIMEOUT},
        "time": "2023-01-01T00:00",
    },
    "WB2ERA5": {
        "class_name": "WB2ERA5",
        "ctor_kwargs": {"verbose": False, "async_timeout": TIMEOUT},
        "time": "2022-01-01T00:00",
    },
    "GFS": {
        "class_name": "GFS",
        "ctor_kwargs": {"source": "aws", "verbose": False, "async_timeout": TIMEOUT},
        "time": "2024-01-01T00:00",
    },
    "HRRR": {
        "class_name": "HRRR",
        "ctor_kwargs": {
            "source": "aws",
            "max_workers": 24,
            "verbose": False,
            "async_timeout": TIMEOUT,
        },
        "time": "2024-01-01T00:00",
    },
    "GEFS_FX": {
        "class_name": "GEFS_FX",
        "ctor_kwargs": {
            "member": "gec00",
            "max_workers": 24,
            "verbose": False,
            "async_timeout": TIMEOUT,
        },
        "time": "2024-01-01T00:00",
        "lead_time_hours": 6,
    },
    "NCAR_ERA5": {
        "class_name": "NCAR_ERA5",
        "ctor_kwargs": {"max_workers": 24, "verbose": False, "async_timeout": TIMEOUT},
        "time": "2024-01-01T00:00",
    },
    "GOES": {
        # CONUS scan instead of full disk to keep single-time arrays manageable
        "class_name": "GOES",
        "ctor_kwargs": {
            "satellite": "goes16",
            "scan_mode": "C",
            "max_workers": 24,
            "verbose": False,
            "async_timeout": TIMEOUT,
        },
        "time": "2024-01-01T00:00",
    },
    "MRMS": {
        "class_name": "MRMS",
        "ctor_kwargs": {"max_workers": 24, "verbose": False, "async_timeout": TIMEOUT},
        "time": "2024-01-01T00:00",
    },
    "NClimGridDaily": {
        "class_name": "NClimGridDaily",
        "ctor_kwargs": {"verbose": False, "async_timeout": TIMEOUT},
        "time": "2024-01-01T00:00",
    },
    "ISD": {
        # Station IDs verified against test/data/test_isd.py in both versions
        "class_name": "ISD",
        "ctor_kwargs": {
            "stations": ["72781024243", "72788324220", "72063800224"],
            "time_tolerance_hours": 1,
            "verbose": False,
        },
        "time": "2024-01-01T00:00",
    },
    "UFSObsConv": {
        "class_name": "UFSObsConv",
        "ctor_kwargs": {
            "time_tolerance_hours": 1,
            "max_workers": 24,
            "verbose": False,
        },
        "time": "2024-01-01T00:00",
    },
    # --- rows whose "after" is an open PR branch (see README) ---
    # PR #914: same source as UFSObsConv; separate key so results don't collide
    "UFSObsConv_PR914": {
        "class_name": "UFSObsConv",
        "ctor_kwargs": {
            "time_tolerance_hours": 1,
            "max_workers": 24,
            "verbose": False,
        },
        "time": "2024-01-01T00:00",
    },
    # PR #1062: VIIRS moderate-resolution bands (the lexicon gates variables by
    # product type, so 16 M-band vars is the max coherent set)
    "JPSS": {
        "class_name": "JPSS",
        "ctor_kwargs": {
            "satellite": "noaa-20",
            "product_type": "M",
            "max_workers": 24,
            "verbose": False,
            "async_timeout": TIMEOUT,
        },
        "time": "2024-06-25T12:00",
    },
    # PR #1065: grib byte-range showcase. Rolling ~recent archive on Planetary
    # Computer — keep the timestamp within ~2 weeks of the run date.
    "PC_IFS": {
        "class_name": "PlanetaryComputerECMWFOpenDataIFS",
        "ctor_kwargs": {
            "max_workers": 24,
            "request_timeout": 60,
            "max_retries": 4,
            "verbose": False,
            "async_timeout": TIMEOUT,
        },
        "time": "2026-08-10T00:00",
    },
    # PR #1065's headline scenario: byte-range win is largest at few variables
    # (PR body: ~6x at 5 vars, parity at all 160) — vars file holds the subset
    "PC_IFS_5var": {
        "class_name": "PlanetaryComputerECMWFOpenDataIFS",
        "ctor_kwargs": {
            "max_workers": 24,
            "request_timeout": 60,
            "max_retries": 4,
            "verbose": False,
            "async_timeout": TIMEOUT,
        },
        "time": "2026-08-10T00:00",
    },
    # PR #1065 control: stable multi-decade netCDF archive (no grib path)
    "PC_OISST": {
        "class_name": "PlanetaryComputerOISST",
        "ctor_kwargs": {
            "max_workers": 24,
            "request_timeout": 60,
            "max_retries": 4,
            "verbose": False,
            "async_timeout": TIMEOUT,
        },
        "time": "2024-06-19T00:00",
    },
    # --- rows whose "before" is the commit just before the PR merged ---
    # (sources added to earth2studio after 0.14.0; see README)
    "CFS_FX": {
        "class_name": "CFS_FX",
        "ctor_kwargs": {"member": 1, "source": "aws", "verbose": False,
                        "async_workers": 16, "retries": 3},
        "time": "2024-06-01T00:00",
        "lead_time_hours": 6,
    },
    "CFS_FX_Flux": {
        "class_name": "CFS_FX_Flux",
        "ctor_kwargs": {"member": 1, "source": "aws", "verbose": False,
                        "async_workers": 16, "retries": 3},
        "time": "2024-06-01T00:00",
        "lead_time_hours": 6,
    },
    # 5-day reforecast cycles, 00/06/12/18Z; whole ~22 MB grib per (IC, lead)
    "CFS_Reforecast_FX": {
        "class_name": "CFS_Reforecast_FX",
        "ctor_kwargs": {"verbose": False, "async_workers": 16, "retries": 3},
        "time": "2010-06-15T00:00",
        "lead_time_hours": 6,
    },
    "CFS_Reforecast_FX_Flux": {
        "class_name": "CFS_Reforecast_FX_Flux",
        "ctor_kwargs": {"verbose": False, "async_workers": 16, "retries": 3},
        "time": "2010-06-15T00:00",
        "lead_time_hours": 6,
    },
    # ODYSSEY era (15-min grid): all 3 vars share one 2200x1900 pixel grid
    # (CIRRUS-era refc is 4400x3800 and can't be mixed with the others)
    "OPERA": {
        "class_name": "OPERA",
        "ctor_kwargs": {"verbose": False, "async_workers": 16, "retries": 3},
        "time": "2024-06-01T00:00",
    },
    # region='ACTIVE' is the small CSV; 'ALL' is a ~100 MB archive
    "IBTrACS": {
        "class_name": "IBTrACS",
        "ctor_kwargs": {"region": "ACTIVE", "time_tolerance_td_days": 7,
                        "verbose": False, "async_workers": 4, "retries": 3},
        "time": "2024-09-01T00:00",
    },
    # bbox limits the fetch to one tile; full disk x 16 bands is many GB
    "HimawariAHI": {
        "class_name": "HimawariAHI",
        "ctor_kwargs": {"satellite": "himawari9",
                        "lat_lon_bbox": [-5.0, 141.0, -2.0, 144.0],
                        "verbose": False, "async_workers": 16, "retries": 3},
        "time": "2024-06-15T00:00",
    },
    # PR #1042 body config: six sequential 5-min bins (CONUS bbox is baked in)
    "GOESGLMGrid": {
        "class_name": "GOESGLMGrid",
        "ctor_kwargs": {"satellite": "east", "verbose": False,
                        "async_workers": 24, "retries": 3},
        "time": ["2024-06-01T18:00", "2024-06-01T18:05", "2024-06-01T18:10",
                 "2024-06-01T18:15", "2024-06-01T18:20", "2024-06-01T18:25"],
    },
    # --- open PR #1063 (before = main, after = obstore-migrate-ghcn) ---
    # PR body config: 2 stations, full 12-var vocab (by_year -> by_station)
    "GHCNDaily": {
        "class_name": "GHCNDaily",
        "ctor_kwargs": {"stations": ["USW00013722", "USW00023234"],
                        "time_tolerance_td_days": 0, "verbose": False,
                        "async_workers": 16, "retries": 3},
        "time": "2023-07-04T00:00",
    },
    "GHCNHourly": {
        "class_name": "GHCNHourly",
        "ctor_kwargs": {"stations": ["USW00013874", "USW00013722"],
                        "time_tolerance_np_minutes": 10, "verbose": False,
                        "async_workers": 16, "retries": 3},
        "time": "2024-01-01T12:00",
    },
    # --- open PR #1062, remaining JPSS instruments (before = 0.14.0) ---
    "JPSS_ATMS": {
        "class_name": "JPSS_ATMS",
        "ctor_kwargs": {"satellites": ["n20"], "time_tolerance_np_minutes": 10,
                        "max_workers": 24, "verbose": False},
        "time": "2024-06-01T12:00",
    },
    "JPSS_CRIS": {
        "class_name": "JPSS_CRIS",
        "ctor_kwargs": {"satellites": ["n20"], "time_tolerance_np_minutes": 2,
                        "subsample": 1, "apodize": True,
                        "max_workers": 24, "verbose": False},
        "time": "2024-06-01T12:00",
    },
    # --- merged #913, satellite obs variant (before = 0.14.0) ---
    # satellites default (all platforms) is an enormous fetch — pin npp/atms
    "UFSObsSat": {
        "class_name": "UFSObsSat",
        "ctor_kwargs": {"satellites": ["npp"], "time_tolerance_hours": 1,
                        "max_workers": 24, "verbose": False},
        "time": "2024-01-01T00:00",
    },
    # --- open PR #1065, remaining PC sources (before = 0.14.0) ---
    "PC_S3AOD": {
        "class_name": "PlanetaryComputerSentinel3AOD",
        "ctor_kwargs": {"max_workers": 24, "verbose": False},
        "time": "2024-06-01T00:00",
        "tz_aware": True,
    },
    "PC_MODISFire": {
        "class_name": "PlanetaryComputerMODISFire",
        "ctor_kwargs": {"tile": "h35v10", "max_workers": 24, "verbose": False},
        "time": "2023-07-28T00:00",
        "tz_aware": True,
    },
    "PC_GOES": {
        "class_name": "PlanetaryComputerGOES",
        "ctor_kwargs": {"satellite": "goes16", "scan_mode": "C",
                        "max_workers": 24, "verbose": False},
        "time": "2022-01-13T05:10",
    },
}


def wipe(path: Path) -> None:
    """Delete and recreate --cache-dir. The directory is destroyed each run."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--version", required=True, choices=["before", "after"])
    ap.add_argument("--variables-file", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    wipe(cache_dir)
    os.environ["EARTH2STUDIO_CACHE"] = str(cache_dir)
    os.environ["EARTH2STUDIO_DATA_CACHE"] = str(cache_dir)
    # 0.14.0 can route ARCO through multi-storage-client; disable so both
    # versions read GCS directly (apples-to-apples)
    os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

    result = {
        "source": args.source,
        "version": args.version,
        "n_variables": None,
        "seconds": None,
        "error": None,
    }

    try:
        variables = json.loads(Path(args.variables_file).read_text())
        result["n_variables"] = len(variables)

        import earth2studio
        import numpy as np
        from earth2studio import data as e2s_data

        result["earth2studio_version"] = getattr(earth2studio, "__version__", "unknown")
        spec = SOURCES[args.source]
        ctor_kwargs = dict(spec["ctor_kwargs"])
        if "time_tolerance_hours" in ctor_kwargs:
            ctor_kwargs["time_tolerance"] = timedelta(
                hours=ctor_kwargs.pop("time_tolerance_hours")
            )
        if "time_tolerance_td_days" in ctor_kwargs:
            ctor_kwargs["time_tolerance"] = timedelta(
                days=ctor_kwargs.pop("time_tolerance_td_days")
            )
        if "time_tolerance_np_minutes" in ctor_kwargs:
            ctor_kwargs["time_tolerance"] = np.timedelta64(
                ctor_kwargs.pop("time_tolerance_np_minutes"), "m"
            )

        # 0.14.0 does eager (network) init in __init__ while main is lazy, so
        # the constructor is part of the cold fetch cost — time it too.
        cls = getattr(e2s_data, spec["class_name"])
        t0 = time.perf_counter()
        ds = cls(**ctor_kwargs)
        t1 = time.perf_counter()

        # tz-aware datetimes for sources that require them (PC STAC search);
        # lists of timestamps for multi-bin sources (GOESGLMGrid)
        def as_time(t):
            if spec.get("tz_aware"):
                return datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            return np.datetime64(t)

        t_spec = spec["time"]
        if isinstance(t_spec, list):
            call_args = [[as_time(t) for t in t_spec]]
        else:
            call_args = [as_time(t_spec)]
        if "lead_time_hours" in spec:
            # datetime.timedelta: 0.14.0's GEFS validator calls .total_seconds()
            call_args.append([timedelta(hours=spec["lead_time_hours"])])
        call_args.append(variables)

        t2 = time.perf_counter()
        out = ds(*call_args)
        t3 = time.perf_counter()

        result["ctor_seconds"] = t1 - t0
        result["call_seconds"] = t3 - t2
        result["seconds"] = (t1 - t0) + (t3 - t2)
        result["result_shape"] = list(getattr(out, "shape", []))
        # bytes landed in the cache dir (proxy for download volume)
        result["cache_bytes"] = sum(
            f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        brief = {k: result.get(k) for k in ("source", "version", "seconds")}
        brief["error"] = (result["error"] or "")[-300:] or None
        print(json.dumps(brief))


if __name__ == "__main__":
    main()
