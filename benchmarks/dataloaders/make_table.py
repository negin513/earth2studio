#!/usr/bin/env python3
"""Aggregate cold-cache benchmark JSON results into a markdown comparison table.

Reads results/<source>__<version>__trial<k>.json files produced by
bench_source.py and emits a markdown table:

| Data loader | Data type | # variables | Before (s) | After (s) | PR | Speed-up |
"""

import argparse
import json
import statistics
from pathlib import Path

# Optimization PR responsible for each data source (NVIDIA/earth2studio),
# plus a note for rows whose "after" is an open PR branch rather than main.
PR_MAP = {
    "ARCO": ("#955 (+#855)", ""),
    "WB2ERA5": ("#955", ""),
    "GFS": ("#964", ""),
    "HRRR": ("#972", ""),
    "GEFS_FX": ("#972", ""),
    "NCAR_ERA5": ("#972", ""),
    "GOES": ("#1042", ""),
    "MRMS": ("#1061", ""),
    "NClimGridDaily": ("#1061", ""),
    "ISD": ("#1058", ""),
    "UFSObsConv": ("#913", ""),
    "UFSObsConv_PR914": ("#913 + #914", "open PR: after = `parallel-decode` branch"),
    "JPSS": ("#1062", "open PR: after = `obstore-jpss-trio` branch"),
    "PC_IFS": ("#1065", "open PR: after = `pc-obstore` branch"),
    "PC_IFS_5var": ("#1065", "open PR: after = `pc-obstore` branch; 5-var subset (PR body scenario)"),
    "PC_OISST": ("#1065", "open PR: after = `pc-obstore` branch"),
    "CFS_FX": ("#972", "before = pre-#972 main (source added after 0.14.0)"),
    "CFS_FX_Flux": ("#972", "before = pre-#972 main"),
    "CFS_Reforecast_FX": ("#1058", "before = pre-#1058 main"),
    "CFS_Reforecast_FX_Flux": ("#1058", "before = pre-#1058 main"),
    "OPERA": ("#1058", "before = pre-#1058 main"),
    "IBTrACS": ("#1058", "before = pre-#1058 main"),
    "HimawariAHI": ("#1043", "before = pre-#1043 main; single-tile bbox"),
    "GOESGLMGrid": ("#1042", "before = pre-#1042 main; six 5-min CONUS bins"),
    "GHCNDaily": ("#1063", "open PR: after = `obstore-migrate-ghcn`; before = main"),
    "GHCNHourly": ("#1063", "open PR: after = `obstore-migrate-ghcn`; before = main"),
    "JPSS_ATMS": ("#1062", "open PR: after = `obstore-jpss-trio` branch"),
    "JPSS_CRIS": ("#1062", "open PR: after = `obstore-jpss-trio` branch"),
    "UFSObsSat": ("#913", "npp/atms only; flaky remote archive"),
    "PC_S3AOD": ("#1065", "open PR: after = `pc-obstore` branch"),
    "PC_MODISFire": ("#1065", "open PR: after = `pc-obstore` branch"),
    "PC_GOES": ("#1065", "open PR: after = `pc-obstore` branch"),
}

# Speed-ups reported in the PR bodies themselves, for corroboration. PR-body
# baselines are the contemporary main at PR time (not 0.14.0), and configs
# differ slightly — see README for the configs.
PR_CLAIMED = {
    "ARCO": "3.9x (#855, 82 vars x 10 times); cold ~20% (#955)",
    "WB2ERA5": "2.3x (#855, 82 vars x 10 times)",
    "GFS": "no numbers in #964 body",
    "HRRR": "1.7x (648 vars)",
    "GEFS_FX": "4.0x (214 vars)",
    "NCAR_ERA5": "1.9x (278 vars)",
    "GOES": "~1.3x (12 times x 16 bands)",
    "MRMS": "~6.4x (2 vars, 2 times)",
    "NClimGridDaily": "~1.2x (3 days)",
    "ISD": "1.8x (8 vars)",
    "UFSObsConv": "image only in #913 body",
    "UFSObsConv_PR914": "image only in #914 body",
    "JPSS": "~1.0x (16 M-band vars); 1.6-3.1x on other configs",
    "PC_IFS": "parity at all 160 vars (download-bound)",
    "PC_IFS_5var": "6x at 5 vars (up to ~240x at 1 var)",
    "PC_OISST": "parity",
    "CFS_FX": "2.6x (81 vars)",
    "CFS_FX_Flux": "no numbers in body",
    "CFS_Reforecast_FX": "~20x (81 vars, pygrib rescan fix)",
    "CFS_Reforecast_FX_Flux": "~3x (13 vars)",
    "OPERA": "1.8x (3 vars)",
    "IBTrACS": "1.6x (9 vars)",
    "HimawariAHI": "~1.8x (2ch) / ~2.3x (4 calls)",
    "GOESGLMGrid": "~4x (six 5-min bins)",
    "GHCNDaily": "~26x (by_station redesign)",
    "GHCNHourly": "parity",
    "JPSS_ATMS": "3.1x (n20, +/-10m)",
    "JPSS_CRIS": "1.9x (3ch) / ~1.5x (all ch)",
    "UFSObsSat": "image only in #913 body",
    "PC_S3AOD": "parity",
    "PC_MODISFire": "parity",
    "PC_GOES": "parity",
}

DATA_TYPE = {
    "ARCO": "zarr",
    "WB2ERA5": "zarr",
    "GFS": "grib2",
    "HRRR": "grib2",
    "GEFS_FX": "grib2",
    "MRMS": "grib2",
    "NCAR_ERA5": "netCDF",
    "GOES": "netCDF",
    "NClimGridDaily": "netCDF",
    "ISD": "csv",
    "UFSObsConv": "netCDF (GSI diag)",
    "UFSObsConv_PR914": "netCDF (GSI diag)",
    "JPSS": "netCDF (VIIRS)",
    "PC_IFS": "grib2 (STAC)",
    "PC_IFS_5var": "grib2 (STAC)",
    "PC_OISST": "netCDF (STAC)",
    "CFS_FX": "grib2",
    "CFS_FX_Flux": "grib2",
    "CFS_Reforecast_FX": "grib2 (HTTPS)",
    "CFS_Reforecast_FX_Flux": "grib2 (HTTPS)",
    "OPERA": "HDF5 (ODIM)",
    "IBTrACS": "csv",
    "HimawariAHI": "netCDF (tiles)",
    "GOESGLMGrid": "netCDF (events)",
    "GHCNDaily": "parquet",
    "GHCNHourly": "parquet (HTTPS)",
    "JPSS_ATMS": "BUFR",
    "JPSS_CRIS": "HDF5",
    "UFSObsSat": "netCDF (GSI diag)",
    "PC_S3AOD": "netCDF (STAC)",
    "PC_MODISFire": "HDF (STAC)",
    "PC_GOES": "netCDF (STAC)",
}

# Display names (result-file source keys can encode variants)
DISPLAY = {
    "UFSObsConv_PR914": "UFSObsConv (+PR #914)",
    "PC_IFS": "PlanetaryComputerECMWFOpenDataIFS",
    "PC_IFS_5var": "PlanetaryComputerECMWFOpenDataIFS (5 vars)",
    "PC_OISST": "PlanetaryComputerOISST",
}


def load(results_dir: Path) -> dict:
    data: dict = {}
    for f in sorted(results_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("error"):
            print(f"WARN: skipping errored run {f.name}: {r['error'][:200]}")
            continue
        data.setdefault(r["source"], {}).setdefault(r["version"], []).append(r)
    return data


def fmt_s(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--stat", choices=["median", "mean", "min"], default="median")
    args = ap.parse_args()

    agg = {"median": statistics.median, "mean": statistics.fmean, "min": min}[args.stat]
    data = load(args.results_dir)

    rows = []
    for source in PR_MAP:
        if source not in data:
            continue
        runs = data[source]
        before = agg([r["seconds"] for r in runs["before"]]) if "before" in runs else None
        after = agg([r["seconds"] for r in runs["after"]]) if "after" in runs else None
        n_vars = next(iter(runs.values()))[0]["n_variables"]
        speedup = f"{before / after:.1f}x" if before and after else "-"
        pr, note = PR_MAP[source]
        rows.append(
            (
                DISPLAY.get(source, source),
                DATA_TYPE.get(source, "?"),
                n_vars,
                fmt_s(before),
                fmt_s(after),
                pr,
                speedup,
                PR_CLAIMED.get(source, ""),
                note,
            )
        )

    header = (
        "Data loader", "Data type", "# variables", "Before (s)", "After (s)",
        "PR", "Speed-up", "PR-reported", "Note",
    )
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


if __name__ == "__main__":
    main()
