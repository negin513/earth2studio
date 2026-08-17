# Data loader cold-cache benchmarks: before/after obstore optimizations

Compares data-source fetch time between earth2studio **0.14.0** (last release
before the data-loader optimization series) and **main**, for the maximum set
of variables supported by both versions, with a fully cold cache.

## Optimization PRs covered

| PR | Data sources |
|---|---|
| [#855](https://github.com/NVIDIA/earth2studio/pull/855) | AsyncCachingFileSystem redundant zarr chunk fix (ARCO et al.) |
| [#913](https://github.com/NVIDIA/earth2studio/pull/913) | UFS observation fetch via obstore |
| [#955](https://github.com/NVIDIA/earth2studio/pull/955) | Zarr sources (ARCO, WB2, rx) to obstore |
| [#964](https://github.com/NVIDIA/earth2studio/pull/964) | GFS grib byte-range fetches to obstore |
| [#972](https://github.com/NVIDIA/earth2studio/pull/972) | HRRR, GEFS, CFS, NCAR ERA5 to obstore |
| [#1042](https://github.com/NVIDIA/earth2studio/pull/1042) | GOES / GOES GLM to obstore |
| [#1058](https://github.com/NVIDIA/earth2studio/pull/1058) | ISD, IBTrACS, CFS reforecast, OPERA to obstore |
| [#1061](https://github.com/NVIDIA/earth2studio/pull/1061) | MRMS, NClimGridDaily to obstore |

Sources added after 0.14.0 (CFS, CFS reforecast, OPERA, IBTrACS, Himawari,
GOES GLM, NNJA, GHCN) have no baseline to compare against and are excluded.
NNJA (#971) and Himawari (#1043) are likewise excluded.

## Open-PR rows

Three row groups benchmark *unmerged* optimization PRs: **before** is still
0.14.0, but **after** is the PR branch installed in its own venv (pass it as
`--python-after` and select the rows with `--sources`):

| Row(s) | PR | After branch |
|---|---|---|
| UFSObsConv_PR914 | [#914](https://github.com/NVIDIA/earth2studio/pull/914) parallel decode | `parallel-decode` |
| JPSS | [#1062](https://github.com/NVIDIA/earth2studio/pull/1062) JPSS to obstore | `obstore-jpss-trio` |
| PC_IFS, PC_IFS_5var, PC_OISST | [#1065](https://github.com/NVIDIA/earth2studio/pull/1065) Planetary Computer to obstore + IFS grib byte-range | `pc-obstore` |

PC_IFS_5var re-runs PlanetaryComputerECMWFOpenDataIFS with the 5-variable
subset from the #1065 PR body: the grib byte-range win is largest at few
variables (~6x at 5 vars per the PR) and shrinks to parity at the full
160-var vocab, so the full-vocab PC_IFS row alone would understate the PR.
PC_OISST is the control (netCDF whole-file path, expected parity).

```bash
# example: the #1065 rows
python3 driver.py --python-before <baseline-venv>/bin/python \
    --python-after  <pr1065-venv>/bin/python \
    --vars-dir vars/ --results-dir results/ --cache-dir /tmp/e2s_bench_cache \
    --sources PC_IFS PC_IFS_5var PC_OISST
```

## Methodology

- **Before**: editable install of the `0.14.0` tag. **After**: editable
  install of `main`.
- Each measurement runs in a fresh subprocess with `EARTH2STUDIO_CACHE` /
  `EARTH2STUDIO_DATA_CACHE` pointed at a directory that is wiped beforehand —
  every number is a fully cold fetch.
- **Variables**: all keys of the 0.14.0 lexicon `VOCAB` for the source (a
  subset of main's vocab in every case, so both versions fetch the identical
  variable list). E.g. ARCO 1408, HRRR 648, NCAR ERA5 276, GFS 246.
- Timed span = constructor + `__call__` (0.14.0 does eager network init in
  `__init__`; main is lazy — timing only `__call__` would flatter 0.14.0).
- Constructor kwargs are restricted to those valid in both versions
  (`max_workers=24` where supported; main-only `async_workers`/`retries` left
  at defaults). GOES uses `scan_mode="C"` (CONUS) to keep single-time arrays
  manageable; `EARTH2STUDIO_DISABLE_MSC=1` so 0.14.0 ARCO reads GCS directly.
- Runs are sequential (never concurrent) so they don't compete for bandwidth;
  2+ trials per (source, version); the table reports the median across trials
  (with 2 trials this is their average). Sources with high baseline network
  variance got extra baseline trials.

## Usage

```bash
# 1. variable lists (run with the BASELINE venv — its vocab is the intersection)
<baseline-venv>/bin/python dump_vocab.py --out-dir vars/

# 2. full suite, both versions
python3 driver.py \
    --python-before <baseline-venv>/bin/python \
    --python-after  <main-venv>/bin/python \
    --vars-dir vars/ --results-dir results/ --cache-dir /tmp/e2s_bench_cache \
    --trials 2

# 3. markdown table
python3 make_table.py --results-dir results/
```

## Results

See [RESULTS.md](RESULTS.md).
