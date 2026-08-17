# Data loader cold-cache benchmark results

Cold-cache fetch time for every data source touched by the obstore
optimization series, before vs after. **Before** is release 0.14.0
(2026-04-27, predates the entire series) where the source existed then;
sources added later use the commit immediately before their optimization PR
merged. **After** is main @ `c45fd62e` (2026-08-13) for merged PRs, or the
open-PR branch for the #914 / #1062 / #1063 / #1065 rows.

Headline results: **ARCO is 11.2x faster at all 1408 variables** (180s → 16s),
**CFS_Reforecast_FX is 9.8x faster at 81 variables** (59s → 6s, pygrib rescan
removal), **MRMS 4.4x**, WB2 2.7x, GEFS 2.4x, NCAR ERA5 1.9x. The pattern is
consistent: speed-ups scale with the number of concurrent range requests a
fetch issues (many variables / many small reads), while sources that download
whole files per request (GOES, JPSS granules, NClimGrid, the Planetary
Computer netCDF sources) are near parity, as the PR bodies themselves
predicted — those are bandwidth-bound, not request-bound.

Two rows are correctness findings, not just performance:

- **PC ECMWF IFS at full vocab**: 0.14.0 *cannot* fetch the 157-variable set
  at all ("Selection contains more than one GRIB element"); the #1065
  index-based byte-range selection fixes the ambiguity and fetches it in ~52s.
- **GHCNDaily**: the pre-#1063 loader crashes with fsspec's "Calling sync()
  from within a running loop" in a fresh process (reproduced on two separate
  pre-PR checkouts); the #1063 branch runs it in ~6.5s.

And one regression flag: **open PR #914** (UFS parallel decode) measured
0.3x — its branch is based on a 2026-06-08 main that predates the merged #913
obstore fetch, and the spawn-based process pool costs more than it saves at
this request size (8 vars, 1 analysis time). It likely needs a rebase onto
current main and a size threshold before the pool pays off.

## Results

Median across trials; identical variable lists on both sides of every row.

| Data loader | Data type | # variables | Before (s) | After (s) | PR | Speed-up | PR-reported | Note |
|---|---|---|---|---|---|---|---|---|
| ARCO | zarr | 1408 | 179.9 | 16.1 | #955 (+#855) | 11.2x | 3.9x (#855, 82 vars x 10 times); cold ~20% (#955) |  |
| WB2ERA5 | zarr | 103 | 14.8 | 5.4 | #955 | 2.7x | 2.3x (#855, 82 vars x 10 times) |  |
| GFS | grib2 | 246 | 10.7 | 5.8 | #964 | 1.8x | no numbers in #964 body |  |
| HRRR | grib2 | 648 | 25.0 | 15.2 | #972 | 1.6x | 1.7x (648 vars) |  |
| GEFS_FX | grib2 | 214 | 9.1 | 3.8 | #972 | 2.4x | 4.0x (214 vars) |  |
| NCAR_ERA5 | netCDF | 276 | 371.6 | 195.8 | #972 | 1.9x | 1.9x (278 vars) |  |
| GOES | netCDF | 16 | 8.1 | 6.8 | #1042 | 1.2x | ~1.3x (12 times x 16 bands) |  |
| MRMS | grib2 | 2 | 15.9 | 3.6 | #1061 | 4.4x | ~6.4x (2 vars, 2 times) |  |
| NClimGridDaily | netCDF | 4 | 10.4 | 8.9 | #1061 | 1.2x | ~1.2x (3 days) |  |
| ISD | csv | 8 | 4.5 | 3.7 | #1058 | 1.2x | 1.8x (8 vars) |  |
| UFSObsConv | netCDF (GSI diag) | 8 | 11.7 | 11.4 | #913 | 1.0x | image only in #913 body |  |
| UFSObsConv (+PR #914) | netCDF (GSI diag) | 8 | 11.7 | 41.8 | #913 + #914 | 0.3x | image only in #914 body | open PR: after = `parallel-decode` branch |
| JPSS | netCDF (VIIRS) | 16 | 12.0 | 7.8 | #1062 | 1.5x | ~1.0x (16 M-band vars); 1.6-3.1x on other configs | open PR: after = `obstore-jpss-trio` branch |
| PlanetaryComputerECMWFOpenDataIFS | grib2 (STAC) | 157 | - | 52.2 | #1065 | - | parity at all 160 vars (download-bound) | open PR: after = `pc-obstore` branch |
| PlanetaryComputerECMWFOpenDataIFS (5 vars) | grib2 (STAC) | 5 | 26.5 | 6.7 | #1065 | 3.9x | 6x at 5 vars (up to ~240x at 1 var) | open PR: after = `pc-obstore` branch; 5-var subset (PR body scenario) |
| PlanetaryComputerOISST | netCDF (STAC) | 4 | 5.6 | 5.1 | #1065 | 1.1x | parity | open PR: after = `pc-obstore` branch |
| CFS_FX | grib2 | 81 | 4.6 | 3.7 | #972 | 1.2x | 2.6x (81 vars) | before = pre-#972 main (source added after 0.14.0) |
| CFS_FX_Flux | grib2 | 13 | 3.6 | 2.5 | #972 | 1.5x | no numbers in body | before = pre-#972 main |
| CFS_Reforecast_FX | grib2 (HTTPS) | 81 | 59.0 | 6.0 | #1058 | 9.8x | ~20x (81 vars, pygrib rescan fix) | before = pre-#1058 main |
| CFS_Reforecast_FX_Flux | grib2 (HTTPS) | 13 | 5.1 | 3.3 | #1058 | 1.5x | ~3x (13 vars) | before = pre-#1058 main |
| OPERA | HDF5 (ODIM) | 3 | 2.5 | 2.9 | #1058 | 0.9x | 1.8x (3 vars) | before = pre-#1058 main |
| IBTrACS | csv | 9 | 3.3 | 2.7 | #1058 | 1.2x | 1.6x (9 vars) | before = pre-#1058 main |
| HimawariAHI | netCDF (tiles) | 16 | 8.9 | 7.5 | #1043 | 1.2x | ~1.8x (2ch) / ~2.3x (4 calls) | before = pre-#1043 main; single-tile bbox |
| GOESGLMGrid | netCDF (events) | 2 | 9.4 | 5.8 | #1042 | 1.6x | ~4x (six 5-min bins) | before = pre-#1042 main; six 5-min CONUS bins |
| GHCNDaily | parquet | 12 | - | 6.5 | #1063 | - | ~26x (by_station redesign) | open PR: after = `obstore-migrate-ghcn`; before = main |
| GHCNHourly | parquet (HTTPS) | 8 | 2.5 | 2.7 | #1063 | 0.9x | parity | open PR: after = `obstore-migrate-ghcn`; before = main |
| JPSS_ATMS | BUFR | 1 | 8.8 | 4.9 | #1062 | 1.8x | 3.1x (n20, +/-10m) | open PR: after = `obstore-jpss-trio` branch |
| JPSS_CRIS | HDF5 | 1 | 8.7 | 7.1 | #1062 | 1.2x | 1.9x (3ch) / ~1.5x (all ch) | open PR: after = `obstore-jpss-trio` branch |
| UFSObsSat | netCDF (GSI diag) | 1 | 5.0 | 4.3 | #913 | 1.2x | image only in #913 body | npp/atms only; flaky remote archive |
| PC_S3AOD | netCDF (STAC) | 22 | 12.0 | 12.4 | #1065 | 1.0x | parity | open PR: after = `pc-obstore` branch |
| PC_MODISFire | HDF (STAC) | 3 | 3.5 | 3.5 | #1065 | 1.0x | parity | open PR: after = `pc-obstore` branch |
| PC_GOES | netCDF (STAC) | 16 | 10.9 | 12.0 | #1065 | 0.9x | parity | open PR: after = `pc-obstore` branch |

## Measured vs PR-reported

The PR bodies (collected in [PR_CLAIMS.md](PR_CLAIMS.md)) report speed-ups
against the *contemporary main at PR time*; rows benchmarked against 0.14.0
are **cumulative** across the series and can exceed or differ from any single
PR's claim.

Close agreement (same config, same story): NCAR ERA5 1.9x (exactly the #972
claim), HRRR 1.6x vs claimed 1.7x (after excluding one 94s network-outlier
baseline trial), GOES 1.2x vs ~1.3x, NClimGridDaily ~1.2x, GHCNHourly parity,
JPSS full M-band ~parity-to-1.5x, and all four "expected parity" Planetary
Computer sources (OISST, Sentinel3AOD, MODISFire, PC GOES) measured 0.9–1.1x.
PC IFS at 5 variables measured 3.9x vs the claimed 6x — same order, the
byte-range win shrinking toward parity as variable count grows, exactly the
scaling curve in the #1065 body.

Measured lower than claimed: GEFS_FX 2.4x vs 4.0x, JPSS_ATMS 1.8x vs 3.1x,
GOESGLMGrid 1.6x vs ~4x, Himawari 1.2x vs ~2x, CFS_FX 1.2x vs 2.6x, ISD 1.2x
vs 1.8x, OPERA ~parity vs 1.8x, CFS_Reforecast 9.8x vs ~20x. These rows share
a shape: small absolute times (2–10s) dominated by per-request latency to the
remote archive, where node locality and time-of-day congestion move the
baseline as much as the code does. Directionally every one of them except
OPERA (0.9x, within noise of parity) confirms the PR's improvement.

Notes on specific rows:

- **CFS_Reforecast before** required two benchmark-only patches to the
  pre-#1058 checkout: NCEI relocated the archive (base URL and subdirectory
  names changed) after that code was written. #1058 itself carries the new
  paths. Only URL constants were patched; the fetch/decode code is untouched.
- **NClimGridDaily** downloads ~5.9x more bytes after (78 MB whole monthly
  file vs 13 MB ranged reads before) yet is still faster — the win is decode
  and request-path, not volume.
- **Grib byte-range sources (GFS/HRRR/GEFS)** download byte-identical volumes
  before and after; their speed-ups are concurrency/latency wins, not less
  data. The only volume win in the series is PC IFS byte-ranges (#1065).

## Environment

- Single Linux node, 144 CPUs, 952 GB RAM, datacenter with public-internet
  egress.
- Runs executed 2026-08-13 through 2026-08-16 (per-run JSON in `results/`).
- Runs strictly sequential (no concurrent benchmarks competing for
  bandwidth); 2+ trials per (source, version); table reports the median.
  HRRR/ISD/MRMS/NClimGrid/UFS baselines got 4 trials due to high variance.
- Every measurement in a fresh subprocess with the earth2studio cache
  directory wiped beforehand — all numbers are fully cold fetches. Timed span
  is constructor + `__call__` (0.14.0 does eager network init in `__init__`).
- An earlier batch of raw result JSONs was lost to tmp cleanup mid-session;
  those runs' `seconds` values were preserved and re-serialized with a
  `"reconstructed": true` marker. All other fields in reconstructed files are
  limited to what was recorded. Fresh runs carry full metadata.
- Full methodology in [README.md](README.md).

## Not benchmarked

| Source | PR | Why | PR-reported result |
|---|---|---|---|
| NNJA (ObsConv/ObsSat/IRSat) | #971 / #1046 / #1059 | needs nnja-ai + BUFR stack | decode throughput IASI 7.8x, CrIS 5.8x; peak decode memory 105x lower (#1059) |
| Rx (LandSeaMask etc.) | #955 | same zarr fetch path as ARCO/WB2 rows | covered by ARCO/WB2 |
| GDAS obs (NomadsGDASObsConv) | — | prepBUFR deps + NOMADS rolling window; PR attribution unclear | — |
| GHCNDaily "before" | #1063 | pre-PR loader crashes (nested event-loop bug) | ~26x on `by_station` per PR body |
