<!-- markdownlint-disable MD013 -->
# Dataloader benchmarks

Cold-cache timing of each Earth2Studio data source on the currently installed
version. Each source is asked for the largest set of variables that any model
in this repository reads from it, so the numbers reflect a realistic worst-case
model initialisation rather than the full lexicon.

## Sources, models and variable counts

A model counts as paired with a source when the repository uses them together
in `examples/`, `recipes/`, `serve/`, `docs/` or a model docstring. For each
source the model with the most variables wins; its exact variable list is in
`specs/<SOURCE>.json`, along with every other pairing found (`all_pairs`) and
the file evidence.

The default run covers these 20 sources. "Full lexicon" means no model in the
repository reads the source, so the spec requests every variable it offers.

| Spec | Data class | File format | Pulled from | Model with max variables | Variables |
|---|---|---|---|---|---|
| ARCO | `ARCO_ERA5` | Zarr (v2) | Google Cloud Storage `gs://gcp-public-data-arco-era5` | UCast | 83 |
| WB2ERA5 | `WB2ERA5` | Zarr (v2) | Google Cloud Storage `gs://weatherbench2` | Aurora | 69 |
| NCAR_ERA5 | `NCAR_ERA5` | NetCDF4 (HDF5) | AWS S3 `s3://nsf-ncar-era5` | Aurora1p5 | 83 |
| GFS | `GFS` | GRIB2 (byte-range via .idx) | AWS S3 `s3://noaa-gfs-bdp-pds` | SFNO | 73 |
| GFS_FX | `GFS_FX` | GRIB2 (byte-range via .idx) | AWS S3 `s3://noaa-gfs-bdp-pds` | StormCast (conditioning) | 26 |
| GEFS_FX | `GEFS_FX` | GRIB2 (byte-range via .idx) | AWS S3 `s3://noaa-gefs-pds` | full lexicon | 214 |
| CFS_FX | `CFS_FX` | GRIB2 (byte-range via .idx) | AWS S3 `s3://noaa-cfs-pds` | full lexicon | 81 |
| HRRR | `HRRR` | GRIB2 (byte-range via .idx) | AWS S3 `s3://noaa-hrrr-bdp-pds` | StormCastCONUS | 99 |
| IFS | `IFS` | GRIB2 | AWS S3, ECMWF open data mirror (`ecmwf-forecasts`) | AIFS2 | 125 |
| PC_IFS | `PlanetaryComputerECMWFOpenDataIFS` | GRIB2 (byte-range via .index) | Azure Blob via Planetary Computer STAC | FCN3 | 72 |
| GOES | `GOES` | NetCDF4 (ABI-L2-MCMIP) | AWS S3 `s3://noaa-goes16` | StormScopeGOES | 8 |
| GOESGLMGrid | `GOESGLMGrid` | NetCDF4 (GLM-L2-LCFA), binned to a 0.1 degree grid | AWS S3 `s3://noaa-goes16` | StormScopeMRMS | 1 |
| MRMS | `MRMS` | GRIB2, gzipped | AWS S3 `s3://noaa-mrms-pds` | StormScopeMRMS | 2 |
| ACE2 | `ACE2ERA5Data` | NetCDF | Hugging Face `allenai/ACE2-ERA5` | ACE2ERA5 | 38 |
| UFSObsConv | `UFSObsConv` | NetCDF4 (GSI diag, whole-file download) | AWS S3 `s3://noaa-ufs-gefsv13replay-pds` | HealDA | 8 |
| UFSObsSat | `UFSObsSat` | NetCDF4 (GSI diag, whole-file download) | AWS S3 `s3://noaa-ufs-gefsv13replay-pds` | HealDA | 4 |
| NNJAObsConv | `NNJAObsConv` | BUFR (PrepBUFR and GPS-RO) | AWS S3 `s3://noaa-reanalyses-pds` | HealDAv2 | 7 |
| NNJAObsSatwnd | `NNJAObsSatwnd` | BUFR (satwnd dump) | AWS S3 `s3://noaa-reanalyses-pds` | HealDAv2 | 2 |
| GHCNHourly | `GHCNHourly` | Parquet (one file per station-year) | NCEI HTTPS `ncei.noaa.gov` | StormCastCONUS | 3 |
| ISD | `ISD` | CSV (one file per station-year) | AWS S3 `s3://noaa-global-hourly-pds` | full lexicon | 8 |

The Zarr version is detected from the opened store at run time (ARCO and
WB2ERA5 are both Zarr v2). Every spec in `specs/` records its own `format` and
`origin`, and the results table prints them for whatever set you run.

Caveats, also recorded in each spec's `unverified` / `disabled_reason` field:

- Variable lists are the class-level defaults read from the model source. No
  model checkpoint was loaded, so lists that a checkpoint can override
  (StormCastCONUS conditioning, AIFSENS, ACE2ERA5) are not confirmed.
- AIFS-family counts on IFS exclude the inputs the model computes itself
  (cos/sin of latitude, longitude, julian day and local time, and insolation):
  AIFS2 has 134 inputs, 125 of which are read from the source.
- NCAR_ERA5: the Aurora1p5 pairing comes from a docstring only. The only
  executable pairing is FCN3 (72, `recipes/tc_tracking/cfg/helene.yaml`).
  AIFSENS is also allowed by that recipe but its input list is defined by the
  checkpoint (estimated at 88).
- CDS is disabled: it needs credentials (`~/.cdsapirc`) and its AIFSENS list is
  the same estimate.
- CMIP6MultiRealm is disabled: it needs a composite constructor (three CMIP6
  realms) that the generic harness does not build, and fetches about 60 GB.
- CBottle3D is disabled: it is a generative model that needs a GPU and a
  checkpoint, not a remote fetch.
- `specs/` holds 83 specs in total, covering most remote sources in
  `earth2studio.data`; run them with `--preset all` or `--sources`.
- The 14 Dynamical sources (dynamical.org Icechunk stores, Zarr v3) are
  disabled: their fetch deadlocks in the tested environment (Python 3.14.3,
  zarr 3.2.1, xarray 2026.2.0, icechunk 2.1.0). A one-variable `DynamicalGFS`
  call made no progress in 90 seconds.
- Specs for the ECMWF forecast variants, Meteosat/MetOp, CMIP6 and EarthMover
  were written but not all confirmed with a real fetch.
- Sample output for the default 20 sources is in `RESULTS.md`.

## Sample results

One cold-cache trial per source, earth2studio 0.21.0a0, run on 2026-11-13 from a
macOS laptop. Most sources ran in three concurrent lanes, so they shared
bandwidth and CPU: treat the times as indicative, not as clean sequential
measurements. IFS hit the fsspec request timeout and NCAR_ERA5 exceeded the
15 minute per-source limit. The same table is in `RESULTS.md`.

| Source | Model | earth2studio version | Format | Zarr version | Pulled from | Variables requested | Variables returned | Shape | Median time (s) | Cache (MB) | Data in memory (MB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ACE2 | ACE2ERA5 | 0.21.0a0 | NetCDF | - | Hugging Face allenai/ACE2-ERA5 | 38 | 38 | 1 x 38 x 180 x 360 (time, variable, lat, lon) | 153.1 | 118.5 | 9.8 | ok (1/1) |
| ARCO | UCast | 0.21.0a0 | Zarr | v2 | Google Cloud Storage gs://gcp-public-data-arco-era5 | 83 | 83 | 1 x 83 x 721 x 1440 (time, variable, lat, lon) | 30.8 | 645.0 | 689.4 | ok (1/1) |
| CFS_FX | - | 0.21.0a0 | GRIB2 (byte-range via .idx) | - | AWS S3 s3://noaa-cfs-pds | 81 | 81 | 1 x 1 x 81 x 181 x 360 (time, lead_time, variable, lat, lon) | 6.3 | 3.2 | 42.2 | ok (1/1) |
| GEFS_FX | - | 0.21.0a0 | GRIB2 (byte-range via .idx) | - | AWS S3 s3://noaa-gefs-pds (0.5 deg, pgrb2a/pgrb2b products) | 214 | 214 | 1 x 1 x 214 x 361 x 720 (time, lead_time, variable, lat, lon) | 15.4 | 45.2 | 445.0 | ok (1/1) |
| GFS | SFNO | 0.21.0a0 | GRIB2 (byte-range via .idx) | - | AWS S3 s3://noaa-gfs-bdp-pds | 73 | 73 | 1 x 73 x 721 x 1440 (time, variable, lat, lon) | 21.8 | 63.6 | 606.3 | ok (1/1) |
| GFS_FX | StormCast | 0.21.0a0 | GRIB2 (byte-range via .idx) | - | AWS S3 s3://noaa-gfs-bdp-pds | 26 | 26 | 1 x 1 x 26 x 721 x 1440 (time, lead_time, variable, lat, lon) | 9.5 | 23.5 | 216.0 | ok (1/1) |
| GHCNHourly | StormCastCONUS | 0.21.0a0 | Parquet (one file per station-year) | - | NCEI HTTPS https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access | 3 | 3 | 15 x 7 | 5.4 | 2.8 | 0.0 | ok (1/1) |
| GOES | StormScopeGOES | 0.21.0a0 | NetCDF4 (ABI-L2-MCMIP) | - | AWS S3 s3://noaa-goes16 | 8 | 8 | 1 x 8 x 1500 x 2500 (time, variable, y, x) | 80.8 | 60.0 | 240.0 | ok (1/1) |
| GOESGLMGrid | StormScopeMRMS | 0.21.0a0 | NetCDF4 (GLM-L2-LCFA, ~20 s files) binned to 0.1-degree grid | - | AWS S3 s3://noaa-goes16 (satellite=east resolves to G16 before 2025-04-04, G19 s3://noaa-goes19 after) | 1 | 1 | 1 x 1 x 350 x 700 (time, variable, lat, lon) | 2.6 | 6.0 | 1.0 | ok (1/1) |
| HRRR | StormCastCONUS | 0.21.0a0 | GRIB2 (byte-range via .idx) | - | AWS S3 s3://noaa-hrrr-bdp-pds | 99 | 99 | 1 x 99 x 1059 x 1799 (time, variable, hrrr_y, hrrr_x) | 14.0 | 146.4 | 1508.9 | ok (1/1) |
| IFS | AIFS2 | 0.21.0a0 | GRIB2 | - | AWS S3 ECMWF open data mirror (ecmwf-forecasts) | 125 | - | - | - | - | - | 0/1 ok; fsspec.exceptions.FSTimeoutError |
| ISD | - | 0.21.0a0 | CSV (one file per station-year) | - | AWS S3 s3://noaa-global-hourly-pds | 8 | 8 | 138 x 9 | 15.2 | 3.7 | 0.0 | ok (1/1) |
| MRMS | StormScopeMRMS | 0.21.0a0 | GRIB2, gzipped | - | AWS S3 s3://noaa-mrms-pds | 2 | 2 | 1 x 2 x 3500 x 7000 (time, variable, lat, lon) | 4.8 | 4.7 | 392.0 | ok (1/1) |
| NCAR_ERA5 | Aurora1p5 | - | NetCDF4 (HDF5) | - | AWS S3 s3://nsf-ncar-era5 | None | - | - | - | - | - | 0/1 ok; TIMEOUT |
| NNJAObsConv | HealDAv2 | 0.21.0a0 | BUFR (PrepBUFR cycle file + GPS-RO BUFR dump), decoded with earth2bufr | - | AWS S3 s3://noaa-reanalyses-pds/observations/reanalysis (conv/prepbufr, gps/gpsro) | 7 | 7 | 4533046 x 20 | 19.8 | 179.4 | 400.1 | ok (1/1) |
| NNJAObsSatwnd | HealDAv2 | 0.21.0a0 | BUFR (NCEP satwnd dump), decoded with earth2bufr | - | AWS S3 s3://noaa-reanalyses-pds/observations/reanalysis/amv (merged <=2019, satwnd after) | 2 | - | 3235903 x 17 | 20.2 | 149.0 | 250.4 | ok (1/1) |
| PC_IFS | FCN3 | 0.21.0a0 | GRIB2 (byte-range via .index) | - | Azure Blob via Planetary Computer STAC (ecmwf-forecast) | 72 | 72 | 1 x 72 x 721 x 1440 (time, variable, lat, lon) | 37.4 | 32.2 | 598.0 | ok (1/1) |
| UFSObsConv | HealDA | 0.21.0a0 | NetCDF4 (GSI diag .nc4, whole-file download, read with h5netcdf) | - | AWS S3 s3://noaa-ufs-gefsv13replay-pds (us-east-1, anonymous, via obstore) | 8 | 8 | 8514267 x 11 | 24.5 | 122.6 | 1696.7 | ok (1/1) |
| UFSObsSat | HealDA | 0.21.0a0 | NetCDF4 (GSI diag .nc4, whole-file download, read with h5netcdf) | - | AWS S3 s3://noaa-ufs-gefsv13replay-pds (us-east-1, anonymous, via obstore) | 4 | 3 | 4378251 x 16 | 29.0 | 203.3 | 955.5 | ok (1/1) |
| WB2ERA5 | Aurora | 0.21.0a0 | Zarr | v2 | Google Cloud Storage gs://weatherbench2 | 69 | 69 | 1 x 69 x 721 x 1440 (time, variable, lat, lon) | 14.5 | 193.0 | 573.1 | ok (1/1) |

## Methodology

- Each trial runs `bench_source.py` in a fresh subprocess. The cache directory
  (`EARTH2STUDIO_CACHE`) is deleted first, so every trial is cold. Specs do not
  set `cache`, so the source's default caching writes into that directory.
- Timed: constructor (may do network initialisation) plus one `__call__` with
  all variables at one timestamp (and one lead time for forecast sources).
- Recorded: seconds, result shape and bytes, cache bytes, variables requested
  versus returned, and any traceback.
- Sources run sequentially so trials do not compete for bandwidth. Results
  depend on the network; use the median over trials.

## Usage

Install Earth2Studio with the data extra, then run one command. It benchmarks
the default 20 sources once each and prints the results table:

```bash
uv sync --extra data
cd benchmarks/dataloaders
uv run python driver.py
```

Other options:

```bash
python driver.py --trials 3                # median over three cold trials
python driver.py --preset all              # every enabled spec
python driver.py --sources GFS HRRR        # only the named specs
python driver.py --sources CDS             # naming a disabled spec runs it
python driver.py --dry-run                 # validate specs, no network
python make_table.py                       # re-render the table from results
```

Outputs go to `results/raw/*.json` (one per trial), `results/results.json` and
`results/table.md`. `results/` is git-ignored. Each source has a 15 minute
limit by default (`--timeout`).

The `PC_IFS` spec reads a rolling archive: update its `time` to a recent
6-hourly cycle before running.
