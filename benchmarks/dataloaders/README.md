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
