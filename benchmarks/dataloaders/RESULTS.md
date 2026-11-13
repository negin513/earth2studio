<!-- markdownlint-disable MD013 -->
# Sample results: default 20 sources

One cold-cache trial per source, earth2studio 0.21.0a0, run on 2026-11-13 from a
macOS laptop. Sources ran in three concurrent lanes, so they shared bandwidth
and CPU: treat the times as indicative, not as clean sequential measurements.
Reproduce with `python driver.py` (sequential).

Two sources failed in this run: IFS hit the fsspec request timeout and
NCAR_ERA5 exceeded the 15 minute per-source limit.

| Source | Model | earth2studio version | Format | Pulled from | Variables requested | Variables returned | Shape | Median time (s) | Cache (MB) | Data in memory (MB) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACE2 | ACE2ERA5 | 0.21.0a0 | NetCDF | Hugging Face allenai/ACE2-ERA5 | 38 | 38 | 1 x 38 x 180 x 360 (time, variable, lat, lon) | 153.1 | 118.5 | 9.8 | ok (1/1) |
| ARCO | UCast | 0.21.0a0 | Zarr | Google Cloud Storage gs://gcp-public-data-arco-era5 | 83 | 83 | 1 x 83 x 721 x 1440 (time, variable, lat, lon) | 24.8 | 645.0 | 689.4 | ok (1/1) |
| CFS_FX | - | 0.21.0a0 | GRIB2 (byte-range via .idx) | AWS S3 s3://noaa-cfs-pds | 81 | 81 | 1 x 1 x 81 x 181 x 360 (time, lead_time, variable, lat, lon) | 6.3 | 3.2 | 42.2 | ok (1/1) |
| GEFS_FX | - | 0.21.0a0 | GRIB2 (byte-range via .idx) | AWS S3 s3://noaa-gefs-pds (0.5 deg, pgrb2a/pgrb2b products) | 214 | 214 | 1 x 1 x 214 x 361 x 720 (time, lead_time, variable, lat, lon) | 15.4 | 45.2 | 445.0 | ok (1/1) |
| GFS | SFNO | 0.21.0a0 | GRIB2 (byte-range via .idx) | AWS S3 s3://noaa-gfs-bdp-pds | 73 | 73 | 1 x 73 x 721 x 1440 (time, variable, lat, lon) | 21.8 | 63.6 | 606.3 | ok (1/1) |
| GFS_FX | StormCast | 0.21.0a0 | GRIB2 (byte-range via .idx) | AWS S3 s3://noaa-gfs-bdp-pds | 26 | 26 | 1 x 1 x 26 x 721 x 1440 (time, lead_time, variable, lat, lon) | 9.5 | 23.5 | 216.0 | ok (1/1) |
| GHCNHourly | StormCastCONUS | 0.21.0a0 | Parquet (one file per station-year) | NCEI HTTPS https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access | 3 | 3 | 15 x 7 | 5.4 | 2.8 | 0.0 | ok (1/1) |
| GOES | StormScopeGOES | 0.21.0a0 | NetCDF4 (ABI-L2-MCMIP) | AWS S3 s3://noaa-goes16 | 8 | 8 | 1 x 8 x 1500 x 2500 (time, variable, y, x) | 80.8 | 60.0 | 240.0 | ok (1/1) |
| GOESGLMGrid | StormScopeMRMS | 0.21.0a0 | NetCDF4 (GLM-L2-LCFA, ~20 s files) binned to 0.1-degree grid | AWS S3 s3://noaa-goes16 (satellite=east resolves to G16 before 2025-04-04, G19 s3://noaa-goes19 after) | 1 | 1 | 1 x 1 x 350 x 700 (time, variable, lat, lon) | 2.6 | 6.0 | 1.0 | ok (1/1) |
| HRRR | StormCastCONUS | 0.21.0a0 | GRIB2 (byte-range via .idx) | AWS S3 s3://noaa-hrrr-bdp-pds | 99 | 99 | 1 x 99 x 1059 x 1799 (time, variable, hrrr_y, hrrr_x) | 14.0 | 146.4 | 1508.9 | ok (1/1) |
| IFS | AIFS2 | 0.21.0a0 | GRIB2 | AWS S3 ECMWF open data mirror (ecmwf-forecasts) | 125 | - | - | - | - | - | 0/1 ok; fsspec.exceptions.FSTimeoutError |
| ISD | - | 0.21.0a0 | CSV (one file per station-year) | AWS S3 s3://noaa-global-hourly-pds | 8 | 8 | 138 x 9 | 15.2 | 3.7 | 0.0 | ok (1/1) |
| MRMS | StormScopeMRMS | 0.21.0a0 | GRIB2, gzipped | AWS S3 s3://noaa-mrms-pds | 2 | 2 | 1 x 2 x 3500 x 7000 (time, variable, lat, lon) | 4.8 | 4.7 | 392.0 | ok (1/1) |
| NCAR_ERA5 | Aurora1p5 | - | NetCDF4 (HDF5) | AWS S3 s3://nsf-ncar-era5 | None | - | - | - | - | - | 0/1 ok; TIMEOUT |
| NNJAObsConv | HealDAv2 | 0.21.0a0 | BUFR (PrepBUFR cycle file + GPS-RO BUFR dump), decoded with earth2bufr | AWS S3 s3://noaa-reanalyses-pds/observations/reanalysis (conv/prepbufr, gps/gpsro) | 7 | 7 | 4533046 x 20 | 19.8 | 179.4 | 400.1 | ok (1/1) |
| NNJAObsSatwnd | HealDAv2 | 0.21.0a0 | BUFR (NCEP satwnd dump), decoded with earth2bufr | AWS S3 s3://noaa-reanalyses-pds/observations/reanalysis/amv (merged <=2019, satwnd after) | 2 | - | 3235903 x 17 | 20.2 | 149.0 | 250.4 | ok (1/1) |
| PC_IFS | FCN3 | 0.21.0a0 | GRIB2 (byte-range via .index) | Azure Blob via Planetary Computer STAC (ecmwf-forecast) | 72 | 72 | 1 x 72 x 721 x 1440 (time, variable, lat, lon) | 37.4 | 32.2 | 598.0 | ok (1/1) |
| UFSObsConv | HealDA | 0.21.0a0 | NetCDF4 (GSI diag .nc4, whole-file download, read with h5netcdf) | AWS S3 s3://noaa-ufs-gefsv13replay-pds (us-east-1, anonymous, via obstore) | 8 | 8 | 8514267 x 11 | 24.5 | 122.6 | 1696.7 | ok (1/1) |
| UFSObsSat | HealDA | 0.21.0a0 | NetCDF4 (GSI diag .nc4, whole-file download, read with h5netcdf) | AWS S3 s3://noaa-ufs-gefsv13replay-pds (us-east-1, anonymous, via obstore) | 4 | 3 | 4378251 x 16 | 29.0 | 203.3 | 955.5 | ok (1/1) |
| WB2ERA5 | Aurora | 0.21.0a0 | Zarr | Google Cloud Storage gs://weatherbench2 | 69 | 69 | 1 x 69 x 721 x 1440 (time, variable, lat, lon) | 15.0 | 193.0 | 573.1 | ok (1/1) |
