# Benchmark 5: S2S long rollout (DLESyM)

## Use case

Use this benchmark to estimate GPU throughput and output storage for subseasonal
outlooks, weekly weather summaries and small atmosphere/ocean ensembles. The
stability checks detect numerical failures; they do not measure forecast skill.

Subseasonal forecast with **DLESyM-V1-ERA5** (checkpoint `v1.1`, the ECMWF AI
Weather Quest submission): a coupled atmosphere/ocean model on a HEALPix
`nside=64` (~1 deg) grid. One model step advances **96 h** and emits 16 six-hourly
atmosphere leads (the ocean component every 48 h). `--days 45` rounds up to 12
steps = 48 days. Ensemble members come from the model's conditional-layer-norm
noise (`use_cln`), one member per batch row, seeded with `set_rng`.

## What is measured

| phase | what |
|---|---|
| `model_download` | resolving the package files (config, atmos/ocean `.mdlus`, HEALPix lat/lon, constants) from Hugging Face; ~0 s on a warm cache |
| `model_load` | `DLESyM.load_model(...).to(cuda)` (HEALPix-native model that drives the rollout) |
| `model_load_latlon_wrapper` | second instance `DLESyMLatLon` used only for input prep and the HPX <-> lat-lon regridders |
| `ic_fetch` | ARCO ERA5 (public GCS, no credentials): nine 6-hourly states `[-48 h .. 0 h]`, 12 variables incl. `ttr03`, 0.25 deg |
| `input_prep` | derived variables (`tau300-700`, `ws10m`, SST NaN infill) + lat-lon -> HEALPix regrid |
| `inference_step` | one 96 h coupled step on HEALPix, all members in one batch (per step) |
| `regrid` | HEALPix -> 0.25 deg lat-lon (721 x 1440) of the step's 16 leads (per step) |
| `select` | variable subset + NaN / max-abs stability check |
| `weekly_stats` | accumulate week-1..6 means on GPU (`--weekly-stats`) |
| `io_write` | `ZarrBackend` write, chunks `{ensemble:1, time:1, lead_time:16}` so each step is one chunk along `lead_time` |
| `io_write_weekly` | second small Zarr of the 6 weekly means |

Output is written **on the lat-lon grid** (regridded from HEALPix inside the
loop, timed as `regrid`), reduced to `t2m, msl, z500, t850` to keep IO
manageable (~265 MB per member per step at 0.25 deg). Pass `--variables` to
change it; the model also produces `tau300-700, z1000, tcwv, z250, ws10m, ttr03,
tp6, sst`.

`extra` carries: `inference_s_per_step_steady`, `pipeline_s_per_step_steady`
(inference + regrid + select + io + weekly), `member_days_per_gpu_minute`,
`output_bytes`, `t2m_final_maxabs` + `nan_detected` (crude stability check),
`gpu_index` / `gpu_shared`, and `cache_mode` / `cache_path` / `cache_bytes`.

## Cold cache

`--cache cold` (default) points `EARTH2STUDIO_CACHE` at a fresh empty directory
before anything is imported, so model weights **and** the ARCO ERA5 fetch
download from scratch on every run. `--cache warm` reuses the normal cache.
For a warm cache on scratch, set
`EARTH2STUDIO_CACHE=/path/to/large-disk/warm-cache` before running.

## Environment notes

Requires the `dlesym` extra, including earth2grid with its CUDA HEALPix padding
extension and the PhysicsNeMo revision pinned in `pyproject.toml`. Set `PY5`
for a dedicated environment, or `BENCH_OVERLAY_05` for prebuilt dependencies.

IC time defaults to `2025-06-01T00:00`, supported by the ARCO ERA5 v3 store.
The suite assigns the GPU; `--parallel` queues at most one benchmark per GPU.

## Run

From the repository root, use the shared suite runner:

```bash
bash benchmarks/workflows/run_all_benchmarks.sh --only 05             # full cold matrix
bash benchmarks/workflows/run_all_benchmarks.sh --only 05 --cache both # cold + warm
bash benchmarks/workflows/run_all_benchmarks.sh --only 05 --smoke      # one short run
```

For custom model options, source the shared environment and run `bench.py`:

```bash
source benchmarks/workflows/env.sh
"$PY" benchmarks/workflows/05_s2s_long_rollout/bench.py --help
```

See the [suite README](../README.md) for GPU selection, environment overrides,
credentials and output locations. The shared runner calls `bench.py` directly;
the individual `run_all.sh` is a legacy entry point with its own settings.
