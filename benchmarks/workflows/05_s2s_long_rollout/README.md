# Benchmark 5: S2S long rollout (DLESyM)

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
`/root` has a user disk quota that was hit during this session; if a warm run
fails with "Disk quota exceeded", point the cache at scratch:
`mkdir -p $BENCH_SCRATCH/warm_cache && export EARTH2STUDIO_CACHE=$BENCH_SCRATCH/warm_cache`.

## Environment notes

- The shared venv (`earth2studio-main-bench/.venv`, physicsnemo 2.2.2) lacks
  `earth2grid` and is too old for the DLESyM checkpoint (`residual_prediction`
  arg). Nothing was installed into it. Both packages live in an overlay dir,
  `$BENCH_SCRATCH/pyoverlay/05`, which `run_all.sh` prepends to `PYTHONPATH`:
  - `earth2grid` built from NVlabs/earth2grid @ `11dcf1b` with
    `CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=10.0 --no-build-isolation`
    (the `healpixpad_cuda` extension must build, otherwise padding fails with
    `'NoneType' object has no attribute 'forward'`).
  - `nvidia-physicsnemo` 2.3.0a0 from NVIDIA/physicsnemo @ `426f755` (the rev
    pinned in `pyproject.toml` for the `dlesym` extra), `--no-deps`.
- IC time defaults to `2025-06-01T00:00` (ARCO ERA5's v3 store covers it; the
  v2 store stops at 2021). A fixed date keeps runs comparable.
- GPU choice: GPUs 0..3 are owned by benchmarks 1..4. `run_all.sh` picks the GPU
  with the least memory in use before each run and `extra.gpu_shared` records
  whether another process was on it; a shared GPU skews per-step timing.

## Run

```bash
source benchmarks/workflows/env.sh && cd benchmarks/workflows
bash 05_s2s_long_rollout/run_all.sh                 # members 1 and 4, 45 days, cold cache
bash 05_s2s_long_rollout/run_all.sh --cache warm    # optional warm pass
$PY 05_s2s_long_rollout/bench.py --days 2 --members 1 --cache warm   # smoke
```
