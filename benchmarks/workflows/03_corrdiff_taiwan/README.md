# Benchmark 3: CorrDiff Taiwan regional downscaling

Realistic high-resolution product: the pretrained `CorrDiffTaiwan` diagnostic
(ERA5-trained, ~3 km over Taiwan) downscales GFS 0.25° analyses at several
consecutive 6 h times, sweeping the number of generated samples and diffusion steps.

## Run

```bash
source benchmarks/workflows/env.sh
export CUDA_VISIBLE_DEVICES=2
cd benchmarks/workflows
# warm smoke test (reuses ~/.cache/earth2studio)
$PY 03_corrdiff_taiwan/bench.py --cache warm --times 1 --samples 1 --diffusion-steps 8
# full cold sweep (default cache mode is cold), then the results table
03_corrdiff_taiwan/run_all.sh            # add --warm to append a warm pass
```

Results: `results/03_corrdiff_taiwan/*.json`, collated by `common/make_table.py`
into `results/RESULTS.md`. Zarr outputs go to `$BENCH_SCRATCH/03/<variant>.zarr`.

## What is measured

| phase | meaning |
|---|---|
| `model_download` | `Package.resolve("corrdiff_inference_package.zip")` (NGC download, ~0 s when warm) |
| `model_load` | `CorrDiffTaiwan.load_model`: unzip, load regression + diffusion checkpoints, move to GPU |
| `ic_fetch` | `fetch_data(GFS)` for all `--times` + `map_coords` onto the 36x40 model input grid |
| `downscale_step` | one `model(x, coords)` call for one time (regression + `number_of_steps` Euler steps x `number_of_samples`) |
| `io_write` | `ZarrBackend.write` of that time's `[sample, variable, 448, 448]` output |

`extra.seconds_per_sample` = steady-state `downscale_step` (first time dropped as
warm-up) / samples. `extra.seconds_per_sample_step` additionally divides by the
diffusion step count. `extra.cache_*` record the cache mode, path and size.

## Cache modes

- `--cache cold` (default): `setup_cache` points `EARTH2STUDIO_CACHE` at a fresh
  empty directory **for every sweep point**, so the model zip and the GFS GRIB
  byte ranges are downloaded from scratch each time. `model_download`,
  `model_load` and `ic_fetch` are then real end-to-end numbers.
- `--cache warm`: model loaded and forcing fetched once; those three phase
  timings are copied into each sweep point's record. `total_wall_s` of a warm
  record covers only the downscale + write loop.

## Options

`--times N` (default 4) consecutive 6 h analysis times ending at
`pick_recent_init_time(hours_back=48)` (or `--end-time ISO`). `--samples` and
`--diffusion-steps` take comma lists. `--solver euler|heun`, `--seed`.

### `--forcing-zarr PATH` (chained to benchmark 1)

Reads the forcing from benchmark 1's FCN3 `ZarrBackend` output instead of GFS:
init 0, the first `--times` lead times, labelled by valid time, for the 12
CorrDiff input variables (`tcwv z500 t500 u500 v500 z850 t850 u850 v850 t2m u10m v10m`).
Exits with a clear message if the path is missing, a variable is absent, there
are too few lead times, or the slice still holds NaNs (benchmark 1 still writing).
The variant is labelled `forcing=fcn3_zarr`.

## Environment notes

The suite venv (`earth2studio-main-bench/.venv`) initially lacked the `corrdiff`
extra. Installed into it on 2026-10-28: `nvidia-physicsnemo 2.2.2` (pulled
torchvision 0.28, timm, warp-lang, tensordict, ...), `einops`, `nvtx`, and
`nvidia-cudnn-cu13` upgraded 9.20.0.48 -> 9.27.0.42 (the 9.20 wheel lacks
`libcudnn_engines_tensor_ir`, which made conv2d fail with
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` against the system 9.21 copy).
