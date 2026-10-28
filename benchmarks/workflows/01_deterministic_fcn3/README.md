# Benchmark 1: medium-range deterministic forecast (FCN3)

Daily-ops style run: FCN3 initialized from GFS, 40 x 6 h steps (10 days),
0.25 deg global, 72 variables, written to a local Zarr v3 store.

## Run

```bash
source ../env.sh            # venv + PYTHONPATH overlay of this clone
export CUDA_VISIBLE_DEVICES=0
cd ..                       # run from benchmarks/workflows/
python 01_deterministic_fcn3/bench.py --io zarr --dtype fp32            # one variant, cold cache
python 01_deterministic_fcn3/bench.py --io async_zarr --dtype bf16 --steps 2 --cache warm   # smoke
./01_deterministic_fcn3/run_all.sh                                       # full 4-variant cold matrix
./01_deterministic_fcn3/run_all.sh --cache warm                          # optional warm pass
```

Results land in `$BENCH_RESULTS/01_deterministic_fcn3/*.json`; `python
common/make_table.py` collates them into `$BENCH_RESULTS/RESULTS.md`.

## Variant axes

| flag | values | meaning |
|---|---|---|
| `--io` | `zarr`, `async_zarr` | `ZarrBackend` (synchronous, chunk per variable/lead) vs `AsyncZarrBackend` (non-blocking, `parallel_coords` = time, lead_time, pool of 8 writers, `close()` drains at the end) |
| `--dtype` | `fp32`, `bf16` | `torch.autocast(cuda, bfloat16)` wrapped around the model step only; IO always fp32 |
| `--cache` | `cold` (default), `warm` | cold = fresh empty `EARTH2STUDIO_CACHE`, so FCN3 weights (~GBs from HF) and the GFS IC are downloaded from scratch; warm = reuse `~/.cache/earth2studio` and the cached IC tensor |
| `--steps` | int, default 40 | number of 6 h steps |

The IC time is `pick_recent_init_time(hours_back=48)` (00/06/12/18Z grid, 2 days
back) so every variant run on the same day shares it; override with `--time`.

## Phases

| phase | what is timed |
|---|---|
| `model_download` | `Package.resolve()` of the 8 FCN3 package files (checkpoint under `training_checkpoints/`, config, stats, masks). ~0 s when warm. |
| `model_load` | makani `load_model_package` + `.to(cuda)` |
| `ic_fetch` | `GFS()` fetch of 72 variables via `fetch_data` + `map_coords` to the 721x1440 grid (or `torch.load` of the cached IC when warm) |
| `io_setup` | backend construction + `add_array` schema creation |
| `inference_step` | one autoregressive FCN3 step, `torch.cuda.synchronize()` before and after; per step (step 0 is the IC and is not counted) |
| `io_write` | `backend.write()` of one step; per step. For `async_zarr` this is only the host-side copy/enqueue. |
| `io_close` | `AsyncZarrBackend.close()`: time to drain the in-flight writes after the last step |

`extra` holds `steps_per_s` (steady state, first step dropped), total IO time,
output bytes on disk, variable count, cache mode/path/bytes.

## Environment notes

The shared venv (`earth2studio-main-bench/.venv`) had no `makani` /
`torch-harmonics`; they were installed per the clone's `uv.lock` pins (makani
b38fcb27, torch-harmonics 0.8.1 a632ca74 built with `FORCE_CUDA_EXTENSION=1`
for the disco kernels) plus makani's runtime deps (wandb, h5py, numba, ...).
