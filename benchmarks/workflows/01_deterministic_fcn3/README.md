# Benchmark 1: medium-range deterministic forecast (FCN3)

## Use case

Use this benchmark to size a daily global forecast service and compare the
latency and GPU memory of FP32/BF16 inference and synchronous/asynchronous
output writes.

Daily-ops style run: FCN3 initialized from GFS, 40 x 6 h steps (10 days),
0.25 deg global, 72 variables, written to a local Zarr v3 store.

## Run

From the repository root, use the shared suite runner:

```bash
bash benchmarks/workflows/run_all_benchmarks.sh --only 01             # full cold matrix
bash benchmarks/workflows/run_all_benchmarks.sh --only 01 --cache both # cold + warm
bash benchmarks/workflows/run_all_benchmarks.sh --only 01 --smoke      # one short run
```

For custom model options, source the shared environment and run `bench.py`:

```bash
source benchmarks/workflows/env.sh
"$PY" benchmarks/workflows/01_deterministic_fcn3/bench.py --help
```

See the [suite README](../README.md) for GPU selection, environment overrides,
credentials and output locations. The shared runner calls `bench.py` directly;
the individual `run_all.sh` is a legacy entry point with its own settings.

Results are written below the suite invocation's results directory;
`RESULTS.md` and `ANALYSIS.md` are generated automatically.

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

Requires the `fcn3` extra, including makani and CUDA-compatible torch-harmonics.
Use this checkout's dependency pins and a compatible GPU environment; see the
[suite README](../README.md).
