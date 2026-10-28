# Benchmark 2: HENS-lite perturbed ensemble with attached diagnostic

## Use case

Use this benchmark to plan an ensemble product that represents forecast
uncertainty, including precipitation, while balancing perturbation cost,
throughput and GPU memory.

Realistic "ops ensemble" workflow: SFNO (73 ch, 0.25 deg) initialised from GFS,
perturbed into N members, rolled out 5 days at 6 h (20 steps) with the
PrecipitationAFNOv2 diagnostic applied on every step, output cropped to CONUS
(lat 20..55 N, lon 230..300 E) and written to a Zarr store.

The loop is written explicitly (mirrors `earth2studio.run.ensemble` +
`run.diagnostic`) so each phase is timed with `common.harness.BenchRecorder`.

## Phases

| phase | what |
|---|---|
| `model_download` | `Package.resolve()` of every SFNO / PrecipitationAFNOv2 file (network; ~0 when warm) |
| `model_load` | deserialise both models and move to GPU |
| `ic_fetch` | GFS 73-variable IC via `fetch_data` (network + GRIB decode) |
| `perturb` | per batch: SphericalGaussian, or BredVector breeding cycles (`--bv-integration-steps`, default 3, upstream default is 20) |
| `inference_step` | per batch-step SFNO forward (step 0 = IC, not counted) |
| `diagnostic_step` | per batch-step `map_coords` to the 720-lat AFNO grid + PrecipitationAFNOv2 forward |
| `io_write` | per batch-step crop to CONUS + `ZarrBackend.write` of the reduced variable set |

`extra.member_steps_per_s` = steady-state (first sample dropped) members x steps
per second over inference + diagnostic; `_px_only` excludes the diagnostic.
Peak GPU memory is `peak_gpu_mem_gib`.

## Variant axes

`--perturbation {spherical_gaussian,bred_vector}`, `--members {16,32}`,
`--batch-size {8,16}` (members per forward pass). `--cache {cold,warm}`
(default cold: fresh empty `EARTH2STUDIO_CACHE` per run, so weights and the GFS
IC download from scratch; `extra.cache_bytes` records what was pulled).

## Output

Only `t2m u10m v10m msl` (SFNO) and `tp` (PrecipitationAFNOv2) are written, on the
CONUS crop (141 x 281), to `$BENCH_SCRATCH/02/<variant>.zarr` (deleted after the
run unless `--keep-output`). Writing all 73 channels globally would make IO
dominate and is not what an ensemble product pipeline does.

## Run

From the repository root, use the shared suite runner:

```bash
bash benchmarks/workflows/run_all_benchmarks.sh --only 02             # full cold matrix
bash benchmarks/workflows/run_all_benchmarks.sh --only 02 --cache both # cold + warm
bash benchmarks/workflows/run_all_benchmarks.sh --only 02 --smoke      # one short run
```

For custom model options, source the shared environment and run `bench.py`:

```bash
source benchmarks/workflows/env.sh
"$PY" benchmarks/workflows/02_ensemble_hens_lite/bench.py --help
```

See the [suite README](../README.md) for GPU selection, environment overrides,
credentials and output locations. The shared runner calls `bench.py` directly;
the individual `run_all.sh` is a legacy entry point with its own settings.

## Environment notes

Requires `sfno`, `perturbation` and `precip-afno-v2` dependencies, including
makani and CUDA-compatible torch-harmonics.

PrecipitationAFNOv2 is used because its inputs are available from SFNO's
73-channel output. The v1 model requires relative humidity fields that this
SFNO output does not provide.

The suite sets `EARTH2STUDIO_PACKAGE_TIMEOUT=7200` unless already configured,
allowing time for the large SFNO checkpoint download.
