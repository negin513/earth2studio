# Benchmark 2: HENS-lite perturbed ensemble with attached diagnostic

Realistic "ops ensemble" workflow: SFNO (73 ch, 0.25 deg) initialised from GFS,
perturbed into N members, rolled out 5 days at 6 h (20 steps) with the
PrecipitationAFNO diagnostic applied on every step, output cropped to CONUS
(lat 20..55 N, lon 230..300 E) and written to a Zarr store.

The loop is written explicitly (mirrors `earth2studio.run.ensemble` +
`run.diagnostic`) so each phase is timed with `common.harness.BenchRecorder`.

## Phases

| phase | what |
|---|---|
| `model_download` | `Package.resolve()` of every SFNO / PrecipitationAFNO file (network; ~0 when warm) |
| `model_load` | deserialise both models and move to GPU |
| `ic_fetch` | GFS 73-variable IC via `fetch_data` (network + GRIB decode) |
| `perturb` | per batch: SphericalGaussian, or BredVector breeding cycles (`--bv-integration-steps`, default 3, upstream default is 20) |
| `inference_step` | per batch-step SFNO forward (step 0 = IC, not counted) |
| `diagnostic_step` | per batch-step `map_coords` to the 720-lat AFNO grid + PrecipitationAFNO forward |
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

Only `t2m u10m v10m msl` (SFNO) and `tp` (PrecipitationAFNO) are written, on the
CONUS crop (141 x 281), to `$BENCH_SCRATCH/02/<variant>.zarr` (deleted after the
run unless `--keep-output`). Writing all 73 channels globally would make IO
dominate and is not what an ensemble product pipeline does.

## Run

```bash
source ../env.sh; export CUDA_VISIBLE_DEVICES=1
$PY 02_ensemble_hens_lite/bench.py --members 2 --steps 2 --cache warm   # smoke
./02_ensemble_hens_lite/run_all.sh          # cold matrix
./02_ensemble_hens_lite/run_all.sh warm     # optional warm pass
```

## Environment notes

The shared venv (`earth2studio-main-bench/.venv`) lacked the `sfno` extra;
`torch-harmonics` and `makani` (uv.lock pinned git revs), `pynvml` and `moviepy`
were installed with `--no-deps` so torch was left untouched.
