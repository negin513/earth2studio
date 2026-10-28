# Benchmark 2: HENS-lite perturbed ensemble with attached diagnostic

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

PrecipitationAFNOv2 is used rather than v1: v1 needs `r500`/`r850` which SFNO-73ch
does not output (it carries `q`), v2's 20 inputs are all SFNO outputs.

`EARTH2STUDIO_PACKAGE_TIMEOUT=7200` is exported by `run_all.sh`: the 5 GB SFNO
checkpoint from NGC does not finish inside fsspec's 300 s default on a busy link.
