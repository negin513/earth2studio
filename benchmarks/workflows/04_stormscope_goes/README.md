# Benchmark 4: StormScope GOES satellite nowcast (booth-style)

## Use case

Use this benchmark to choose storage, commit frequency and compilation settings
for a frequently refreshed US satellite nowcast display. First-frame latency
measures how quickly users can see the first forecast product.

US counterpart of the Earthmover GTC Berlin StormScope-EU booth pipeline
(`nvidia-gtc-berlin-2026-stormscope_eu-booth/scripts/stormscope_eu_pipeline.py`):

| | Booth (EU) | This benchmark (US) |
|---|---|---|
| Observations | MTG-I1 FCI, EUMETSAT Data Store (credentials, 62 min licence lag) | GOES-East (goes19) ABI L2 MCMIP CONUS sector, AWS public bucket, no credentials |
| Model | `StormScopeMeteosatEU` | `StormScopeGOES`, variant `3km_10min_flash` (default) or `3km_10min` baseline |
| Input window | 6 frames x 10 min | 6 frames x 10 min (model `input_coords`) |
| Rollout | 12 x 10 min | 12 x 10 min (`--steps`) |
| Products | `true_color`, `airmass` uint8 composites | raw 8 ABI channels (float32) + `true_color` uint8 (R=abi02c, G=synthetic, B=abi01c) |
| Store | Icechunk on Arraylake, one commit per init | local `ZarrBackend` or `IceChunkBackend`, `--commit-mode per_step|per_init` |

Grid: HRRR 3 km CONUS, 1024 x 1792. Chunks `time 1, lead_time 1, variable 1, y 512, x 896`
(2 x 2 per frame, like the booth's 560 x 1216), zstd level 3.

## Run

From the repository root, use the shared suite runner:

```bash
bash benchmarks/workflows/run_all_benchmarks.sh --only 04             # full cold matrix
bash benchmarks/workflows/run_all_benchmarks.sh --only 04 --cache both # cold + warm
bash benchmarks/workflows/run_all_benchmarks.sh --only 04 --smoke      # one short run
```

For custom model options, source the shared environment and run `bench.py`:

```bash
source benchmarks/workflows/env.sh
"$PY" benchmarks/workflows/04_stormscope_goes/bench.py --help
```

See the [suite README](../README.md) for GPU selection, environment overrides,
credentials and output locations. The shared runner calls `bench.py` directly;
the individual `run_all.sh` is a legacy entry point with its own settings.

The full suite matrix runs these four configurations:

1. `--io zarr`
2. `--io icechunk --commit-mode per_step`
3. `--io icechunk --commit-mode per_init`
4. `--io icechunk --commit-mode per_init --compile`

Set `BASELINE=1` to add `--variant 3km_10min` with Icechunk/per-init commits.
The suite pins one initial time for every variant, defaulting to three hours
ago rounded down to ten minutes; override with `START=YYYY-MM-DDTHH:MM`.

## Environment notes

Requires the `stormscope` extra, including CUDA-compatible NATTEN for the Flash
backbone, and Icechunk for its IO variants. Set `PY4` to use a dedicated
StormScope environment. Model weights and GOES data use public sources.

## Phases

| phase | what |
|---|---|
| `model_download` | `package.resolve` of registry, lat/lon, topo and the variant's 3 expert checkpoints (~2 GB). Cold: HF download. Warm: ms. |
| `model_load` | checkpoint deserialisation, `.to(cuda)`, nearest-neighbour GOES to HRRR interpolator |
| `ic_fetch` | six ABI CONUS files (~60 MB each) from `s3://noaa-goes19`, regrid on GPU |
| `compile_warmup` | one forward pass to trigger `torch.compile` (`--compile`) |
| `inference_step` | one 10-min step, CUDA-synchronised (per step) |
| `composite` | true-colour uint8 build on GPU + D2H (per step) |
| `io_write` | `backend.write` of raw + composite. Icechunk is non-blocking: this is the enqueue |
| `io_commit` | `IceChunkBackend.commit`, which flushes in-flight writes first |

`extra`: `step_seconds_steady` (mean of steps 2..N), `time_to_first_frame_s`
(ic_fetch + lead-0 write + first step + its write; model already loaded, like the
booth's persistent container), `end_to_end_s` (ic_fetch + whole rollout incl. writes
and commit), `io_overhead_frac_of_rollout`, `store_bytes`, `cache_bytes`.

Booth reference (EU, stage 5 of `docs/END2END_NEARREALTIME_PERFORMANCE.md`): ~17 s/step,
3.6 min per 12-step init including Arraylake uploads.
