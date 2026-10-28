# Benchmark 6: StormScope-EU satellite nowcast (booth pipeline)

## Use case

Use this benchmark to budget EUMETSAT acquisition, inference and output delivery
for a European satellite nowcast service with true-colour and airmass products.

The direct counterpart to the Earthmover GTC Berlin 2026 StormScope-EU booth
pipeline (`scripts/stormscope_eu_pipeline.py` in
https://github.com/earthmover-support/nvidia-gtc-berlin-2026-stormscope_eu-booth).

| | |
|---|---|
| Model | `StormScopeMeteosatEU` (2 denoising experts, 48 diffusion steps, 1120 x 2432 px MTG-I1 FCI box) |
| Data | `MeteosatFCI` FDHSI from the EUMETSAT Data Store, 16 channels (VIS/NIR at 1 km downsampled to 2 km, IR/WV at 2 km) |
| Initial condition | 6 frames at -50..0 min; newest slot at least `--min-lag` (90) min old, or `--start-time` |
| Rollout | `--steps` x 10 min (default 12 = 2 h); step 0 is the observed analysis frame |
| Output | uint8 `true_color` (VIS 0.6/0.5/0.4) and `airmass` (WV 6.3-7.3, IR 9.7-10.5, WV 6.3 inv.) composites, 560 x 1216 spatial chunks, plus lat/lon |
| Store | `--io zarr` (ZarrBackend, sync) or `--io icechunk` (IceChunkBackend, background writers) |
| Commit | `--commit-mode per_step` or `per_init` (icechunk only) |
| Compile | `--compile` runs `model.compile_model()` and a warm-up forward pass |
| Cache | `--cache cold` (default): fresh `EARTH2STUDIO_CACHE`, model weights (~2.4 GB) and FCI products (~0.5-1 GB zip each) re-downloaded |

## Run

From the repository root, use the shared suite runner:

```bash
bash benchmarks/workflows/run_all_benchmarks.sh --only 06             # full cold matrix
bash benchmarks/workflows/run_all_benchmarks.sh --only 06 --cache both # cold + warm
bash benchmarks/workflows/run_all_benchmarks.sh --only 06 --smoke      # one short run
```

For custom model options, source the shared environment and run `bench.py`:

```bash
source benchmarks/workflows/env.sh
"$PY" benchmarks/workflows/06_stormscope_eu/bench.py --help
```

See the [suite README](../README.md) for GPU selection, environment overrides,
credentials and output locations. The shared runner calls `bench.py` directly;
the individual `run_all.sh` is a legacy entry point with its own settings.

EUMETSAT credentials come from `~/.eumdac/credentials` (or the
`EUMETSAT_CONSUMER_KEY/SECRET` env vars). Without the NRT licence, products
younger than ~62 min past slot time return 403, hence the 90 min default lag.

The shared runner assigns the GPU explicitly and queues one benchmark per GPU
in parallel mode. Set `PY6` to use a dedicated environment. Requires the
`stormscope` and data/IO dependencies; this EU model does not use the GOES
Flash backbone's NATTEN kernels.

## Phases

| phase | what |
|---|---|
| `model_download` | `Package.resolve` of config + both expert checkpoints (0 s when warm) |
| `model_load` | checkpoint -> `EDMPreconditioner` x2, `.to(cuda)` |
| `ic_fetch` | Data Store download + decode of 6 FDHSI products at 1 km and 2 km, merge, `map_coords` |
| `compile_warmup` | one forward pass after `compile_model()` (compile variant only) |
| `io_schema` | create coordinate arrays, two uint8 cubes, lat/lon |
| `inference_step` | one 10-min diffusion step (steps 1..N; step 0 is the IC) |
| `composite` | GPU -> CPU copy, true-colour and airmass RGB |
| `io_write` | `backend.write` (sync for zarr, hands off to a thread for icechunk) |
| `io_commit` | icechunk `commit` (flushes pending writes), per step or once |

`extra`: `step_seconds_steady` (mean of steps 2..N), `time_to_first_frame_s`
(IC fetch start -> analysis frame written), `end_to_end_s` (IC fetch start ->
last commit, i.e. booth stages 4+5), `end_to_end_incl_model_s`, `commits`,
`output_bytes`, `cache_bytes`, `gpu_index`, `gpu_shared`.

## Results vs the booth

Booth `docs/END2END_NEARREALTIME_PERFORMANCE.md`, stage 4 (pickup + download +
preprocess, warm model, newest zip only) 2.1 min, stage 5 (inference + ingest to
Arraylake, 12 x 10 min) 3.6 min, ~17 s/step, on a persistent GPU container.
