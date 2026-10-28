# Benchmark 4: StormScope GOES satellite nowcast (booth-style)

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

```bash
source benchmarks/workflows/env.sh
cd benchmarks/workflows
export CUDA_VISIBLE_DEVICES=3
PY4=/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/users/nsobhani/forks/earth2studio/.venv/bin/python
$PY4 04_stormscope_goes/bench.py --steps 2 --cache warm            # smoke
START=2026-10-28T16:50 BASELINE=1 bash 04_stormscope_goes/run_all.sh  # cold matrix
```

`run_all.sh` runs, cold cache (fresh `EARTH2STUDIO_CACHE` per variant, so model
weights and GOES frames are downloaded from scratch):

1. `--io zarr`
2. `--io icechunk --commit-mode per_step`
3. `--io icechunk --commit-mode per_init`
4. `--io icechunk --commit-mode per_init --compile`
5. (`BASELINE=1`) `--variant 3km_10min` baseline diffusion sampler, icechunk per_init

Pin `START` so every variant sees the same init. Default init is now minus 3 h
floored to 10 min. Logs land in `results/04_stormscope_goes/logs/`.

### Environment note

`env.sh`'s `$PY` (earth2studio-main-bench venv) has no NATTEN, which StormScope
requires. This benchmark therefore uses `forks/earth2studio/.venv` (torch 2.12.1+cu130,
natten 0.21.7, icechunk 2.1.0, zarr 3.3.0) with the same PYTHONPATH overlay of this
clone. Nothing was installed in either venv. The `stormscope` HF package cache was moved
from `/root/.cache/earth2studio/stormscope` (home NFS hit its quota mid-download) to
`forks/e2s_bench_cache/hf_cache/stormscope` and symlinked back.

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
