# Earth2Studio workflow benchmarks

End-to-end, realistic inference workflows timed phase by phase (IC fetch, model
load, per-step inference, IO write/commit) with peak GPU memory. Each benchmark is
a standalone script that shares `common/harness.py` and emits one JSON per run
under `results/<benchmark>/`; `common/make_table.py` collates them into
`results/RESULTS.md`.

| # | folder | workflow | variant axes | headline metric |
|---|---|---|---|---|
| 1 | `01_deterministic_fcn3` | FCN3 from GFS, 10 days @ 6 h, Zarr output | `ZarrBackend` vs `AsyncZarrBackend`; fp32 vs bf16 autocast | s/step, IO s/step |
| 2 | `02_ensemble_hens_lite` | SFNO ensemble + `PrecipitationAFNO` every step, 5 days, CONUS crop | perturbation (SphericalGaussian / BredVector), members 16/32, batch size | member-steps/s, peak GiB |
| 3 | `03_corrdiff_taiwan` | CorrDiff Taiwan driven by GFS at several times; optional forcing from benchmark 1's Zarr | samples {1,4,8} x diffusion steps {8,18,36} | s/sample, peak GiB |
| 4 | `04_stormscope_goes` | StormScope GOES nowcast, 12 x 10 min, raw channels + uint8 true-colour composite | Zarr vs Icechunk, commit per step vs per init, torch.compile | s/step, first-frame latency, end-to-end |
| 5 | `05_s2s_long_rollout` | FuXi-S2S (or DLESyM) 45-day rollout, lagged ensemble, weekly means, lead-time-chunked Zarr | members 1/4 | forecast days per GPU-minute, stability |

Benchmark 4 mirrors the Earthmover GTC Berlin StormScope-EU booth pipeline
(https://github.com/earthmover-support/nvidia-gtc-berlin-2026-stormscope_eu-booth)
so its per-step and end-to-end numbers compare directly with that repo's
`docs/END2END_NEARREALTIME_PERFORMANCE.md`.

## Environment

```bash
source benchmarks/workflows/env.sh     # sets $PY, PYTHONPATH overlay, $BENCH_RESULTS, $BENCH_SCRATCH
cd benchmarks/workflows
CUDA_VISIBLE_DEVICES=0 bash 01_deterministic_fcn3/run_all.sh
$PY common/make_table.py
```

`env.sh` reuses an existing venv (`../earth2studio-main-bench/.venv`, torch 2.13
cu130) and puts this clone first on `PYTHONPATH`, so the benchmarks exercise this
checkout's `earth2studio` without a reinstall. Model weights come from
`$EARTH2STUDIO_CACHE` (default `~/.cache/earth2studio`). Outputs go to
`$BENCH_SCRATCH`, not into the repo.

## Conventions

- Each `bench.py` writes the inference loop explicitly rather than calling
  `earth2studio.run.*`, so phases are separable.
- Phases synchronize CUDA before and after, so per-step numbers are device time.
- `mean_steady_s` drops the first sample of any repeated phase (warm-up).
- IC time comes from `common.harness.pick_recent_init_time`, so all variants of a
  benchmark share the same initial condition on a given day.
