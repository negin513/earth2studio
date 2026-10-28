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

## Running everything, step by step

1. **Environment.** No install is needed on the GB200 box; `env.sh` reuses an
   existing venv and overlays this checkout on `PYTHONPATH`:

   ```bash
   cd benchmarks/workflows
   source env.sh          # sets $PY, $BENCH_RESULTS (results/), $BENCH_SCRATCH (outputs + cold caches)
   ```

   Elsewhere, create a venv with `uv pip install -e ".[all]"` from the repo root,
   then edit the `PY=` line in `env.sh`. Benchmark 6 needs EUMETSAT credentials
   (`eumdac set-credentials <key> <secret>`); everything else uses public data.

2. **Run all benchmarks** (cold cache by default, each run gets a fresh empty
   `EARTH2STUDIO_CACHE` so model weights and input data are downloaded):

   ```bash
   bash run_all_benchmarks.sh               # sequential on one GPU, ~1.5-2.5 h
   bash run_all_benchmarks.sh --parallel    # one benchmark per GPU, round-robin
   bash run_all_benchmarks.sh --only 01,04  # subset
   bash run_all_benchmarks.sh --warm        # also run each benchmark's warm-cache pass
   ```

   Or run a single benchmark: `CUDA_VISIBLE_DEVICES=0 bash 01_deterministic_fcn3/run_all.sh`.
   Each `bench.py` also runs standalone; see `--help` and the per-benchmark README.

3. **Timing outputs are saved automatically** under `$BENCH_RESULTS`:

   | path | contents |
   |---|---|
   | `<benchmark>/*.json` | one file per run: per-phase samples and stats, peak GPU memory, variant, `extra` metrics, environment |
   | `logs/<benchmark>.log` | full stdout/stderr of that benchmark's `run_all.sh` |
   | `run_manifest.json` | start, end, GPU and exit code per benchmark for the last invocation |

4. **Analyze** (also run automatically at the end of `run_all_benchmarks.sh`):

   ```bash
   $PY common/make_table.py                 # -> results/RESULTS.md, raw per-run table
   $PY common/analyze.py --latest-only      # -> results/ANALYSIS.md, all_runs.csv, phase_breakdown.png
   ```

   `ANALYSIS.md` has a cross-benchmark summary (best variant, headline metric,
   download vs compute vs IO seconds), per-benchmark variant tables, the share of
   wall time in each bucket, and cold-vs-warm deltas when both passes exist.
   `all_runs.csv` is the flat table for pandas or a spreadsheet. `--latest-only`
   keeps the newest run per variant so re-runs do not double count.

Outputs (Zarr/Icechunk stores, cold caches) go to `$BENCH_SCRATCH`, not the repo.

## Conventions

- Each `bench.py` writes the inference loop explicitly rather than calling
  `earth2studio.run.*`, so phases are separable.
- Phases synchronize CUDA before and after, so per-step numbers are device time.
- `mean_steady_s` drops the first sample of any repeated phase (warm-up).
- IC time comes from `common.harness.pick_recent_init_time`, so all variants of a
  benchmark share the same initial condition on a given day.
