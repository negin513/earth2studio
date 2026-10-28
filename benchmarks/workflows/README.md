# Earth2Studio workflow benchmarks

Run six complete weather inference pipelines and compare model download, input
data fetch, model loading, inference, diagnostics, output writes and peak GPU
memory. These are performance benchmarks; they do not evaluate forecast skill.

| # | Workflow | Use case | Variants | Main metric |
|---|---|---|---|---|
| [01](01_deterministic_fcn3/README.md) | FCN3 from GFS, global 10-day forecast at 6-hour intervals, Zarr output | Size a daily forecast service and choose precision and output strategy | Synchronous/asynchronous Zarr × FP32/BF16 (4) | Seconds per step and IO time |
| [02](02_ensemble_hens_lite/README.md) | SFNO ensemble with precipitation diagnostic, 5 days, CONUS crop | Plan an ensemble product that represents weather uncertainty within a GPU memory budget | Gaussian/bred-vector perturbations × 16/32 members × batch 8/16 (8) | Member-steps per second and peak GiB |
| [03](03_corrdiff_taiwan/README.md) | CorrDiff Taiwan downscaling of four GFS analysis times to about 3 km | Trade regional product diversity and sampling cost against delivery latency | Samples 1/4/8 × diffusion steps 8/18/36 (9) | Seconds per generated sample |
| [04](04_stormscope_goes/README.md) | StormScope GOES, 2-hour US satellite nowcast, raw channels and true-colour images | Choose storage and commit settings for a frequently refreshed nowcast display | Zarr/Icechunk, commit per step/per init, compilation (4) | Step latency, first-frame latency and end-to-end time |
| [05](05_s2s_long_rollout/README.md) | DLESyM coupled atmosphere/ocean rollout, lat/lon output and weekly means; 45 requested days round up to 48 | Estimate compute and storage for subseasonal outlooks and small ensembles | 1/4 members (2) | Member-days per GPU-minute and numerical stability |
| [06](06_stormscope_eu/README.md) | StormScope Meteosat EU, 2-hour satellite nowcast with true-colour and airmass images | Budget data acquisition, inference and delivery for a European nowcast service | Zarr/Icechunk, commit per step/per init, compilation (4) | Step latency, first-frame latency and end-to-end time |

The default full matrix produces **31 result JSON files per cache mode**, from
23 Python processes (CorrDiff runs its nine sweep points in one process).
`BASELINE=1` adds the GOES baseline diffusion variant. The benchmarks run
independently; CorrDiff also accepts `--forcing-zarr` to consume retained FCN3
output in a separate, explicitly chained run.

## Clone and configure

```bash
git clone --branch workflow-benchmarks https://github.com/negin513/earth2studio.git
cd earth2studio
```

Use an NVIDIA CUDA GPU host with the model dependencies installed. `env.sh`
automatically finds this checkout, adds it to `PYTHONPATH`, and uses
`.venv/bin/python` by default. To reuse an existing compatible GPU environment:

```bash
export PY=/absolute/path/to/gpu-venv/bin/python
```

For a new environment, follow the repository's [installation guidance](../../README.md).
The relevant extras in `pyproject.toml` are `fcn3` (01), `sfno`, `perturbation`
and `precip-afno-v2` (02), `corrdiff` (03), `stormscope` (04/06), `dlesym` (05),
and `data`/`utils` for data sources and IO. Icechunk variants require Python
3.12 or newer. CUDA extensions such as NATTEN, torch-harmonics and earth2grid
must match your PyTorch/CUDA installation; DLESyM also uses the PhysicsNeMo
revision pinned in `[tool.uv.sources]`. The broad `all` extra includes unrelated
models and is unnecessary for this suite.

Different benchmarks can use different environments without editing scripts:

```bash
export PY4=/absolute/path/to/stormscope-venv/bin/python
export PY5=/absolute/path/to/dlesym-venv/bin/python
# Optional directory of prebuilt DLESyM dependencies:
export BENCH_OVERLAY_05=/absolute/path/to/dlesym-overlay
```

`PY1` through `PY6` override the interpreter for the corresponding benchmark;
`PY` runs the suite and report tools. Benchmark 06 additionally requires
EUMETSAT credentials, either configured with `eumdac set-credentials` or supplied
as `EUMETSAT_CONSUMER_KEY` and `EUMETSAT_CONSUMER_SECRET`. Its default observation
time is 90 minutes old to allow for data availability and licensing lag.

## Run the suite

All commands below run **from the repository root**. No manual `source` or
per-benchmark launch is needed:

```bash
bash benchmarks/workflows/run_all_benchmarks.sh
```

The default runs all six benchmarks sequentially on the first selected GPU,
with the full cold-cache matrix, then builds the reports. Preview the commands
before a long run, or start with a reduced run:

```bash
# Preview only: no GPU access, downloads or result/output writes
bash benchmarks/workflows/run_all_benchmarks.sh --dry-run
# One short configuration of each benchmark; still downloads and runs models
bash benchmarks/workflows/run_all_benchmarks.sh --smoke
# Run one benchmark at a time on each GPU, queuing the remaining benchmarks
bash benchmarks/workflows/run_all_benchmarks.sh --parallel --gpus 0,1,2,3
# Select benchmarks and cache mode
bash benchmarks/workflows/run_all_benchmarks.sh --only 01,04 --cache warm
bash benchmarks/workflows/run_all_benchmarks.sh --cache both
# Choose persistent result storage and a large scratch disk
bash benchmarks/workflows/run_all_benchmarks.sh --results /path/to/results --scratch /path/to/scratch
```

GPU selection defaults to `CUDA_VISIBLE_DEVICES`, then `nvidia-smi`; `--gpus`
selects numeric GPU IDs explicitly. Parallel execution is bounded to one running
benchmark per selected GPU. All variants of that benchmark run sequentially on
its assigned GPU. Choose GPUs with sufficient free memory; other jobs on the
host can affect timings.

`--cache cold` uses a fresh cache for each variant, including repeated model and
input downloads. `--cache warm` reuses `EARTH2STUDIO_CACHE` (default
`~/.cache/earth2studio`) and can still download missing files. `--cache both`
runs each cold variant followed by a warm pass reusing its downloaded model
and data cache; `--warm` is its compatibility alias. Cold caches and retained
model output can use substantial disk space. Set `BENCH_SCRATCH` or `--scratch`
to a suitable disk. The default scratch location is
`${TMPDIR:-/tmp}/earth2studio-benchmarks`.

The runner calls each `bench.py` directly and prints Rich progress and a final
status table. A failed variant is recorded and the remaining work continues;
the suite exits nonzero if a benchmark or report fails. Full per-run Rich phase
tables are captured in the benchmark logs.

For an individual script and its custom flags:

```bash
source benchmarks/workflows/env.sh
"$PY" benchmarks/workflows/01_deterministic_fcn3/bench.py --help
```

The individual `run_all.sh` scripts remain legacy entry points with their own
settings; use the suite command above for portable configuration and GPU
scheduling. `START` (GOES) and `START_TIME` (Meteosat) optionally pin satellite
initial conditions; otherwise the suite fixes one recent time per satellite
benchmark for all of its variants.

## Results and reports

Each invocation gets a unique `<results-base>/<run_id>/` directory so full
matrices, smoke tests and previous runs stay separate. The results base defaults
to the repository's `results/` directory; `--results` overrides `BENCH_RESULTS`.
`--scratch` overrides `BENCH_SCRATCH`. The runner prints the invocation's results
directory at completion.

| Path under results | Contents |
|---|---|
| `<benchmark>/*.json` | Variant, phase samples/statistics, wall time, peak GPU memory, extra metrics and environment |
| `logs/<benchmark>.log` | Full stdout/stderr and command lines for that benchmark in this invocation |
| `run_manifest.json` | This invocation: assigned GPU, start/end times, commands, exit codes and log paths |
| `RESULTS.md` | Per-run comparison tables; also printed with Rich |
| `ANALYSIS.md` | Cross-benchmark summary, phase breakdowns, variant comparisons and available cold/warm deltas |
| `all_runs.csv` | Flat results for analysis or spreadsheets |
| `phase_breakdown.png` | Phase timing plot |

Reports are generated automatically with `--latest-only`, selecting the newest
record per variant/cache mode within this invocation's results directory. The
manifest and logs identify its exact commands and outcomes.

To regenerate reports without loading a model or requiring CUDA:

```bash
source benchmarks/workflows/env.sh
# Replace this with the results directory printed by the runner:
RUN_RESULTS=/path/to/results/run_id
"$PY" benchmarks/workflows/common/make_table.py --results "$RUN_RESULTS" --latest-only
"$PY" benchmarks/workflows/common/analyze.py --results "$RUN_RESULTS" --latest-only
```

Omit `--latest-only` to include historical records. Rich is already a core
Earth2Studio dependency; redirected logs use plain text and terminals can show
colour. A dash in a table means that metric was not recorded.

## Timing conventions

- Each `bench.py` writes its inference loop explicitly so phases are separable.
- Timed GPU phases synchronize CUDA before and after the measured operation.
- `mean_steady_s` drops the first sample of repeated phases as warm-up.
- Asynchronous `io_write` measures enqueue time; include `io_close` or
  `io_commit` when assessing completed writes.
- Cold and warm timing semantics differ for CorrDiff; see its README before
  comparing total wall times.
- Benchmark 05 reports a basic NaN/max-absolute-value stability check, not
  physical forecast accuracy.
