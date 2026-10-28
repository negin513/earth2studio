# Workflow benchmark summary

| Workflow | Variant | Model download s | Model load s | Input fetch s | Inference total s | Inference s/step | Total wall s | Peak GPU GiB |
|---|---|---|---|---|---|---|---|---|
| Weather: FCN3 (6 h) | cache=cold, dtype=bf16, io=async_zarr, steps=40 | 13.437 | 119.375 | 4.439 | 35.331 | 0.754 | 220.860 | 49.023 |
| Weather: FCN3 (6 h) | cache=cold, dtype=bf16, io=zarr, steps=40 | 14.285 | 120.450 | 7.409 | 68.027 | 1.587 | 284.686 | 49.023 |
| Weather: FCN3 (6 h) | cache=cold, dtype=fp32, io=async_zarr, steps=40 | 13.303 | 119.326 | 4.494 | 35.628 | 0.754 | 218.219 | 49.023 |
| Weather: FCN3 (6 h) | cache=cold, dtype=fp32, io=zarr, steps=40 | 16.778 | 118.246 | 4.266 | 34.072 | 0.749 | 244.041 | 49.023 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=18, forcing=gfs, samples=1, solver=euler | 45.157 | 5.753 | 14.095 | 3.881 | 0.962 | 69.537 | 1.217 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=18, forcing=gfs, samples=4, solver=euler | 40.245 | 5.949 | 6.462 | 8.690 | 2.171 | 62.047 | 3.757 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=18, forcing=gfs, samples=8, solver=euler | 35.811 | 5.677 | 6.139 | 16.345 | 4.074 | 64.755 | 7.138 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=36, forcing=gfs, samples=1, solver=euler | 52.990 | 5.721 | 7.329 | 7.124 | 1.774 | 73.699 | 1.217 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=36, forcing=gfs, samples=4, solver=euler | 38.779 | 5.642 | 4.151 | 17.188 | 4.294 | 66.668 | 3.757 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=36, forcing=gfs, samples=8, solver=euler | 37.848 | 5.742 | 3.832 | 32.373 | 8.092 | 80.503 | 7.138 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=8, forcing=gfs, samples=1, solver=euler | 46.270 | 5.786 | 12.113 | 2.277 | 0.441 | 67.114 | 1.218 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=8, forcing=gfs, samples=4, solver=euler | 50.041 | 5.693 | 6.829 | 4.019 | 0.996 | 67.235 | 3.757 |
| Downscaling: CorrDiff (forcing time) | cache=cold, diffusion_steps=8, forcing=gfs, samples=8, solver=euler | 38.761 | 5.745 | 3.753 | 7.401 | 1.841 | 56.598 | 7.137 |
| Nowcasting: GOES (10 min) | cache=cold, commit=n/a, compile=0, io=zarr, model=3km_10min_flash | 13.188 | 6.427 | 35.166 | 13.186 | 1.073 | 96.585 | 9.616 |
| Nowcasting: GOES (10 min) | cache=cold, commit=per_init, compile=0, io=icechunk, model=3km_10min | 155.891 | 11.137 | 41.825 | 414.156 | 34.777 | 651.065 | 9.438 |
| Nowcasting: GOES (10 min) | cache=cold, commit=per_init, compile=0, io=icechunk, model=3km_10min_flash | 14.617 | 6.327 | 102.543 | 13.164 | 1.076 | 162.701 | 9.616 |
| Nowcasting: GOES (10 min) | cache=cold, commit=per_init, compile=1, io=icechunk, model=3km_10min_flash | 15.901 | 6.345 | 61.544 | 7.089 | 0.493 | 144.226 | 8.085 |
| Nowcasting: GOES (10 min) | cache=cold, commit=per_step, compile=0, io=icechunk, model=3km_10min_flash | 12.712 | 6.351 | 97.476 | 13.114 | 1.072 | 156.589 | 9.616 |
| Nowcasting: GOES (10 min) | cache=warm, commit=per_step, compile=0, io=icechunk, model=3km_10min_flash | 0.013 | 4.842 | 4.097 | 4.220 | 1.073 | 40.714 | 9.616 |
| S2S: DLESyM (96 h) | batch=1, cache=cold, days=48, members=1, weekly=1 | 5.867 | 3.631 | 71.776 | 7.610 | 0.245 | 92.383 | 2.756 |
| S2S: DLESyM (96 h) | batch=4, cache=cold, days=48, members=4, weekly=1 | 5.747 | 3.454 | 67.880 | 9.775 | 0.316 | 98.953 | 9.294 |

Seconds per step use steady inference timing (first call excluded when available). Ensembles are per batch step; CorrDiff is per forcing time for all requested samples, not per diffusion iteration. Input fetch includes download/decode/preparation. Model load includes DLESyM's lat/lon wrapper. Total wall includes IO, compilation and other overhead; phase columns need not sum to it. Peak GPU is allocated memory.
