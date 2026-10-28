# Workflow benchmark analysis

## Summary (cold cache, best variant per benchmark)

| benchmark | variant | headline | total wall s | download+load s | compute s | io s | peak GiB | cache GB |
|---|---|---|---|---|---|---|---|---|
| 01_deterministic_fcn3 | dtype=fp32, io=zarr, steps=40 | 1.33 steps/s | 244.0 | 139.3 | 34.1 | 35.0 | 49.02 | 2.91 |
| 03_corrdiff_taiwan | diffusion_steps=8, forcing=gfs, samples=8, solver=euler | 0.23 s/sample | 56.6 | 48.3 | 7.4 | 0.4 | 7.14 | 1.56 |
| 04_stormscope_goes | commit=per_init, compile=1, io=icechunk, model=3km_10min_flash | 0.49 s/step | 144.2 | 83.8 | 32.0 | 2.0 | 8.09 | 2.50 |
| 05_s2s_long_rollout | batch=4, days=48, members=4, weekly=1 | 927.92 fcst days/GPU-min | 99.0 | 76.1 | 10.0 | 8.3 | 9.29 | 1.73 |

## 01_deterministic_fcn3

4 runs. GPU NVIDIA GB200, torch 2.13.0+cu130, earth2studio 4070f04f.

### Variants

| variant | cache | steps/s | total wall s | peak GiB | ic_fetch (steady mean s) | inference_step (steady mean s) | io_close (steady mean s) | io_setup (steady mean s) | io_write (steady mean s) | model_download (steady mean s) | model_load (steady mean s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dtype=bf16, io=async_zarr, steps=40 | cold | 1.33 | 220.9 | 49.02 | 4.439 | 0.754 | 0.478 | 1.788 | 0.185 | 13.437 | 119.375 |
| dtype=bf16, io=zarr, steps=40 | cold | 0.63 | 284.7 | 49.02 | 7.409 | 1.587 | 0.000 | 1.580 | 0.848 | 14.285 | 120.450 |
| dtype=fp32, io=async_zarr, steps=40 | cold | 1.33 | 218.2 | 49.02 | 4.494 | 0.754 | 0.140 | 2.174 | 0.123 | 13.303 | 119.326 |
| dtype=fp32, io=zarr, steps=40 | cold | 1.33 | 244.0 | 49.02 | 4.266 | 0.749 | 0.000 | 1.491 | 0.817 | 16.778 | 118.246 |

### Where the time goes (share of total wall)

| variant | cache | download+load % | compute % | io % | other/untimed % |
|---|---|---|---|---|---|
| dtype=bf16, io=async_zarr, steps=40 | cold | 62 | 16 | 4 | 17 |
| dtype=bf16, io=zarr, steps=40 | cold | 50 | 24 | 13 | 13 |
| dtype=fp32, io=async_zarr, steps=40 | cold | 63 | 16 | 3 | 17 |
| dtype=fp32, io=zarr, steps=40 | cold | 57 | 14 | 14 | 15 |

## 03_corrdiff_taiwan

9 runs. GPU NVIDIA GB200, torch 2.13.0+cu130, earth2studio 0952d61f.

### Variants

| variant | cache | s/sample | total wall s | peak GiB | downscale_step (steady mean s) | ic_fetch (steady mean s) | io_write (steady mean s) | model_download (steady mean s) | model_load (steady mean s) |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_steps=18, forcing=gfs, samples=1, solver=euler | cold | 0.96 | 69.5 | 1.22 | 0.962 | 14.095 | 0.048 | 45.157 | 5.753 |
| diffusion_steps=18, forcing=gfs, samples=4, solver=euler | cold | 0.54 | 62.0 | 3.76 | 2.171 | 6.462 | 0.076 | 40.245 | 5.949 |
| diffusion_steps=18, forcing=gfs, samples=8, solver=euler | cold | 0.51 | 64.8 | 7.14 | 4.074 | 6.139 | 0.086 | 35.811 | 5.677 |
| diffusion_steps=36, forcing=gfs, samples=1, solver=euler | cold | 1.77 | 73.7 | 1.22 | 1.774 | 7.329 | 0.046 | 52.990 | 5.721 |
| diffusion_steps=36, forcing=gfs, samples=4, solver=euler | cold | 1.07 | 66.7 | 3.76 | 4.294 | 4.151 | 0.065 | 38.779 | 5.642 |
| diffusion_steps=36, forcing=gfs, samples=8, solver=euler | cold | 1.01 | 80.5 | 7.14 | 8.092 | 3.832 | 0.089 | 37.848 | 5.742 |
| diffusion_steps=8, forcing=gfs, samples=1, solver=euler | cold | 0.44 | 67.1 | 1.22 | 0.441 | 12.113 | 0.044 | 46.270 | 5.786 |
| diffusion_steps=8, forcing=gfs, samples=4, solver=euler | cold | 0.25 | 67.2 | 3.76 | 0.996 | 6.829 | 0.066 | 50.041 | 5.693 |
| diffusion_steps=8, forcing=gfs, samples=8, solver=euler | cold | 0.23 | 56.6 | 7.14 | 1.841 | 3.753 | 0.096 | 38.761 | 5.745 |

### Where the time goes (share of total wall)

| variant | cache | download+load % | compute % | io % | other/untimed % |
|---|---|---|---|---|---|
| diffusion_steps=18, forcing=gfs, samples=1, solver=euler | cold | 93 | 6 | 0 | 1 |
| diffusion_steps=18, forcing=gfs, samples=4, solver=euler | cold | 85 | 14 | 0 | 1 |
| diffusion_steps=18, forcing=gfs, samples=8, solver=euler | cold | 74 | 25 | 1 | 1 |
| diffusion_steps=36, forcing=gfs, samples=1, solver=euler | cold | 90 | 10 | 0 | 0 |
| diffusion_steps=36, forcing=gfs, samples=4, solver=euler | cold | 73 | 26 | 0 | 1 |
| diffusion_steps=36, forcing=gfs, samples=8, solver=euler | cold | 59 | 40 | 0 | 0 |
| diffusion_steps=8, forcing=gfs, samples=1, solver=euler | cold | 96 | 3 | 0 | 1 |
| diffusion_steps=8, forcing=gfs, samples=4, solver=euler | cold | 93 | 6 | 0 | 1 |
| diffusion_steps=8, forcing=gfs, samples=8, solver=euler | cold | 85 | 13 | 1 | 1 |

## 04_stormscope_goes

6 runs. GPU NVIDIA GB200, torch 2.12.1+cu130, earth2studio c00d096a.

### Variants

| variant | cache | s/step | total wall s | peak GiB | compile_warmup (steady mean s) | composite (steady mean s) | ic_fetch (steady mean s) | inference_step (steady mean s) | io_commit (steady mean s) | io_setup (steady mean s) | io_write (steady mean s) | model_download (steady mean s) | model_load (steady mean s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| commit=n/a, compile=0, io=zarr, model=3km_10min_flash | cold | 1.07 | 96.6 | 9.62 |  | 0.0006 | 35.166 | 1.073 |  | 0.167 | 0.112 | 13.188 | 6.427 |
| commit=per_init, compile=0, io=icechunk, model=3km_10min | cold | 34.78 | 651.1 | 9.44 |  | 0.0009 | 41.825 | 34.777 | 0.165 | 0.086 | 0.037 | 155.891 | 11.137 |
| commit=per_init, compile=0, io=icechunk, model=3km_10min_flash | cold | 1.08 | 162.7 | 9.62 |  | 0.0007 | 102.543 | 1.076 | 0.088 | 0.083 | 0.059 | 14.617 | 6.327 |
| commit=per_init, compile=1, io=icechunk, model=3km_10min_flash | cold | 0.49 | 144.2 | 8.09 | 24.929 | 0.0007 | 61.544 | 0.493 | 0.088 | 0.089 | 0.152 | 15.901 | 6.345 |
| commit=per_step, compile=0, io=icechunk, model=3km_10min_flash | cold | 1.07 | 156.6 | 9.62 |  | 0.0006 | 97.476 | 1.072 | 0.071 | 0.117 | 0.101 | 12.712 | 6.351 |
| commit=per_step, compile=0, io=icechunk, model=3km_10min_flash | warm | 1.07 | 40.7 | 9.62 |  | 0.0006 | 4.097 | 2.110 | 0.097 | 0.088 | 0.021 | 0.013 | 4.842 |

### Where the time goes (share of total wall)

| variant | cache | download+load % | compute % | io % | other/untimed % |
|---|---|---|---|---|---|
| commit=n/a, compile=0, io=zarr, model=3km_10min_flash | cold | 57 | 14 | 2 | 28 |
| commit=per_init, compile=0, io=icechunk, model=3km_10min | cold | 32 | 64 | 0 | 4 |
| commit=per_init, compile=0, io=icechunk, model=3km_10min_flash | cold | 76 | 8 | 1 | 15 |
| commit=per_init, compile=1, io=icechunk, model=3km_10min_flash | cold | 58 | 22 | 1 | 18 |
| commit=per_step, compile=0, io=icechunk, model=3km_10min_flash | cold | 74 | 8 | 1 | 16 |
| commit=per_step, compile=0, io=icechunk, model=3km_10min_flash | warm | 22 | 10 | 1 | 66 |

### Cold vs warm cache

| variant | cold wall s | warm wall s | cold penalty s | cold download+load s | cache GB fetched |
|---|---|---|---|---|---|
| commit=per_step, compile=0, io=icechunk, model=3km_10min_flash | 156.6 | 40.7 | 115.9 | 116.5 | 2.50 |

## 05_s2s_long_rollout

2 runs. GPU NVIDIA GB200, torch 2.13.0+cu130, earth2studio 4070f04f.

### Variants

| variant | cache | fcst days/GPU-min | total wall s | peak GiB | ic_fetch (steady mean s) | inference_step (steady mean s) | input_prep (steady mean s) | io_write (steady mean s) | io_write_weekly (steady mean s) | model_download (steady mean s) | model_load (steady mean s) | model_load_latlon_wrapper (steady mean s) | regrid (steady mean s) | select (steady mean s) | weekly_stats (steady mean s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batch=1, days=48, members=1, weekly=1 | cold | 578.89 | 92.4 | 2.76 | 71.776 | 0.245 | 0.376 | 0.162 | 0.211 | 5.867 | 2.665 | 0.966 | 0.0049 | 0.0023 | 0.0006 |
| batch=4, days=48, members=4, weekly=1 | cold | 927.92 | 99.0 | 9.29 | 67.880 | 0.316 | 1.410 | 0.693 | 0.359 | 5.747 | 2.474 | 0.980 | 0.019 | 0.0062 | 0.0008 |

### Where the time goes (share of total wall)

| variant | cache | download+load % | compute % | io % | other/untimed % |
|---|---|---|---|---|---|
| batch=1, days=48, members=1, weekly=1 | cold | 87 | 8 | 2 | 1 |
| batch=4, days=48, members=4, weekly=1 | cold | 77 | 10 | 8 | 2 |
