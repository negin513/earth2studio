# Workflow benchmark results

## 01_deterministic_fcn3

GPU: NVIDIA GB200  torch 2.13.0+cu130  earth2studio 61b3e1dc

| cache | dtype | io | source | steps | total_s | peak_GiB | ic_fetch (mean s) | inference_step (mean s) | io_close (mean s) | io_setup (mean s) | io_write (mean s) | model_download (mean s) | model_load (mean s) | cache_bytes | io_write_total_s | n_variables | output_bytes | steps | steps_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | bf16 | async_zarr |  | 40 | 220.86 | 49.023 | 4.439 (n=1) | 0.754 (n=40) | 0.478 (n=1) | 1.788 (n=1) | 0.185 (n=41) | 13.437 (n=1) | 119.375 (n=1) | 2.906e+09 | 7.517 | 72 | 1.226e+10 | 40 | 1.326 |
| cold | bf16 | zarr |  | 40 | 284.686 | 49.023 | 7.409 (n=1) | 1.587 (n=40) | 0.000 (n=1) | 1.580 (n=1) | 0.848 (n=41) | 14.285 (n=1) | 120.450 (n=1) | 2.906e+09 | 34.92 | 72 | 1.226e+10 | 40 | 0.6299 |
| cold | fp32 | async_zarr | arco | 40 | 245.636 | 49.023 | 15.389 (n=1) | 0.753 (n=40) | 0.331 (n=1) | 2.149 (n=1) | 0.125 (n=41) | 15.382 (n=1) | 117.546 (n=1) | 3.376e+09 | 5.096 | 72 | 1.226e+10 | 40 | 1.328 |
| cold | fp32 | async_zarr |  | 40 | 218.219 | 49.023 | 4.494 (n=1) | 0.754 (n=40) | 0.140 (n=1) | 2.174 (n=1) | 0.123 (n=41) | 13.303 (n=1) | 119.326 (n=1) | 2.906e+09 | 5.095 | 72 | 1.226e+10 | 40 | 1.326 |
| cold | fp32 | zarr |  | 40 | 244.041 | 49.023 | 4.266 (n=1) | 0.749 (n=40) | 0.000 (n=1) | 1.491 (n=1) | 0.817 (n=41) | 16.778 (n=1) | 118.246 (n=1) | 2.906e+09 | 33.51 | 72 | 1.226e+10 | 40 | 1.334 |
| warm | fp32 | async_zarr | arco | 2 | 174.497 | 49.023 | 16.159 (n=1) | 3.046 (n=2) | 0.135 (n=1) | 1.822 (n=1) | 0.095 (n=3) | 0.009 (n=1) | 119.073 (n=1) | 3.738e+10 | 0.291 | 72 | 8.971e+08 | 2 | 1.325 |

## 02_ensemble_hens_lite

GPU: NVIDIA GB200  torch 2.13.0+cu130  earth2studio 61b3e1dc

| batch_size | members | perturbation | total_s | peak_GiB | diagnostic_step (mean s) | ic_fetch (mean s) | inference_step (mean s) | io_write (mean s) | model_download (mean s) | model_load (mean s) | perturb (mean s) | batch_size | bv_integration_steps | cache_bytes | member_steps_per_s | member_steps_per_s_px_only | members | steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 16 | bred_vector | 685.083 | 41.902 | 0.663 (n=21) | 5.638 (n=1) | 0.875 (n=20) | 0.037 (n=21) | 582.656 (n=1) | 15.967 (n=1) | 18.684 (n=1) | 16 | 3 | 7.453e+09 | 10.17 | 18.27 | 16 | 20 |
| 16 | 16 | spherical_gaussian | 525.503 | 30.796 | 0.602 (n=21) | 10.594 (n=1) | 0.868 (n=20) | 0.036 (n=21) | 438.586 (n=1) | 12.090 (n=1) | 1.155 (n=1) | 16 |  | 7.453e+09 | 10.65 | 18.43 | 16 | 20 |
| 16 | 32 | bred_vector | 818.834 | 51.247 | 0.625 (n=42) | 7.748 (n=1) | 0.869 (n=40) | 0.041 (n=42) | 655.144 (n=1) | 11.896 (n=1) | 5.890 (n=2) | 16 | 3 | 7.453e+09 | 10.48 | 18.41 | 32 | 20 |
| 16 | 32 | spherical_gaussian | 571.536 | 40.145 | 0.627 (n=42) | 9.078 (n=1) | 0.871 (n=40) | 0.038 (n=42) | 452.075 (n=1) | 12.562 (n=1) | 1.138 (n=2) | 16 |  | 7.453e+09 | 10.45 | 18.36 | 32 | 20 |
| 2 | 2 | spherical_gaussian | 441.601 | 16.529 | 0.083 (n=3) | 5.064 (n=1) | 6.085 (n=2) | 0.015 (n=3) | 364.211 (n=1) | 13.255 (n=1) | 1.348 (n=1) | 2 |  | 7.453e+09 | 7.074 | 17.12 | 2 | 2 |
| 8 | 16 | bred_vector | 580.217 | 33.021 | 0.310 (n=42) | 20.496 (n=1) | 0.441 (n=40) | 0.023 (n=42) | 481.236 (n=1) | 14.727 (n=1) | 3.967 (n=2) | 8 | 3 | 7.453e+09 | 10.42 | 18.13 | 16 | 20 |
| 8 | 16 | spherical_gaussian | 518.651 | 21.929 | 0.351 (n=42) | 6.274 (n=1) | 0.443 (n=40) | 0.023 (n=42) | 429.769 (n=1) | 13.229 (n=1) | 0.994 (n=2) | 8 |  | 7.453e+09 | 9.851 | 18.07 | 16 | 20 |
| 8 | 32 | bred_vector | 726.513 | 33.021 | 0.306 (n=84) | 12.844 (n=1) | 0.439 (n=80) | 0.023 (n=84) | 574.044 (n=1) | 13.695 (n=1) | 1.761 (n=4) | 8 | 3 | 7.453e+09 | 10.51 | 18.21 | 32 | 20 |
| 8 | 32 | spherical_gaussian | 558.959 | 21.929 | 0.339 (n=84) | 6.649 (n=1) | 0.440 (n=80) | 0.029 (n=84) | 436.336 (n=1) | 12.657 (n=1) | 0.979 (n=4) | 8 |  | 7.453e+09 | 10.05 | 18.18 | 32 | 20 |

## 03_corrdiff_taiwan

GPU: NVIDIA GB200  torch 2.13.0+cu130  earth2studio 0952d61f

| cache | diffusion_steps | forcing | samples | solver | total_s | peak_GiB | downscale_step (mean s) | ic_fetch (mean s) | io_write (mean s) | model_download (mean s) | model_load (mean s) | cache_bytes | diffusion_steps | first_call_s | samples | seconds_per_sample | seconds_per_sample_step | times |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | 18 | gfs | 1 | euler | 69.537 | 1.217 | 0.962 (n=4) | 14.095 (n=1) | 0.048 (n=4) | 45.157 (n=1) | 5.753 (n=1) | 1.557e+09 | 18 | 0.9946 | 1 | 0.9623 | 0.05346 | 4 |
| cold | 18 | gfs | 4 | euler | 62.047 | 3.757 | 2.171 (n=4) | 6.462 (n=1) | 0.076 (n=4) | 40.245 (n=1) | 5.949 (n=1) | 1.557e+09 | 18 | 2.177 | 4 | 0.5428 | 0.03015 | 4 |
| cold | 18 | gfs | 8 | euler | 64.755 | 7.138 | 4.074 (n=4) | 6.139 (n=1) | 0.086 (n=4) | 35.811 (n=1) | 5.677 (n=1) | 1.557e+09 | 18 | 4.122 | 8 | 0.5093 | 0.02829 | 4 |
| cold | 36 | gfs | 1 | euler | 73.699 | 1.217 | 1.774 (n=4) | 7.329 (n=1) | 0.046 (n=4) | 52.990 (n=1) | 5.721 (n=1) | 1.557e+09 | 36 | 1.801 | 1 | 1.774 | 0.04929 | 4 |
| cold | 36 | gfs | 4 | euler | 66.668 | 3.757 | 4.294 (n=4) | 4.151 (n=1) | 0.065 (n=4) | 38.779 (n=1) | 5.642 (n=1) | 1.557e+09 | 36 | 4.305 | 4 | 1.074 | 0.02982 | 4 |
| cold | 36 | gfs | 8 | euler | 80.503 | 7.138 | 8.092 (n=4) | 3.832 (n=1) | 0.089 (n=4) | 37.848 (n=1) | 5.742 (n=1) | 1.557e+09 | 36 | 8.095 | 8 | 1.012 | 0.0281 | 4 |
| cold | 8 | gfs | 1 | euler | 67.114 | 1.218 | 0.441 (n=4) | 12.113 (n=1) | 0.044 (n=4) | 46.270 (n=1) | 5.786 (n=1) | 1.557e+09 | 8 | 0.9528 | 1 | 0.4414 | 0.05518 | 4 |
| cold | 8 | gfs | 4 | euler | 67.235 | 3.757 | 0.996 (n=4) | 6.829 (n=1) | 0.066 (n=4) | 50.041 (n=1) | 5.693 (n=1) | 1.557e+09 | 8 | 1.031 | 4 | 0.2489 | 0.03112 | 4 |
| cold | 8 | gfs | 8 | euler | 56.598 | 7.137 | 1.841 (n=4) | 3.753 (n=1) | 0.096 (n=4) | 38.761 (n=1) | 5.745 (n=1) | 1.557e+09 | 8 | 1.878 | 8 | 0.2301 | 0.02876 | 4 |

## 04_stormscope_goes

GPU: NVIDIA GB200  torch 2.12.1+cu130  earth2studio c00d096a

| cache | commit | compile | io | model | total_s | peak_GiB | compile_warmup (mean s) | composite (mean s) | ic_fetch (mean s) | inference_step (mean s) | io_commit (mean s) | io_setup (mean s) | io_write (mean s) | model_download (mean s) | model_load (mean s) | cache_bytes | end_to_end_s | ic_bytes_gpu | ic_frames | io_overhead_frac_of_rollout | io_overhead_s | rollout_s | step_seconds_first | step_seconds_steady | steps | store_bytes | time_to_first_frame_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | n/a | 0 | zarr | 3km_10min_flash | 96.585 | 9.616 |  | 0.001 (n=13) | 35.166 (n=1) | 1.073 (n=12) |  | 0.167 (n=1) | 0.112 (n=13) | 13.188 (n=1) | 6.427 (n=1) | 2.5e+09 | 49.88 | 7.2e+08 | 6 | 0.1019 | 1.499 | 14.71 | 1.386 | 1.073 | 12 | 6.07e+08 | 36.84 |
| cold | per_init | 0 | icechunk | 3km_10min | 651.065 | 9.438 |  | 0.001 (n=13) | 41.825 (n=1) | 34.777 (n=12) | 0.165 (n=1) | 0.086 (n=1) | 0.037 (n=13) | 155.891 (n=1) | 11.137 (n=1) | 3.506e+09 | 456.7 | 7.2e+08 | 6 | 0.0016 | 0.675 | 414.9 | 31.61 | 34.78 | 12 | 6.065e+08 | 73.53 |
| cold | per_init | 0 | icechunk | 3km_10min_flash | 162.701 | 9.616 |  | 0.001 (n=13) | 102.543 (n=1) | 1.076 (n=12) | 0.088 (n=1) | 0.083 (n=1) | 0.059 (n=13) | 14.617 (n=1) | 6.327 (n=1) | 2.5e+09 | 116.6 | 7.2e+08 | 6 | 0.0605 | 0.849 | 14.04 | 1.329 | 1.076 | 12 | 6.069e+08 | 104 |
| cold | per_init | 1 | icechunk | 3km_10min_flash | 144.226 | 8.085 | 24.929 (n=1) | 0.001 (n=13) | 61.544 (n=1) | 0.493 (n=12) | 0.088 (n=1) | 0.089 (n=1) | 0.152 (n=13) | 15.901 (n=1) | 6.345 (n=1) | 2.5e+09 | 70.6 | 7.2e+08 | 6 | 0.2162 | 1.958 | 9.054 | 1.665 | 0.493 | 12 | 6.069e+08 | 63.48 |
| cold | per_step | 0 | icechunk | 3km_10min_flash | 156.589 | 9.616 |  | 0.001 (n=13) | 97.476 (n=1) | 1.072 (n=12) | 0.071 (n=14) | 0.117 (n=1) | 0.101 (n=13) | 12.712 (n=1) | 6.351 (n=1) | 2.5e+09 | 112.8 | 7.2e+08 | 6 | 0.1454 | 2.23 | 15.34 | 1.324 | 1.072 | 12 | 6.071e+08 | 99.04 |
| warm | per_step | 0 | icechunk | 3km_10min_flash | 40.714 | 9.616 |  | 0.001 (n=3) | 4.097 (n=1) | 2.110 (n=2) | 0.097 (n=4) | 0.088 (n=1) | 0.021 (n=3) | 0.013 (n=1) | 4.842 (n=1) | 5.173e+10 | 8.723 | 7.2e+08 | 6 | 0.0931 | 0.431 | 4.626 | 3.147 | 1.073 | 2 | 1.218e+08 | 7.553 |

## 05_s2s_long_rollout

GPU: NVIDIA GB200  torch 2.13.0+cu130  earth2studio 4070f04f

| batch | cache | days | members | weekly | total_s | peak_GiB | ic_fetch (mean s) | inference_step (mean s) | input_prep (mean s) | io_write (mean s) | io_write_weekly (mean s) | model_download (mean s) | model_load (mean s) | model_load_latlon_wrapper (mean s) | regrid (mean s) | select (mean s) | weekly_stats (mean s) | batch_size | cache_bytes | days | forecast_days_per_gpu_minute | gpu_index | gpu_shared | inference_s_per_step_steady | leads_per_step | member_days_per_gpu_minute | members | model_step_hours | nan_detected | output_bytes | pipeline_s_per_step_steady | steps | t2m_final_maxabs | weekly_output_bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cold | 48 | 1 | 1 | 92.383 | 2.756 | 71.776 (n=1) | 0.245 (n=12) | 0.376 (n=1) | 0.162 (n=12) | 0.211 (n=1) | 5.867 (n=1) | 2.665 (n=1) | 0.966 (n=1) | 0.005 (n=12) | 0.002 (n=12) | 0.001 (n=12) | 1 | 1.729e+09 | 48 | 578.9 | 1 | 0 | 0.2449 | 16 | 578.9 | 1 | 96 | 0 | 3.189e+09 | 0.4146 | 12 | 324.1 | 9.969e+07 |
| 4 | cold | 48 | 4 | 1 | 98.953 | 9.294 | 67.880 (n=1) | 0.316 (n=12) | 1.410 (n=1) | 0.693 (n=12) | 0.359 (n=1) | 5.747 (n=1) | 2.474 (n=1) | 0.980 (n=1) | 0.019 (n=12) | 0.006 (n=12) | 0.001 (n=12) | 4 | 1.729e+09 | 48 | 927.9 | 1 | 0 | 0.3162 | 16 | 927.9 | 4 | 96 | 0 | 1.276e+10 | 1.035 | 12 | 322.5 | 3.987e+08 |
