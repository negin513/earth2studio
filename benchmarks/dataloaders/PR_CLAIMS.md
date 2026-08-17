# Performance numbers reported in the PR bodies

Extracted from NVIDIA/earth2studio PR descriptions for comparison against the
measurements in RESULTS.md. Important: PR-body baselines are the
*contemporary main at PR time*, while this benchmark's baseline is the 0.14.0
release — so this benchmark's "before" can be slower than a PR body's
"before" when earlier PRs in the series had already landed on main.

## Merged PRs

- **#855** (ARCO/WB2 zarr chunk dedup) — GraphCast init set, 82 vars, 10 init
  times, cold, avg of 3: ARCO 195.4s → 50.6s (3.9x, 910 → 118 remote reads);
  WB2 82.9s → 35.5s (2.3x, 906 → 114 reads).
- **#913** (UFS obs → obstore) — no table in body (image only); prose: s3fs is
  GIL-bound, "caps concurrent S3 reads at ~36 MB/s regardless of concurrency".
- **#955** (ARCO/WB2/rx → obstore) — ARCO cold cached fetch ~20% faster
  (7.3/5.7s → 5.8/4.5s); warm parity (disk-bound); 512 concurrent 64 KB range
  GETs: 450–470 → 990–1190 req/s (~2.4x); store open ~10% faster.
- **#964** (GFS grib byte-range → obstore) — no before/after table; smoke test
  warm-cache fetch ~0.03s. Defaults: async_workers=16, retries=3.
- **#971** (NNJA decode off IO loop) — no perf numbers (CI incident fix).
- **#972** (HRRR/GEFS/CFS/NCAR → obstore) — cold, full vocab, sequential:
  GEFS_FX 214 vars 8.2s → 2.1s (4.0x); CFS_FX 81 vars 4.2s → 1.6s (2.6x);
  NCAR_ERA5 278 vars 289s → 152s (1.9x); HRRR 648 vars 22.3s → 13.3s (1.7x).
- **#1042** (GOES/GLM → obstore) — GOES 12 CONUS times x 16 bands 20.4s →
  13.9s (~30%); GLM six 5-min bins 24.8s → 6.1s (~4x). "s3fs-to-obstore is
  approximately performance-neutral for GOES (bandwidth-bound whole files)".
- **#1043** (Himawari → obstore) — 2 channels/176 tiles 9.5s → 5.2s (~1.8x);
  4 sequential calls 28.3s → 12.3s (~2.3x).
- **#1058** (ISD/IBTrACS/CFS-reforecast/OPERA) — cold, full vocab, 3 reps:
  ISD ~5.3s → ~3.0s (1.8x); IBTrACS ~6.1s → ~3.8s (1.6x); CFS_Reforecast_FX
  81 vars ~102s → ~5s (~20x, pygrib.select rescan removal); Flux ~4.6s → 1.5s
  (~3x); OPERA ~3.0s → ~1.7s (1.8x).
- **#1061** (MRMS/NClimGrid) — MRMS 2 vars x 2 times 31.4–36.8s → 4.6–5.4s
  (~6.4x; win mostly from header-key axis computation replacing ~8s
  `latlons()` + threaded decode); NClimGridDaily 3 days ~1.2x median.

## Open PRs

- **#914** (UFS parallel decode) — no numbers in body (image only).
- **#1059** (NNJA IR decode vectorization) — decode throughput IASI 7.8x,
  CrIS 5.8x (ch-rows/s); peak decode memory 314 MB → 3 MB (105x).
- **#1062** (JPSS → obstore) — cold medians vs main: VIIRS 2t x 2 M vars
  3.7s → 2.4s (1.6x); ATMS 2.8s → 0.9s (3.1x); VIIRS full 16-var M-band
  ~6.4s → ~7.1s (~parity, granule downloads dominate); CrIS 3ch ~8.2s → 4.4s
  (1.9x); CrIS all 2211 ch ~7.0s → 4.7s (~1.5x).
- **#1063** (GHCN by_station) — GHCNDaily 55–60s → 2.0–2.4s (~26x, from the
  by_station redesign; obstore migration alone was parity); GHCNHourly parity.
  (No 0.14.0 baseline — GHCN added after 0.14.0; not in this benchmark.)
- **#1065** (Planetary Computer → obstore + IFS grib byte-range) — IFS 5 vars
  33.3s (117.6 MB) → 5.5s (3 MB) = 6x; scaling: 1 var ~240x, 3 vars ~115x,
  8 vars ~38x, 16 vars ~23x, 32 vars ~9x, 64 vars ~3.4x, 100 vars ~1.8x,
  160 vars parity. OISST/Sentinel3AOD/MODISFire/PC-GOES: parity
  (bandwidth-bound whole files).
