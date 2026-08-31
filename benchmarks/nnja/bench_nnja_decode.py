"""Decode-only NNJA IR-sounder benchmark: isolates decode CPU time from network fetch.

Downloads or use a pre-fetched local BUFR aggregate file and times
decode_ir_sounder directly, bypassing the async S3 fetch path. Useful for
measuring the #1059 vectorized-decode speedup independent of network
variability.

Usage:
    python bench_nnja_decode.py <path-to-local-bufr-file> [decode_workers]

Example:
    curl -o iasi.bufr_d \\
        https://noaa-reanalyses-pds.s3.amazonaws.com/observations/reanalysis/iasi/mtiasi/2019/01/bufr/gdas.20190101.t00z.mtiasi.tm00.bufr_d
    python bench_nnja_decode.py iasi.bufr_d 1
"""

import sys
import time
from datetime import datetime

from earth2studio.data.utils_ncep import decode_ir_sounder

path = sys.argv[1]
decode_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1

t0 = time.perf_counter()
df = decode_ir_sounder(
    path,
    "iasi",
    None,
    datetime(2019, 1, 1, 0, 0),
    datetime(2019, 1, 1, 6, 0),
    None,
    decode_workers=decode_workers,
)
t1 = time.perf_counter()
print(f"RESULT decode_workers={decode_workers} rows={len(df)} seconds={t1 - t0:.2f}")
