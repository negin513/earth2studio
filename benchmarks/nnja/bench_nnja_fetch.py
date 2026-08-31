"""End-to-end NNJAObsSat benchmark: full async fetch + decode for one sensor/cycle.

Usage:
    python bench_nnja_fetch.py [sensor] [decode_workers]

Example:
    python bench_nnja_fetch.py iasi 8
"""

import sys
import time
from datetime import datetime

from earth2studio.data import NNJAObsSat

sensor = sys.argv[1] if len(sys.argv) > 1 else "iasi"
decode_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

source = NNJAObsSat(
    cache=False,
    verbose=False,
    decode_workers=decode_workers,
)

t0 = time.perf_counter()
df = source(datetime(2019, 1, 1), sensor)
t1 = time.perf_counter()

print(
    f"RESULT sensor={sensor} decode_workers={decode_workers} "
    f"rows={len(df)} seconds={t1 - t0:.2f}"
)
