"""Benchmark 1: medium-range deterministic forecast with FCN3.

FCN3 initialized from GFS, `--steps` x 6 h (default 40 = 10 days), written to a
local Zarr store. The deterministic loop is written out explicitly (not
`earth2studio.run.deterministic`) so each phase is timed separately:

    ic_fetch        GFS fetch + interpolation to the model grid (or load of the
                    cached IC tensor with --local-ic)
    model_load      FCN3.load_model(load_default_package()) + .to(device)
    inference_step  one autoregressive step (CUDA-synchronized), per step
    io_write        backend.write() of one step, per step
    io_close        AsyncZarrBackend.close() drain at the end (async_zarr only)

Variant axes: --io {zarr,async_zarr}  --dtype {fp32,bf16}
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from common.harness import BenchRecorder, cache_size_bytes, pick_recent_init_time, setup_cache

BENCHMARK = "01_deterministic_fcn3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--io", choices=["zarr", "async_zarr"], default="zarr")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument(
        "--cache", choices=["cold", "warm"], default="cold",
        help="cold: fresh empty EARTH2STUDIO_CACHE so weights and GFS cache download from scratch (default)",
    )
    p.add_argument("--steps", type=int, default=40, help="6 h forecast steps (40 = 10 days)")
    p.add_argument("--time", type=str, default=None, help="IC time ISO (default: pick_recent_init_time(48))")
    p.add_argument(
        "--local-ic", dest="local_ic", action=argparse.BooleanOptionalAction, default=True,
        help="Cache the fetched GFS IC tensor under $BENCH_SCRATCH/01 and reuse it (default on)",
    )
    p.add_argument("--out-root", default=os.path.join(os.environ.get("BENCH_SCRATCH", "/tmp"), "01"))
    p.add_argument("--pool-size", type=int, default=8, help="AsyncZarrBackend pool size")
    p.add_argument("--tag", default="", help="extra variant tag for the results slug")
    return p.parse_args()


# Files makani's load_model_package reads for the FCN3 checkpoint (see the
# cached package dir); resolved up front so model_download is a separate phase.
FCN3_PACKAGE_FILES = [
    "config.json",
    "training_checkpoints/best_ckpt_mp0.tar",
    "global_means.npy",
    "global_stds.npy",
    "maxs.npy",
    "mins.npy",
    "land_mask.nc",
    "orography.nc",
]


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    args = parse_args()
    cache_path = setup_cache(args.cache, BENCHMARK)  # before any model/data class is used
    device = torch.device("cuda")
    ic_time = np.datetime64(args.time) if args.time else np.datetime64(pick_recent_init_time(hours_back=48))
    variant = {"io": args.io, "dtype": args.dtype, "steps": args.steps, "cache": args.cache}
    if args.tag:
        variant["tag"] = args.tag
    rec = BenchRecorder(BENCHMARK, variant)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    store_path = out_root / f"fcn3_{args.io}_{args.dtype}_{args.steps}steps.zarr"
    if store_path.exists():
        import shutil
        shutil.rmtree(store_path)

    # ----------------------------------------------------------- model load
    from earth2studio.models.px import FCN3

    # Package files are downloaded lazily on resolve(); resolve everything the
    # checkpoint loader reads so the download is a separate phase.
    with rec.phase("model_download"):
        package = FCN3.load_default_package()
        for f in FCN3_PACKAGE_FILES:
            package.resolve(f)
    with rec.phase("model_load"):
        model = FCN3.load_model(package).to(device)
    ic = model.input_coords()

    # ------------------------------------------------------------- IC fetch
    ic_cache = out_root / f"ic_{str(ic_time).replace(':', '')}.pt"
    with rec.phase("ic_fetch"):
        if args.cache == "warm" and args.local_ic and ic_cache.exists():
            blob = torch.load(ic_cache, weights_only=False)
            x, coords = blob["x"].to(device), blob["coords"]
            rec.extra["ic_source"] = "local_cache"
        else:
            from earth2studio.data import GFS, fetch_data
            from earth2studio.utils.coords import map_coords

            x, coords = fetch_data(
                source=GFS(),
                time=np.array([ic_time]),
                variable=ic["variable"],
                lead_time=ic["lead_time"],
                device=device,
            )
            x, coords = map_coords(x, coords, ic)
            rec.extra["ic_source"] = "GFS"
            if args.local_ic:
                torch.save({"x": x.cpu(), "coords": coords}, ic_cache)

    # ----------------------------------------------------------- IO setup
    total_coords = model.output_coords(ic).copy()
    for k, v in list(total_coords.items()):
        if v.shape == (0,):
            del total_coords[k]
    dt = model.output_coords(ic)["lead_time"]
    total_coords["time"] = np.array([ic_time])
    total_coords["lead_time"] = np.asarray([dt * i for i in range(args.steps + 1)]).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)
    var_names = total_coords.pop("variable")

    with rec.phase("io_setup"):
        if args.io == "zarr":
            from earth2studio.io import ZarrBackend

            io = ZarrBackend(file_name=str(store_path), backend_kwargs={"overwrite": True})
        else:
            from earth2studio.io import AsyncZarrBackend

            io = AsyncZarrBackend(
                file_name=str(store_path),
                parallel_coords=OrderedDict(
                    {"time": total_coords["time"], "lead_time": total_coords["lead_time"]}
                ),
                blocking=False,
                pool_size=args.pool_size,
            )
        io.add_array(total_coords, list(var_names))

    # ------------------------------------------------------------ rollout
    from earth2studio.utils.coords import split_coords

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bf16"
        else torch.autocast(device_type="cuda", enabled=False)
    )
    with torch.inference_mode():
        it = model.create_iterator(x, coords)
        # step 0 yields the IC without running the model; don't count as inference
        x0, c0 = next(it)
        with rec.phase("io_write"):
            io.write(*split_coords(x0, c0))
        for step in range(1, args.steps + 1):
            with rec.phase("inference_step"):
                with autocast:
                    xs, cs = next(it)
            with rec.phase("io_write"):
                io.write(*split_coords(xs, cs))

    with rec.phase("io_close"):
        if hasattr(io, "close"):
            io.close()

    steady = rec.phases["inference_step"][1:] or rec.phases["inference_step"]
    rec.finish(
        extra={
            "steps": args.steps,
            "ic_time": str(ic_time),
            "steps_per_s": round(len(steady) / sum(steady), 4),
            "io_write_total_s": round(sum(rec.phases["io_write"]), 3),
            "output_bytes": dir_bytes(store_path),
            "n_variables": int(len(var_names)),
            "store": str(store_path),
            "cache_mode": args.cache,
            "cache_path": cache_path,
            "cache_bytes": cache_size_bytes(cache_path),
        }
    )
    rec.print_summary()
    rec.write()


if __name__ == "__main__":
    main()
