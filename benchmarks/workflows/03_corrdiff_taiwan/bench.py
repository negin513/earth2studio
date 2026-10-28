"""Benchmark 3: CorrDiff Taiwan regional downscaling driven by GFS.

Sweeps (number_of_samples x number_of_steps). Default is a COLD cache: every sweep
point gets a fresh empty EARTH2STUDIO_CACHE, so the model package and the GFS
forcing are downloaded from scratch for each point. With --cache warm the model is
loaded and the forcing fetched once and reused across points (number_of_samples /
number_of_steps are plain attributes on the model). Each sweep point gets its own
BenchRecorder / JSON. Phases:

    model_download  Package.resolve of corrdiff_inference_package.zip (network; meaningful when cold)
    model_load      CorrDiffTaiwan.load_model (unzip + checkpoint load + to(device))
    ic_fetch        GFS fetch for all times
    downscale_step  one call of the model on one time (per time)
    io_write        ZarrBackend.write of that time's samples (per time)

extra.seconds_per_sample = steady-state (first time dropped) downscale_step / samples.

Forcing can instead come from benchmark 1's FCN3 Zarr via --forcing-zarr; see README.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from common.harness import BenchRecorder, cache_size_bytes, pick_recent_init_time, setup_cache

BENCH = "03_corrdiff_taiwan"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--times", type=int, default=4, help="number of consecutive 6 h forcing times")
    p.add_argument("--end-time", type=str, default=None, help="last forcing time, ISO UTC; default pick_recent_init_time(48h)")
    p.add_argument("--samples", type=str, default="1", help="comma list of number_of_samples, e.g. 1,4,8")
    p.add_argument("--diffusion-steps", type=str, default="8", help="comma list of number_of_steps, e.g. 8,18,36")
    p.add_argument("--solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forcing-zarr", type=str, default=None, help="read forcing from benchmark 1's FCN3 Zarr instead of GFS")
    p.add_argument("--forcing", choices=["gfs", "fcn3_zarr"], default=None, help="label only; inferred from --forcing-zarr")
    p.add_argument("--out-dir", type=str, default=None, help="Zarr output dir; default $BENCH_SCRATCH/03")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cache", choices=["cold", "warm"], default="cold",
                   help="cold: fresh empty EARTH2STUDIO_CACHE per sweep point (default); warm: reuse existing cache")
    return p.parse_args()


def forcing_times(args: argparse.Namespace) -> np.ndarray:
    if args.end_time:
        end = np.datetime64(args.end_time)
    else:
        end = np.datetime64(pick_recent_init_time(hours_back=48))
    return np.array([end - np.timedelta64(6 * (args.times - 1 - i), "h") for i in range(args.times)])


def fetch_gfs(times: np.ndarray, variables: np.ndarray, device: str):
    from earth2studio.data import GFS, fetch_data

    src = GFS()
    x, coords = fetch_data(src, times, variables, device="cpu")
    # fetch_data returns [time, lead_time, variable, lat, lon]; drop lead_time
    x = x[:, 0]
    coords = OrderedDict((k, v) for k, v in coords.items() if k != "lead_time")
    return x.to(device), coords


def fetch_fcn3_zarr(path: str, n_times: int, variables: np.ndarray, device: str):
    """Forcing from benchmark 1's FCN3 Zarr (ZarrBackend layout: one array per
    variable with dims [time, lead_time, lat, lon]). Uses init 0 and the first
    n_times lead times, so the forcing is a forecast, labelled by valid time."""
    import xarray as xr

    if not Path(path).exists():
        sys.exit(f"[bench03] --forcing-zarr path does not exist: {path}\n"
                 "Run benchmarks/workflows/01_deterministic_fcn3 first, or point at its output .zarr.")
    ds = xr.open_zarr(path, consolidated=False)
    missing = [v for v in variables if v not in ds]
    if missing:
        sys.exit(f"[bench03] forcing Zarr {path} lacks variables {missing}; have {list(ds.data_vars)[:20]}...")
    if "lead_time" not in ds.dims or ds.sizes["lead_time"] < n_times:
        sys.exit(f"[bench03] forcing Zarr needs >= {n_times} lead_times, has {ds.sizes.get('lead_time')}")
    sub = ds[list(variables)].isel(time=0, lead_time=slice(0, n_times))
    arr = sub.to_array("variable").transpose("lead_time", "variable", "lat", "lon")
    valid = (sub["time"].values + sub["lead_time"].values).astype("datetime64[ns]")
    coords = OrderedDict(time=valid, variable=np.array(variables), lat=arr["lat"].values, lon=arr["lon"].values)
    x = torch.as_tensor(np.ascontiguousarray(arr.values), dtype=torch.float32)
    if torch.isnan(x).any():
        sys.exit("[bench03] forcing Zarr has NaNs in the requested slice (benchmark 1 may still be writing it)")
    return x.to(device), coords


def main() -> None:
    args = parse_args()
    cache_path = setup_cache(args.cache, BENCH)  # must precede any model/data class use
    from earth2studio.io import ZarrBackend
    from earth2studio.models.dx import CorrDiffTaiwan
    from earth2studio.utils.coords import map_coords, split_coords

    device = args.device
    forcing = "fcn3_zarr" if args.forcing_zarr else "gfs"
    samples_list = [int(s) for s in args.samples.split(",")]
    steps_list = [int(s) for s in args.diffusion_steps.split(",")]
    out_dir = Path(args.out_dir or os.path.join(os.environ.get("BENCH_SCRATCH", "/tmp"), "03"))
    out_dir.mkdir(parents=True, exist_ok=True)
    times = forcing_times(args)

    def load_and_fetch(rec: BenchRecorder):
        with rec.phase("model_download"):
            pkg = CorrDiffTaiwan.load_default_package()
            pkg.resolve("corrdiff_inference_package.zip")
        with rec.phase("model_load"):
            model = CorrDiffTaiwan.load_model(pkg, device=device)
            model.solver = args.solver
            model.seed = args.seed
        in_coords = model.input_coords()
        with rec.phase("ic_fetch"):
            if forcing == "gfs":
                x, coords = fetch_gfs(times, in_coords["variable"], device)
            else:
                x, coords = fetch_fcn3_zarr(args.forcing_zarr, args.times, in_coords["variable"], device)
            x, coords = map_coords(x, coords, in_coords)
        print(f"[bench03] cache={args.cache} forcing={forcing} times={coords['time']} x={tuple(x.shape)} "
              f"download={rec.phases['model_download'][0]:.1f}s load={rec.phases['model_load'][0]:.1f}s "
              f"fetch={rec.phases['ic_fetch'][0]:.1f}s")
        return model, x, coords

    shared = None  # warm mode: load once, copy the shared phase timings into each point
    if args.cache == "warm":
        shared = BenchRecorder(BENCH, {"shared": True}, device=device)
        model, x, coords = load_and_fetch(shared)

    first = True
    for n_samples in samples_list:
        for n_steps in steps_list:
            variant = {"cache": args.cache, "forcing": forcing, "samples": n_samples,
                       "diffusion_steps": n_steps, "solver": args.solver}
            if args.cache == "cold":
                if not first:
                    model = x = None
                    torch.cuda.empty_cache()
                    cache_path = setup_cache("cold", BENCH)
                rec = BenchRecorder(BENCH, variant, device=device)
                model, x, coords = load_and_fetch(rec)
            else:
                rec = BenchRecorder(BENCH, variant, device=device)
                for k in ("model_download", "model_load", "ic_fetch"):
                    rec.record(k, shared.phases[k][0])
            first = False
            in_coords = model.input_coords()
            model.number_of_samples = n_samples
            model.number_of_steps = n_steps

            out_coords = model.output_coords(in_coords)
            store = out_dir / f"{rec.slug()}.zarr"
            if store.exists():
                shutil.rmtree(store)
            io = ZarrBackend(str(store), chunks={"time": 1, "sample": 1}, backend_kwargs={"overwrite": True})
            total_coords = OrderedDict(
                time=coords["time"], sample=out_coords["sample"], lat=out_coords["lat"], lon=out_coords["lon"]
            )
            io.add_array(total_coords, out_coords["variable"])

            for i in range(x.shape[0]):
                xi = x[i : i + 1]
                ci = OrderedDict(coords)
                ci["time"] = coords["time"][i : i + 1]
                # model expects [batch, variable, lat, lon]; "time" plays the batch role
                with rec.phase("downscale_step"):
                    y, cy = model(xi, ci)
                with rec.phase("io_write"):
                    io.write(*split_coords(y, cy))

            ds = rec.phases["downscale_step"]
            steady = sum(ds[1:]) / (len(ds) - 1) if len(ds) > 1 else ds[0]
            rec.finish(
                extra={
                    "samples": n_samples,
                    "diffusion_steps": n_steps,
                    "times": int(x.shape[0]),
                    "seconds_per_sample": round(steady / n_samples, 4),
                    "seconds_per_sample_step": round(steady / n_samples / n_steps, 5),
                    "first_call_s": round(ds[0], 4),
                    "cache_mode": args.cache,
                    "cache_path": cache_path,
                    "cache_bytes": cache_size_bytes(cache_path),
                    "zarr_store": str(store),
                    "forcing_times": [str(t) for t in coords["time"]],
                }
            )
            rec.print_summary()
            rec.write()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
