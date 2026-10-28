"""Benchmark 5: subseasonal (S2S) long rollout with DLESyM.

DLESyM (DLESyM-V1-ERA5, checkpoint v1.1) is a coupled atmosphere/ocean model on a
HEALPix nside=64 grid. One model step advances 96 h and emits 16 six-hourly
atmosphere leads (ocean every 48 h). Initial conditions are nine 6-hourly ERA5
states ([-48 h .. 0 h]) at 0.25 deg from ARCO ERA5 (public, no credentials).

The loop below uses `DLESyMLatLon` for input preparation (derived variables,
lat-lon -> HEALPix regrid) and then drives the HEALPix-native generator
(`DLESyM._default_generator`) itself, so the HEALPix -> lat-lon regrid of each
step's output is timed as its own `regrid` phase. Output is written on the
0.25 deg lat-lon grid (721 x 1440) for a reduced variable set, chunked along
lead_time (16 leads = one model step per chunk).

Ensemble members use the model's conditional-layer-norm noise (`use_cln`), one
member per batch row, seeded with `set_rng`.

Phases: ic_fetch, model_download, model_load, inference_step (per step),
regrid (per step), weekly_stats (per step), io_write (per step), io_write_weekly.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

BENCHMARK = "05_s2s_long_rollout"
STEP_HOURS = 96
WEEK_HOURS = 168
N_WEEKS = 6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=45, help="forecast length; rounded up to whole 96 h model steps")
    p.add_argument("--steps", type=int, default=None, help="override number of model steps")
    p.add_argument("--members", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=None, help="members per forward pass (default: all)")
    p.add_argument("--ic-time", type=str, default="2025-06-01T00:00", help="UTC init time (ARCO ERA5 coverage)")
    p.add_argument("--variables", type=str, default="t2m,msl,z500,t850", help="variables written to zarr")
    p.add_argument("--weekly-stats", action="store_true", help="accumulate week-1..6 means on GPU and write them")
    p.add_argument("--cache", choices=["cold", "warm"], default="cold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="zarr output dir (default $BENCH_SCRATCH/05/<variant>)")
    p.add_argument("--version", default="v1.1", choices=["v1.0", "v1.1"])
    return p.parse_args()


def gpu_info() -> tuple[int | None, bool]:
    """Return (physical GPU index in use, whether another process already uses it)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    idx = int(vis.split(",")[0]) if vis else 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
        )
        uuids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"], text=True
        )
        uuid = {int(l.split(",")[0]): l.split(",")[1].strip() for l in uuids.strip().splitlines()}[idx]
        others = [l for l in out.strip().splitlines() if l.startswith(uuid) and int(l.split(",")[1]) != os.getpid()]
        return idx, len(others) > 0
    except Exception:
        return idx, False


def main() -> None:
    args = parse_args()

    # --- cache mode must be set before any Package / DataSource is touched -----
    from common.harness import BenchRecorder, cache_size_bytes, setup_cache

    cache_path = setup_cache(args.cache, BENCHMARK)

    import numpy as np
    import torch

    from earth2studio.data import ARCO_ERA5
    from earth2studio.data.utils import fetch_data
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px.dlesym import DLESyM, DLESyMLatLon
    from earth2studio.utils.coords import map_coords, split_coords

    device = torch.device("cuda")
    steps = args.steps or math.ceil(args.days * 24 / STEP_HOURS)
    days = steps * STEP_HOURS // 24
    members = args.members
    batch_size = args.batch_size or members
    variables = args.variables.split(",")
    variant = {
        "members": members,
        "batch": batch_size,
        "days": days,
        "cache": args.cache,
        "weekly": int(args.weekly_stats),
    }
    rec = BenchRecorder(BENCHMARK, variant)
    gpu_idx, gpu_shared = gpu_info()
    scratch = Path(os.environ.get("BENCH_SCRATCH", "/tmp/e2s_bench")) / "05"
    out_dir = Path(args.out) if args.out else scratch / rec.slug()
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / "forecast.zarr"
    weekly_path = out_dir / "weekly_means.zarr"
    for p in (zarr_path, weekly_path):
        if p.exists():
            import shutil

            shutil.rmtree(p)

    # --- model download (package resolve) vs load -------------------------------
    with rec.phase("model_download", sync=False):
        from omegaconf import OmegaConf

        pkg = DLESyMLatLon.load_default_package()
        top_cfg = OmegaConf.load(Path(pkg.resolve("config.yaml")))
        vpath = top_cfg.versions[args.version].path
        cfg = OmegaConf.load(Path(pkg.resolve(f"{vpath}/config.yaml")))
        pkg.resolve(f"{vpath}/atmos/{cfg.models.atmos_model_checkpoints[0]}")
        pkg.resolve(f"{vpath}/ocean/{cfg.models.ocean_model_checkpoints[0]}")
        for name in ("hpx_lat", "hpx_lon"):
            pkg.resolve(f"{name}.npy")
        for const in list(cfg.data.get("atmos_constants", [])) + list(cfg.data.get("ocean_constants", [])):
            try:
                pkg.resolve(f"{const}.npy")
            except Exception:
                pass  # load_model resolves whatever it actually needs

    with rec.phase("model_load"):
        # HEALPix-native model drives the rollout (so regrid is timed separately);
        # the lat-lon wrapper supplies input prep and the HPX<->lat-lon regridders.
        model_hpx = DLESyM.load_model(pkg, version=args.version).to(device)
        model_hpx.set_rng(args.seed)
    with rec.phase("model_load_latlon_wrapper"):
        model = DLESyMLatLon.load_model(pkg, version=args.version).to(device)
    in_coords = model.input_coords()

    # --- initial condition ------------------------------------------------------
    ic_time = datetime.fromisoformat(args.ic_time)
    with rec.phase("ic_fetch", sync=False):
        source = ARCO_ERA5(verbose=False)
        x, coords = fetch_data(
            source=source,
            time=np.array([np.datetime64(ic_time)]),
            variable=in_coords["variable"],
            lead_time=in_coords["lead_time"],
            device=device,
        )
    # Add the batch (ensemble) dimension: identical IC per member, CLN noise differs.
    x = x.unsqueeze(0)
    coords = OrderedDict({"batch": np.arange(1), **coords})

    # --- output store -----------------------------------------------------------
    out_coords_ll = model.output_coords(coords)  # lat-lon, 16 leads of the first step
    step_leads = out_coords_ll["lead_time"]
    all_leads = np.concatenate([step_leads + step_leads[-1] * i for i in range(steps)])
    total_coords = OrderedDict(
        {
            "ensemble": np.arange(members),
            "time": coords["time"],
            "lead_time": all_leads,
            "lat": out_coords_ll["lat"],
            "lon": out_coords_ll["lon"],
        }
    )
    io = ZarrBackend(
        str(zarr_path),
        chunks={"ensemble": 1, "time": 1, "lead_time": len(step_leads)},
        backend_kwargs={"overwrite": True},
    )
    io.add_array(total_coords, variables)

    # weekly accumulators on GPU: (members, N_WEEKS, nvar, lat, lon)
    if args.weekly_stats:
        wk_sum = torch.zeros(
            (members, N_WEEKS, len(variables), len(total_coords["lat"]), len(total_coords["lon"])),
            device=device,
            dtype=torch.float32,
        )
        wk_cnt = torch.zeros(N_WEEKS, device=device)

    # --- rollout ----------------------------------------------------------------
    nan_detected = False
    t2m_final_maxabs = 0.0
    for m0 in range(0, members, batch_size):
        mb = min(batch_size, members - m0)
        xb = x.repeat(mb, 1, 1, 1, 1, 1)
        cb = coords.copy()
        cb["batch"] = np.arange(mb)
        model_hpx.set_rng(args.seed + m0)
        # input prep (derived vars + lat-lon -> HEALPix), timed once per batch
        with rec.phase("input_prep"):
            xp, cp = model._prepare_derived_variables(xb, cb)
            x_hpx, c_hpx = model.to_hpx(xp), model.coords_to_hpx(cp)
        gen = model_hpx.create_iterator(x_hpx, c_hpx)
        next(gen)  # initial condition
        for step in range(steps):
            with rec.phase("inference_step"):
                y_hpx, yc_hpx = next(gen)
            with rec.phase("regrid"):
                y = model.to_ll(y_hpx)
                yc = model.coords_to_ll(yc_hpx)
            with rec.phase("select"):
                yc_sel = yc.copy()
                yc_sel["variable"] = np.array(variables)
                y, yc = map_coords(y, yc, yc_sel)
                if torch.isnan(y).any():
                    nan_detected = True
                if step == steps - 1:
                    ti = variables.index("t2m") if "t2m" in variables else 0
                    t2m_final_maxabs = max(t2m_final_maxabs, float(y[:, :, :, ti].abs().max()))
            if args.weekly_stats:
                with rec.phase("weekly_stats"):
                    lead_h = (yc["lead_time"] / np.timedelta64(1, "h")).astype(int)
                    for li, lh in enumerate(lead_h):
                        w = (lh - 1) // WEEK_HOURS
                        if 0 <= w < N_WEEKS:
                            wk_sum[m0 : m0 + mb, w] += y[:, 0, li]
                            if m0 == 0:
                                wk_cnt[w] += 1
            with rec.phase("io_write", sync=False):
                wc = OrderedDict(yc)
                wc["ensemble"] = np.arange(m0, m0 + mb)
                del wc["batch"]
                wc.move_to_end("ensemble", last=False)
                arrays, wcoords, names = split_coords(y, wc, dim="variable")
                io.write(arrays, wcoords, list(names))
        del gen

    if args.weekly_stats:
        with rec.phase("io_write_weekly", sync=False):
            wk_mean = wk_sum / wk_cnt.clamp(min=1).view(1, N_WEEKS, 1, 1, 1)
            wio = ZarrBackend(str(weekly_path), chunks={"ensemble": 1, "week": 1}, backend_kwargs={"overwrite": True})
            wcoords = OrderedDict(
                {
                    "ensemble": np.arange(members),
                    "week": np.arange(1, N_WEEKS + 1),
                    "variable": np.array(variables),
                    "lat": total_coords["lat"],
                    "lon": total_coords["lon"],
                }
            )
            arrays, wc2, names = split_coords(wk_mean, wcoords, dim="variable")
            wio.add_array(wc2, list(names))
            wio.write(arrays, wc2, list(names))

    # --- metrics ----------------------------------------------------------------
    def du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    ph = rec.phases
    inf = ph["inference_step"]
    steady = inf[1:] if len(inf) > 1 else inf
    gpu_s_per_step = sum(steady) / len(steady)
    # include regrid + select + io so the number reflects the delivered product
    per_step_total = gpu_s_per_step + sum(
        (sum(ph[k][1:]) / max(len(ph[k]) - 1, 1)) for k in ("regrid", "select", "io_write") if k in ph
    )
    if args.weekly_stats and "weekly_stats" in ph:
        per_step_total += sum(ph["weekly_stats"][1:]) / max(len(ph["weekly_stats"]) - 1, 1)
    n_batches = math.ceil(members / batch_size)
    days_per_step_all_members = (STEP_HOURS / 24) * members / n_batches
    rec.finish(
        extra={
            "steps": steps,
            "days": days,
            "members": members,
            "batch_size": batch_size,
            "ic_time": ic_time.isoformat(),
            "variables": variables,
            "model_step_hours": STEP_HOURS,
            "leads_per_step": int(len(step_leads)),
            "inference_s_per_step_steady": round(gpu_s_per_step, 4),
            "pipeline_s_per_step_steady": round(per_step_total, 4),
            "forecast_days_per_gpu_minute": round(days_per_step_all_members / per_step_total * 60, 3),
            "member_days_per_gpu_minute": round(days_per_step_all_members / per_step_total * 60, 3),
            "output_bytes": du(zarr_path),
            "weekly_output_bytes": du(weekly_path) if args.weekly_stats else 0,
            "output_path": str(zarr_path),
            "output_grid": "latlon 0.25deg (721x1440), regridded from HEALPix nside=64",
            "t2m_final_maxabs": round(t2m_final_maxabs, 2),
            "nan_detected": nan_detected,
            "gpu_index": gpu_idx,
            "gpu_shared": gpu_shared,
            "cache_mode": args.cache,
            "cache_path": cache_path,
            "cache_bytes": cache_size_bytes(cache_path),
            "pyoverlay": os.environ.get("PYTHONPATH", "").split(":")[0],
        }
    )
    rec.print_summary()
    rec.write()


if __name__ == "__main__":
    main()
