"""Benchmark 2: HENS-lite perturbed ensemble with attached diagnostic.

SFNO (73ch) initialised from GFS, perturbed with SphericalGaussian or BredVector,
rolled out N steps at 6 h with PrecipitationAFNOv2 applied on every step
(v2: its 20 inputs incl. q500/q850 are all SFNO-73ch outputs; v1 needs r500/r850,
which SFNO does not produce), output
cropped to CONUS and written to a ZarrBackend. Loop written explicitly, mirroring
earth2studio.run.ensemble / run.diagnostic so each phase can be timed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import OrderedDict
from math import ceil

import numpy as np
import torch
from loguru import logger

from common.harness import BenchRecorder, cache_size_bytes, pick_recent_init_time, setup_cache

BENCHMARK = "02_ensemble_hens_lite"

OUTPUT_VARS_PX = ["t2m", "u10m", "v10m", "msl"]  # from SFNO
OUTPUT_VARS_DX = ["tp06"]  # PrecipitationAFNOv2 outputs 6-h accumulated precip as tp06
# CONUS crop, 0.25 deg, descending lat like the model grid
CONUS_LAT = np.arange(55.0, 20.0 - 1e-6, -0.25)
CONUS_LON = np.arange(230.0, 300.0 + 1e-6, 0.25)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perturbation", choices=["spherical_gaussian", "bred_vector"], default="spherical_gaussian")
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=20, help="6 h steps; 20 = 5 days")
    p.add_argument("--time", type=str, default=None, help="IC time ISO (UTC); default pick_recent_init_time(48)")
    p.add_argument("--bv-integration-steps", type=int, default=3, help="BredVector breeding cycles (default 20 upstream, too slow for a benchmark)")
    p.add_argument("--noise-amplitude", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None, help="Zarr store path; default $BENCH_SCRATCH/02/<variant>.zarr")
    p.add_argument("--keep-output", action="store_true")
    p.add_argument("--cache", choices=["cold", "warm"], default="cold", help="cold: fresh empty EARTH2STUDIO_CACHE so model weights and data download from scratch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = setup_cache(args.cache, BENCHMARK)  # must precede any model/data class use
    device = torch.device("cuda")
    variant = {
        "perturbation": args.perturbation,
        "members": args.members,
        "batch_size": args.batch_size,
    }
    rec = BenchRecorder(BENCHMARK, variant)

    from earth2studio.data import GFS, fetch_data
    from earth2studio.io import ZarrBackend
    from earth2studio.models.dx import PrecipitationAFNOv2
    from earth2studio.models.px import SFNO
    from earth2studio.perturbation import BredVector, SphericalGaussian
    from earth2studio.utils.coords import map_coords, split_coords
    from earth2studio.utils.time import to_time_array

    # ---------------------------------------------------------------- model load
    # Separable download step: resolve every file load_model will ask for, so the
    # network time lands in model_download and model_load is deserialisation only.
    px_pkg = SFNO.load_default_package()
    dx_pkg = PrecipitationAFNOv2.load_default_package()
    with rec.phase("model_download", sync=False):
        for f in ["config.json", "global_means.npy", "global_stds.npy"]:
            px_pkg.resolve(f)
        try:
            from makani.models.model_package import LocalPackage

            px_pkg.resolve(LocalPackage.MODEL_PACKAGE_CHECKPOINT_PATH)
        except Exception as e:  # pragma: no cover - fall back to load_model download
            logger.warning(f"could not pre-resolve SFNO checkpoint: {e}")
        for f in ["afno_precip.mdlus", "global_means.npy", "global_stds.npy", "land_sea_mask.nc", "orography.nc"]:
            dx_pkg.resolve(f)
    with rec.phase("model_load"):
        px = SFNO.load_model(px_pkg).to(device)
        dx = PrecipitationAFNOv2.load_model(dx_pkg).to(device)
    px_ic = px.input_coords()
    dx_ic = dx.input_coords()

    # ------------------------------------------------------------------ IC fetch
    ic_time = (
        np.datetime64(args.time) if args.time else np.datetime64(pick_recent_init_time(48))
    )
    time_arr = to_time_array([ic_time])
    with rec.phase("ic_fetch"):
        x0, coords0 = fetch_data(
            source=GFS(),
            time=time_arr,
            variable=px_ic["variable"],
            lead_time=px_ic["lead_time"],
            device=device,
        )
    logger.success(f"IC fetched for {ic_time}")

    # --------------------------------------------------------------- output grid
    out_dir = args.out or os.path.join(
        os.environ.get("BENCH_SCRATCH", "/tmp"), "02", rec.slug() + ".zarr"
    )
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    io = ZarrBackend(out_dir)

    step_lt = px.output_coords(px_ic)["lead_time"]
    lead_times = np.asarray([step_lt * i for i in range(args.steps + 1)]).flatten()
    base_coords = OrderedDict(
        ensemble=np.arange(args.members),
        time=time_arr,
        lead_time=lead_times,
        lat=CONUS_LAT,
        lon=CONUS_LON,
    )
    io.add_array(base_coords, OUTPUT_VARS_PX + OUTPUT_VARS_DX)

    crop_px = OrderedDict(variable=np.array(OUTPUT_VARS_PX), lat=CONUS_LAT, lon=CONUS_LON)
    crop_dx = OrderedDict(variable=np.array(OUTPUT_VARS_DX), lat=CONUS_LAT, lon=CONUS_LON)

    # ------------------------------------------------------------- perturbation
    if args.perturbation == "spherical_gaussian":
        perturbation = SphericalGaussian(noise_amplitude=args.noise_amplitude)
    else:
        perturbation = BredVector(
            px,
            noise_amplitude=args.noise_amplitude,
            integration_steps=args.bv_integration_steps,
        )

    # ------------------------------------------------------------------- rollout
    batch_size = min(args.batch_size, args.members)
    nbatches = ceil(args.members / batch_size)
    logger.info(f"{args.members} members, {nbatches} batches of {batch_size}, {args.steps} steps")
    for batch_id in range(0, args.members, batch_size):
        mb = min(batch_size, args.members - batch_id)
        ens = np.arange(batch_id, batch_id + mb)
        x = x0.to(device)
        coords = OrderedDict({"ensemble": ens}) | coords0.copy()
        x = x.unsqueeze(0).repeat(mb, *([1] * x.ndim))
        x, coords = map_coords(x, coords, px_ic)
        with rec.phase("perturb"):
            x, coords = perturbation(x, coords)

        it = px.create_iterator(x, coords)
        for step in range(args.steps + 1):
            if step == 0:
                x, coords = next(it)  # IC, not timed as inference
            else:
                with rec.phase("inference_step"):
                    x, coords = next(it)
            with rec.phase("diagnostic_step"):
                xd, cd = map_coords(x, coords, dx_ic)
                xd, cd = dx(xd, cd)
            with rec.phase("io_write"):
                xp, cp = map_coords(x, coords, crop_px)
                io.write(*split_coords(xp, cp))
                xd, cd = map_coords(xd, cd, crop_dx)
                io.write(*split_coords(xd, cd))
        logger.info(f"batch {batch_id // batch_size + 1}/{nbatches} done")

    # ------------------------------------------------------------------- report
    inf = rec.phases.get("inference_step", [])
    dia = rec.phases.get("diagnostic_step", [])
    # steady state: drop the first sample of each (compile/warmup)
    inf_s = sum(inf[1:]) if len(inf) > 1 else sum(inf)
    dia_s = sum(dia[1:]) if len(dia) > 1 else sum(dia)
    n_steady = (len(inf) - 1) if len(inf) > 1 else len(inf)
    member_steps = n_steady * batch_size
    rec.finish(
        extra={
            "members": args.members,
            "batch_size": batch_size,
            "steps": args.steps,
            "ic_time": str(ic_time),
            "bv_integration_steps": args.bv_integration_steps if args.perturbation == "bred_vector" else None,
            "member_steps_per_s": round(member_steps / (inf_s + dia_s), 3) if (inf_s + dia_s) > 0 else None,
            "member_steps_per_s_px_only": round(member_steps / inf_s, 3) if inf_s > 0 else None,
            "output_vars": OUTPUT_VARS_PX + OUTPUT_VARS_DX,
            "output_shape_latlon": [len(CONUS_LAT), len(CONUS_LON)],
            "zarr_path": out_dir,
            "cache_mode": args.cache,
            "cache_path": cache_path,
            "cache_bytes": cache_size_bytes(cache_path),
        }
    )
    rec.print_summary()
    rec.write()
    if not args.keep_output:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
