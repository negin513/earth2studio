"""Benchmark 4: StormScope GOES satellite nowcast, booth-style.

Mirrors the Earthmover GTC Berlin StormScope-EU booth pipeline
(MTG/FCI -> StormScopeMeteosatEU -> composites -> Icechunk) with the US model:

    GOES-East ABI (CONUS sector, AWS, public) -> StormScopeGOES 3 km / 10 min
    -> raw predicted ABI channels + uint8 true-colour composite
    -> ZarrBackend or IceChunkBackend (local store), commit per step or per init.

Phases recorded with common.harness.BenchRecorder:
    ic_fetch        six 10-min ABI frames (model sliding window) from S3
    model_load      package resolve + checkpoint load + .to(cuda)
    compile_warmup  one untimed-for-rollout forward to trigger torch.compile (--compile)
    inference_step  one 10-min diffusion step (per step)
    composite       true-colour uint8 composite build on GPU + D2H (per step)
    io_write        backend.write of raw channels + composite (per step; for
                    icechunk this is the enqueue of a background write)
    io_commit       IceChunkBackend.commit incl. flush of in-flight writes
                    (per step with --commit-mode per_step, once with per_init)

extra: steps, step_seconds_steady, time_to_first_frame_s, end_to_end_s, ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from common.harness import BenchRecorder, cache_size_bytes, setup_cache

BENCHMARK = "04_stormscope_goes"

RGB_CHANNELS = ("abi02c", "abi03c", "abi01c")  # red 0.64, veggie 0.86, blue 0.47
RGB_DTYPE = np.uint8
RGB_FILL = 0  # 0 = missing / off-domain, data in 1..255 (booth convention)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-time", default=None, help="Init time UTC, ISO (e.g. 2026-10-28T15:00). Default: now - 3 h floored to 10 min.")
    p.add_argument("--steps", type=int, default=12, help="10-min rollout steps (12 = 2 h).")
    p.add_argument("--io", choices=["zarr", "icechunk"], default="zarr")
    p.add_argument("--commit-mode", choices=["per_step", "per_init"], default="per_init", help="icechunk only")
    p.add_argument("--compile", action="store_true", help="torch.compile the staged experts (reduce-overhead)")
    p.add_argument("--variant", default="3km_10min_flash", choices=["3km_10min_flash", "3km_10min"])
    p.add_argument("--satellite", default="goes19", help="GOES-East is goes19 since 2025-04-07")
    p.add_argument("--scan-mode", default="C", help="C (CONUS, 5 min) or F (full disk, 10 min)")
    p.add_argument("--out-dir", default=None, help="Store root; default $BENCH_SCRATCH/04/<slug>")
    p.add_argument("--keep", action="store_true", help="Keep the output store (default: delete after run)")
    p.add_argument("--blocking-io", action="store_true", help="IceChunkBackend(blocking=True)")
    p.add_argument("--tag", default=None, help="Extra variant tag recorded in results")
    p.add_argument("--cache", choices=["cold", "warm"], default="cold", help="cold: fresh empty EARTH2STUDIO_CACHE (model weights + GOES files download); warm: reuse existing cache")
    return p.parse_args()


def default_start_time() -> datetime:
    t = datetime.now(timezone.utc) - timedelta(hours=3)
    t = t.replace(minute=(t.minute // 10) * 10, second=0, microsecond=0, tzinfo=None)
    return t


def true_color(pred: torch.Tensor, variables: np.ndarray) -> torch.Tensor:
    """[C, H, W] float (reflectance factor for VIS/NIR) -> [H, W, 3] uint8.

    GOES synthetic green: 0.45 R + 0.10 NIR + 0.45 B (CIMSS recipe), gamma 1/2.2,
    clipped to [0, 1], scaled to 1..255. NaN (off-domain) -> 0.
    """
    idx = {v: i for i, v in enumerate(variables.tolist())}
    r = pred[idx["abi02c"]]
    nir = pred[idx["abi03c"]]
    b = pred[idx["abi01c"]]
    g = 0.45 * r + 0.10 * nir + 0.45 * b
    rgb = torch.stack([r, g, b], dim=-1)
    nan = torch.isnan(rgb).any(dim=-1, keepdim=True)
    rgb = torch.clamp(torch.nan_to_num(rgb, nan=0.0), 0.0, 1.0) ** (1.0 / 2.2)
    out = (rgb * 254.0 + 1.0).round().to(torch.uint8)
    return torch.where(nan, torch.zeros_like(out), out)


def main() -> None:
    args = parse_args()
    cache_path = setup_cache(args.cache, BENCHMARK)  # before any model/data class touches the cache
    device = torch.device("cuda")
    start = datetime.fromisoformat(args.start_time) if args.start_time else default_start_time()

    variant = {
        "io": args.io,
        "commit": args.commit_mode if args.io == "icechunk" else "n/a",
        "compile": int(args.compile),
        "model": args.variant,
        "cache": args.cache,
    }
    if args.tag:
        variant["tag"] = args.tag
    rec = BenchRecorder(BENCHMARK, variant)
    out_dir = Path(args.out_dir or os.path.join(os.environ.get("BENCH_SCRATCH", "/tmp"), "04", rec.slug() + "_" + rec.started.strftime("%H%M%S")))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"[bench4] init {start} steps={args.steps} variant={variant} store={out_dir}")

    from zarr.codecs import ZstdCodec

    from earth2studio.data import GOES, fetch_data
    from earth2studio.io import IceChunkBackend, ZarrBackend
    from earth2studio.models.px import StormScopeGOES

    # -------------------------------------------------- model download (HF)
    import json

    with rec.phase("model_download", sync=False):
        package = StormScopeGOES.load_default_package()
        registry = json.load(open(package.resolve("registry.json")))
        entry = registry["goes"]["models"][args.variant] if "models" in registry.get("goes", {}) else registry["goes"][args.variant]
        files = ["config.json", "lat.npy", "lon.npy"] + [c["path"] for c in entry["checkpoints"]]
        if entry.get("topo"):
            files.append("topo.npy")
        for f in files:
            try:
                package.resolve(f)
            except FileNotFoundError:
                pass
        # normalization *_means/_stds.npy are resolved inside load_model (small files)
    rec.extra["model_files"] = files

    # ------------------------------------------------------------ model load
    with rec.phase("model_load"):
        model = StormScopeGOES.load_model(
            package, model_name=args.variant, conditioning_data_source=None, amp=True, compile=args.compile
        )
        model = model.to(device).eval()
        goes_lat, goes_lon = GOES.grid(satellite=args.satellite, scan_mode=args.scan_mode)
        model.build_input_interpolator(goes_lat, goes_lon)
    in_coords = model.input_coords()
    variables = np.array(in_coords["variable"])
    n_in = len(in_coords["lead_time"])
    step_dt = model.output_times[-1]
    print(f"[bench4] model loaded: {n_in} input frames, step {step_dt}, grid {tuple(model.latitudes.shape)}")

    # -------------------------------------------------------------- IC fetch
    goes = GOES(satellite=args.satellite, scan_mode=args.scan_mode, verbose=False)
    with rec.phase("ic_fetch", sync=False):
        x, x_coords = fetch_data(goes, time=[np.datetime64(start)], variable=variables, lead_time=in_coords["lead_time"], device=device)
    x = x.unsqueeze(0).to(torch.float32)
    x_coords["batch"] = np.arange(1)
    x_coords.move_to_end("batch", last=False)
    rec.extra["ic_frames"] = n_in
    rec.extra["ic_bytes_gpu"] = int(x.numel() * 4)

    # -------------------------------------------------------- compile warmup
    if args.compile:
        with rec.phase("compile_warmup"):
            _ = model(x, x_coords)
            torch.cuda.synchronize()

    # ------------------------------------------------------------- IO setup
    lead_times = np.array([step_dt * k for k in range(args.steps + 1)]).astype("timedelta64[ns]")
    H, W = model.latitudes.shape
    chunks = {"time": 1, "lead_time": 1, "variable": 1, "y": H // 2, "x": W // 2}
    codec = ZstdCodec(level=3)
    if args.io == "zarr":
        io = ZarrBackend(str(out_dir / "store.zarr"), chunks=chunks, zarr_codecs=codec)
    else:
        io = IceChunkBackend(str(out_dir / "store.icechunk"), chunks=chunks, zarr_codecs=codec, blocking=args.blocking_io)

    from collections import OrderedDict

    raw_coords = OrderedDict(time=np.array([np.datetime64(start)]), lead_time=lead_times, variable=variables, y=model.y, x=model.x)
    rgb_coords = OrderedDict(time=np.array([np.datetime64(start)]), lead_time=lead_times, y=model.y, x=model.x, rgb=np.array(["red", "green", "blue"]))
    with rec.phase("io_setup", sync=False):
        io.add_array(raw_coords, "goes")  # float32 default, NaN fill
        io.add_array(
            rgb_coords,
            "true_color",
            data=torch.zeros((1, len(lead_times), H, W, 3), dtype=torch.uint8),
            fill_value=RGB_FILL,
        )
        io.root["true_color"].attrs.update({"valid_min": 0, "valid_max": 255, "color_interpretation": "red green blue"})
        if args.io == "icechunk" and args.commit_mode == "per_step":
            with rec.phase("io_commit", sync=False):
                io.commit("schema")

    def write_frame(pred: torch.Tensor, coords, lead_idx: int) -> None:
        # pred: [1, 1, 1, C, H, W]
        with rec.phase("composite"):
            rgb = true_color(pred[0, 0, 0], variables).cpu()
        c_raw = OrderedDict(time=coords["time"], lead_time=lead_times[lead_idx : lead_idx + 1], variable=variables, y=model.y, x=model.x)
        c_rgb = OrderedDict(time=coords["time"], lead_time=lead_times[lead_idx : lead_idx + 1], y=model.y, x=model.x, rgb=rgb_coords["rgb"])
        with rec.phase("io_write", sync=False):
            io.write(pred[0], c_raw, "goes")
            io.write(rgb[None, None], c_rgb, "true_color")
        if args.io == "icechunk" and args.commit_mode == "per_step":
            with rec.phase("io_commit", sync=False):
                io.commit(f"lead {lead_idx}")

    # ---------------------------------------------------------------- rollout
    t_roll0 = time.perf_counter()
    it = model.create_iterator(x, x_coords)
    pred, pcoords = next(it)  # analysis frame, lead 0
    write_frame(torch.where(torch.isnan(pred), torch.nan, pred), pcoords, 0)
    first_frame_t = None
    for k in range(1, args.steps + 1):
        with rec.phase("inference_step"):
            pred, pcoords = next(it)
        write_frame(pred, pcoords, k)
        if first_frame_t is None:
            first_frame_t = time.perf_counter() - t_roll0
        print(f"[bench4] step {k}/{args.steps} inference {rec.phases['inference_step'][-1]:.2f}s", flush=True)
    if args.io == "icechunk" and args.commit_mode == "per_init":
        with rec.phase("io_commit", sync=False):
            io.commit(f"init {start.isoformat()} {args.steps} steps")
    elif args.io == "icechunk":
        io.flush()
    rollout_s = time.perf_counter() - t_roll0

    # ---------------------------------------------------------------- summary
    rec.finish()
    inf = rec.phases["inference_step"]
    steady = inf[1:] if len(inf) > 1 else inf
    ic = sum(rec.phases["ic_fetch"])
    io_total = sum(rec.phases.get("io_write", [])) + sum(rec.phases.get("io_commit", [])) + sum(rec.phases.get("composite", []))
    rec.extra.update(
        {
            "steps": args.steps,
            "init_time": start.isoformat(),
            "step_seconds_steady": round(sum(steady) / len(steady), 3),
            "step_seconds_first": round(inf[0], 3),
            "time_to_first_frame_s": round(ic + first_frame_t, 3),  # fetch + lead-0 write + first step + its write
            "rollout_s": round(rollout_s, 3),
            "end_to_end_s": round(ic + rollout_s, 3),  # fetch + inference + composites + writes + commit (model already loaded)
            "io_overhead_s": round(io_total, 3),
            "io_overhead_frac_of_rollout": round(io_total / rollout_s, 4),
            "store_bytes": sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()),
            "satellite": args.satellite,
            "scan_mode": args.scan_mode,
            "cache_mode": args.cache,
            "cache_path": cache_path,
            "cache_bytes": cache_size_bytes(cache_path),
        }
    )
    rec.print_summary()
    rec.write()
    if args.io == "icechunk":
        io.flush()
        del io
    if not args.keep:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
