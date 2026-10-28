"""Benchmark 6: StormScope-EU satellite nowcast, booth-pipeline style.

Mirrors the Earthmover GTC Berlin 2026 StormScope-EU booth pipeline
(earthmover-support/nvidia-gtc-berlin-2026-stormscope_eu-booth,
scripts/stormscope_eu_pipeline.py): a 6-frame MTG-I1 FCI initial condition from
the EUMETSAT Data Store, ``StormScopeMeteosatEU`` rolled out N x 10 min, each step
turned into uint8 ``true_color`` and ``airmass`` RGB composites and written to a
Zarr or Icechunk store (local), with one Icechunk commit per step or per init.

Phases (see common/harness.py): ic_fetch, model_download, model_load,
compile_warmup, inference_step, composite, io_write, io_commit.

Default cache mode is cold: EARTH2STUDIO_CACHE is pointed at a fresh empty dir so
both the model weights (~2.4 GB) and the six FCI FDHSI products (~0.5-1 GB zip
each) are downloaded from scratch, like the booth's live runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import UTC, datetime, timedelta

BENCHMARK = "06_stormscope_eu"

# Composite recipe copied from the booth script -------------------------------
RGB_CHANNELS = ["fci06vis", "fci05vis", "fci04vis"]  # VIS 0.6 / 0.5 / 0.4 um
AIRMASS_CHANNELS = ["fci63wv", "fci73wv", "fci97ir", "fci105ir"]
WAVELENGTH_CM = {
    "fci63wv": 6.300e-4,
    "fci73wv": 7.350e-4,
    "fci97ir": 9.660e-4,
    "fci105ir": 10.500e-4,
}
RGB_DTYPE = "uint8"
RGB_FILL = 0
# 2x2 spatial chunks per frame of the Model_FCI_BBox (y=1120, x=2432), as in the booth.
CHUNKS = {"ensemble": 1, "time": 1, "lead_time": 1, "y": 560, "x": 1216}
INIT_OFFSET_MIN = 10  # stored init time = slot time + scan duration


def pick_gpu() -> tuple[int, bool]:
    """Choose the GPU with the least memory in use; return (index, shared)."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        idx = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return idx, int(out.strip()) > 500
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    rows = [tuple(int(v) for v in line.split(",")) for line in out.strip().splitlines()]
    idx, used = min(rows, key=lambda r: r[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    return idx, used > 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=12, help="10-min rollout steps (12 = 2 h)")
    p.add_argument("--io", choices=["zarr", "icechunk"], default="icechunk")
    p.add_argument("--commit-mode", choices=["per_step", "per_init"], default="per_init",
                   help="icechunk only: commit after every step or once at the end")
    p.add_argument("--compile", action="store_true", help="torch.compile the denoising experts")
    p.add_argument("--cache", choices=["cold", "warm"], default="cold")
    p.add_argument("--start-time", type=str, default=None,
                   help="FCI slot time, ISO UTC (e.g. 2026-10-28T10:00). Default: newest slot >= --min-lag old")
    p.add_argument("--min-lag", type=int, default=90, help="minutes; slots younger than this are skipped")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--blocking-io", action="store_true", help="icechunk: synchronous writes")
    return p.parse_args()


def floor_to_slot(when: datetime, scan_s: int) -> datetime:
    epoch = datetime(2000, 1, 1, tzinfo=UTC)
    when = when.astimezone(UTC) if when.tzinfo else when.replace(tzinfo=UTC)
    slots = int((when - epoch).total_seconds()) // scan_s
    return epoch + timedelta(seconds=slots * scan_s)


def latest_fci_slot(n_frames: int, min_lag: timedelta, lookback: timedelta = timedelta(hours=6)) -> datetime:
    """Newest FDHSI slot with a complete n_frames input window, >= min_lag old (Data Store search)."""
    import eumdac
    from earth2studio.data.meteosat_fci import MeteosatFCI
    from earth2studio.data.utils_eumetsat import eumetsat_credentials

    now = datetime.now(UTC)
    token = eumdac.AccessToken(credentials=eumetsat_credentials())
    coll = eumdac.DataStore(token).get_collection(MeteosatFCI.COLLECTION_ID["FDHSI"])
    products = coll.search(dtstart=now - lookback, dtend=now - min_lag)
    step = timedelta(seconds=MeteosatFCI.SCAN_FREQUENCY)
    slots = {floor_to_slot(p.sensing_start, MeteosatFCI.SCAN_FREQUENCY) for p in products}
    for slot in sorted(slots, reverse=True):
        if slots.issuperset(slot - i * step for i in range(n_frames)):
            return slot
    raise FileNotFoundError(f"no {n_frames} consecutive FDHSI slots between {lookback} and {min_lag} ago")


def true_color(arr, variables):
    import numpy as np

    idx = [list(variables).index(ch) for ch in RGB_CHANNELS]
    rgb = np.clip(arr[..., idx, :, :] / 24.0, 0, 1) * 255
    rgb = np.interp(rgb, [0, 30, 60, 120, 190, 255], [0, 110, 160, 210, 240, 255])
    rgb = np.where(np.isnan(rgb), RGB_FILL, np.clip(np.round(rgb), 1, 255))
    return np.moveaxis(rgb, -3, -1).astype(RGB_DTYPE)


def airmass(arr, variables):
    import numpy as np
    from earth2studio.data.utils import radiance_to_bt

    bt = {
        v: radiance_to_bt(arr[..., list(variables).index(v), :, :], 1.0 / WAVELENGTH_CM[v])
        for v in AIRMASS_CHANNELS
    }
    r = ((bt["fci63wv"] - bt["fci73wv"]) - (-23.8)) / (1.4 - (-23.8))
    g = ((bt["fci97ir"] - bt["fci105ir"]) - (-39.7)) / (4.1 - (-39.7))
    b = (bt["fci63wv"] - 244.5) / (209.4 - 244.5)
    rgb = np.stack([r, g, b], axis=-1) * 255
    return np.where(np.isnan(rgb), RGB_FILL, np.clip(np.round(rgb), 1, 255)).astype(RGB_DTYPE)


def main() -> None:
    args = parse_args()
    gpu_index, gpu_shared = pick_gpu()

    # Must run before anything resolves EARTH2STUDIO_CACHE.
    from common.harness import BenchRecorder, cache_size_bytes, setup_cache

    cache_path = setup_cache(args.cache, BENCHMARK)
    if args.cache == "warm" and "EARTH2STUDIO_DATA_CACHE" not in os.environ:
        # ~/.cache lives on an NFS home with a user quota that FCI zips (~1 GB each)
        # blow through; keep warm data caches on the scratch filesystem instead.
        os.environ["EARTH2STUDIO_DATA_CACHE"] = os.path.join(
            os.environ.get("BENCH_SCRATCH", "/tmp/e2s_bench"), "data_cache_warm"
        )
        os.makedirs(os.environ["EARTH2STUDIO_DATA_CACHE"], exist_ok=True)

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from earth2studio.data import fetch_data
    from earth2studio.data.meteosat_fci import MeteosatFCI
    from earth2studio.io import IceChunkBackend, ZarrBackend
    from earth2studio.models.px.stormscope_meteosat import StormScopeMeteosatEU
    from earth2studio.utils.coords import map_coords

    variant = {
        "io": args.io,
        "commit": args.commit_mode if args.io == "icechunk" else "n/a",
        "compile": args.compile,
        "cache": args.cache,
    }
    rec = BenchRecorder(BENCHMARK, variant)
    device = torch.device("cuda")
    t_main0 = time.perf_counter()

    scratch = os.environ.get("BENCH_SCRATCH", "/tmp/e2s_bench")
    out_dir = args.output_dir or os.path.join(
        scratch, "06", f"{args.io}_{variant['commit']}_{'compile' if args.compile else 'eager'}_{rec.started:%Y%m%dT%H%M%SZ}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------ model
    package = StormScopeMeteosatEU.load_default_package()
    with rec.phase("model_download", sync=False):
        cfg = OmegaConf.load(package.resolve("config.yaml"))
        for spec in cfg["model_spec"]:
            package.resolve(spec["path"])
    with rec.phase("model_load"):
        model = StormScopeMeteosatEU.load_model(package=package).to(device).eval()
    in_coords = model.input_coords()
    n_frames = len(in_coords["lead_time"])
    variables = [str(v) for v in in_coords["variable"]]
    if args.compile:
        model.compile_model()

    # ------------------------------------------------------------ init time
    if args.start_time:
        slot = floor_to_slot(datetime.fromisoformat(args.start_time), MeteosatFCI.SCAN_FREQUENCY)
    else:
        try:
            slot = latest_fci_slot(n_frames, timedelta(minutes=args.min_lag))
        except Exception as err:  # noqa: BLE001
            print(f"[bench] Data Store search failed ({err!r}); falling back to now - {args.min_lag + 30} min")
            slot = floor_to_slot(datetime.now(UTC) - timedelta(minutes=args.min_lag + 30), MeteosatFCI.SCAN_FREQUENCY)
    start_time = np.datetime64(slot.replace(tzinfo=None))
    print(f"[bench] init slot {slot:%Y-%m-%d %H:%M} UTC ({(datetime.now(UTC)-slot).total_seconds()/60:.0f} min ago)")

    # ------------------------------------------------------------ IC fetch (6 frames, 1 km + 2 km)
    bbox_2km = StormScopeMeteosatEU.Model_FCI_BBox
    bbox_1km = ((2 * bbox_2km[0][0], 2 * bbox_2km[0][1]), (2 * bbox_2km[1][0], 2 * bbox_2km[1][1]))
    # Six ~750 MB FDHSI zips at the ~2.5 MB/s seen from this box take ~30 min; the source default is 1200 s.
    fci_timeout = int(os.environ.get("FCI_FETCH_TIMEOUT", "7200"))
    fci = {
        "2km": MeteosatFCI(resolution="2km", pixel_bbox=bbox_2km, async_timeout=fci_timeout),
        "1km": MeteosatFCI(resolution="1km", pixel_bbox=bbox_1km, async_timeout=fci_timeout),
    }
    t_e2e0 = time.perf_counter()
    with rec.phase("ic_fetch"), torch.no_grad():
        x_res = {}
        for res in ["2km", "1km"]:
            avail = fci[res].available_variables()
            if res == "1km":
                avail -= fci["2km"].available_variables()
            x_res[res] = fetch_data(
                fci[res], time=np.array([start_time]), variable=[v for v in variables if v in avail],
                lead_time=in_coords["lead_time"], device=device,
            )
        x, coords = StormScopeMeteosatEU.combine_1km_2km_inputs(*x_res["1km"], *x_res["2km"])
        del x_res
        x, coords = map_coords(x, coords, in_coords)
        x = x.expand(1, -1, -1, -1, -1, -1)
        coords["ensemble"] = np.arange(1)
        coords.move_to_end("ensemble", last=False)

    if args.compile:
        with rec.phase("compile_warmup"), torch.no_grad():
            model(x, coords)

    # ------------------------------------------------------------ store schema
    lead_time = np.arange(args.steps + 1) * model.output_times[0]
    stored_time = coords["time"] + np.timedelta64(INIT_OFFSET_MIN, "m")
    total_coords = OrderedDict(ensemble=coords["ensemble"], time=stored_time, lead_time=lead_time,
                               y=coords["y"], x=-np.asarray(coords["x"]))  # east-positive like the booth
    rgb_coords = OrderedDict(**total_coords, rgb=np.array(["red", "green", "blue"]))
    with rec.phase("io_schema", sync=False):
        if args.io == "zarr":
            backend = ZarrBackend(os.path.join(out_dir, "store.zarr"), chunks=CHUNKS)
        else:
            backend = IceChunkBackend(os.path.join(out_dir, "store.icechunk"), chunks=CHUNKS, blocking=args.blocking_io)
        backend.add_array(rgb_coords, [])  # coordinate arrays only
        for name in ("true_color", "airmass"):
            backend.root.create_array(
                name, shape=[len(v) for v in rgb_coords.values()],
                chunks=[CHUNKS.get(d, len(v)) for d, v in rgb_coords.items()],
                dtype=RGB_DTYPE, fill_value=RGB_FILL, dimension_names=list(rgb_coords),
            )
        backend.root["true_color"].attrs.update({"valid_min": 0, "valid_max": 255, "source_channels": " ".join(RGB_CHANNELS)})
        backend.root["airmass"].attrs.update({"valid_min": 0, "valid_max": 255, "source_channels": " ".join(AIRMASS_CHANNELS)})
        for cname, arr in (("lat", model.lat), ("lon", model.lon)):
            backend.root.create_array(cname, shape=tuple(arr.shape), chunks=(CHUNKS["y"], CHUNKS["x"]),
                                      dtype="float32", dimension_names=["y", "x"])[:] = arr.detach().cpu().numpy().astype("float32")
        backend.root.attrs.update({"title": "StormScope-EU nowcast on MTG-I1 FCI (benchmark 06)", "model": "StormScopeMeteosatEU"})

    # ------------------------------------------------------------ rollout
    first_frame_s = None
    commits = 0
    with torch.no_grad():
        for step, (x_pred, coords_pred) in enumerate(model.create_iterator(x, coords)):
            if step > 0:
                rec.record("inference_step", time.perf_counter() - t_step0)
            # forecast only: step 0 yields the 6 observed frames; keep lead >= 0
            lead = coords_pred["lead_time"]
            s0 = int(np.searchsorted(lead, np.timedelta64(0, "s")))
            with rec.phase("composite"):
                arr = x_pred[:, :, s0:].detach().to("cpu", torch.float32, copy=True).numpy()
                step_coords = OrderedDict((d, v) for d, v in coords_pred.items() if d != "variable")
                step_coords["lead_time"] = lead[s0:]
                step_coords["time"] = step_coords["time"] + np.timedelta64(INIT_OFFSET_MIN, "m")
                step_coords["x"] = -np.asarray(step_coords["x"])
                step_coords["rgb"] = rgb_coords["rgb"]
                tc, am = torch.from_numpy(true_color(arr, variables)), torch.from_numpy(airmass(arr, variables))
            with rec.phase("io_write", sync=False):
                backend.write([tc, am], step_coords, ["true_color", "airmass"])
            if args.io == "icechunk" and args.commit_mode == "per_step":
                with rec.phase("io_commit", sync=False):
                    backend.commit(f"step {step}")
                    commits += 1
            if step == 0:
                first_frame_s = time.perf_counter() - t_e2e0
            if step == args.steps:
                break
            torch.cuda.synchronize()
            t_step0 = time.perf_counter()
    with rec.phase("io_commit", sync=False):
        if args.io == "icechunk" and args.commit_mode == "per_init":
            backend.commit(f"stormscope-eu {np.datetime_as_string(stored_time[0], unit='m')} lead 0-{args.steps*10} min")
            commits += 1
        elif args.io == "icechunk":
            backend.flush()
    end_to_end_s = time.perf_counter() - t_e2e0

    rec.finish()
    inf = rec.phases.get("inference_step", [])
    steady = sum(inf[1:]) / (len(inf) - 1) if len(inf) > 1 else (inf[0] if inf else None)
    out_bytes = sum(p.stat().st_size for p in __import__("pathlib").Path(out_dir).rglob("*") if p.is_file())
    rec.extra.update({
        "steps": args.steps,
        "init_slot_utc": str(start_time),
        "step_seconds_steady": None if steady is None else round(steady, 3),
        "time_to_first_frame_s": None if first_frame_s is None else round(first_frame_s, 3),
        "end_to_end_s": round(end_to_end_s, 3),
        "end_to_end_incl_model_s": round(time.perf_counter() - t_main0, 3),
        "commits": commits,
        "output_bytes": out_bytes,
        "output_dir": out_dir,
        "cache_mode": args.cache,
        "cache_path": cache_path,
        "cache_bytes": cache_size_bytes(cache_path),
        "gpu_index": gpu_index,
        "gpu_shared": gpu_shared,
        "nan_in_last_step": bool(np.isnan(arr).all()),
    })
    rec.print_summary()
    rec.write()


if __name__ == "__main__":
    main()
