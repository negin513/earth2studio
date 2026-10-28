"""Analyze workflow benchmark timing outputs.

Reads every results/<benchmark>/*.json written by common.harness.BenchRecorder and
produces:

  results/ANALYSIS.md     human-readable report (per-benchmark headline table, phase
                          breakdown as share of wall time, cold-vs-warm deltas,
                          cross-benchmark summary)
  results/all_runs.csv    one row per run, flattened (variant, extra, phase totals)
  results/phase_breakdown.png   stacked bars of phase time per run (if matplotlib is installed)

Usage:  python common/analyze.py [--results DIR] [--latest-only]
  --latest-only keeps only the newest run per (benchmark, variant) so re-runs
  do not double count.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

# Phase name -> coarse bucket for the breakdown. Anything else -> "other".
BUCKETS = {
    "download": {"model_download", "ic_fetch", "model_load"},
    "compute": {"inference_step", "downscale_step", "diagnostic_step", "perturb",
                "composite", "weekly_stats", "compile_warmup", "regrid"},
    "io": {"io_write", "io_commit", "io_setup", "io_close", "io_flush"},
}
# Headline metric per benchmark: (extra key, label, lower_is_better)
HEADLINE = {
    "01_deterministic_fcn3": ("steps_per_s", "steps/s", False),
    "02_ensemble_hens_lite": ("member_steps_per_s", "member-steps/s", False),
    "03_corrdiff_taiwan": ("seconds_per_sample", "s/sample", True),
    "04_stormscope_goes": ("step_seconds_steady", "s/step", True),
    "05_s2s_long_rollout": ("forecast_days_per_gpu_minute", "fcst days/GPU-min", False),
    "06_stormscope_eu": ("step_seconds_steady", "s/step", True),
}


def load(results: Path, latest_only: bool) -> list[dict]:
    runs = []
    for bdir in sorted(p for p in results.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        for f in sorted(bdir.glob("*.json")):
            try:
                r = json.loads(f.read_text())
            except Exception as e:  # noqa: BLE001
                print(f"skip {f}: {e}")
                continue
            r["_file"] = f.name
            runs.append(r)
    if latest_only:
        keep = {}
        for r in runs:
            k = (r["benchmark"], json.dumps(r["variant"], sort_keys=True))
            if k not in keep or r["started_utc"] > keep[k]["started_utc"]:
                keep[k] = r
        runs = list(keep.values())
    return runs


def phase_total(r: dict, name: str) -> float:
    ph = r["phases"].get(name)
    return ph["total_s"] if ph else 0.0


def bucket_totals(r: dict) -> dict[str, float]:
    out = defaultdict(float)
    for name, ph in r["phases"].items():
        b = next((k for k, v in BUCKETS.items() if name in v), "other")
        out[b] += ph["total_s"]
    return dict(out)


def fmt(x, nd=2):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) >= 0.01 or x == 0 else f"{x:.3g}"
    return str(x)


def variant_label(r: dict) -> str:
    v = {k: val for k, val in r["variant"].items() if k != "cache"}
    return ", ".join(f"{k}={val}" for k, val in sorted(v.items()))


def write_csv(runs: list[dict], path: Path) -> None:
    keys = set()
    rows = []
    for r in runs:
        row = {"benchmark": r["benchmark"], "started_utc": r["started_utc"],
               "total_wall_s": r["total_wall_s"], "peak_gpu_mem_gib": r["peak_gpu_mem_gib"],
               "gpu": r["env"].get("gpu"), "earth2studio_sha": r["env"].get("earth2studio_sha")}
        row.update({f"variant.{k}": v for k, v in r["variant"].items()})
        row.update({f"extra.{k}": v for k, v in r["extra"].items() if not isinstance(v, (dict, list))})
        for name, ph in r["phases"].items():
            row[f"phase.{name}.total_s"] = ph["total_s"]
            row[f"phase.{name}.mean_s"] = ph.get("mean_steady_s", ph["mean_s"])
            row[f"phase.{name}.count"] = ph["count"]
        keys |= row.keys()
        rows.append(row)
    keys = sorted(keys, key=lambda k: (not k.startswith(("benchmark", "variant")), k))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def report(runs: list[dict]) -> str:
    md = ["# Workflow benchmark analysis\n"]
    by_bench = defaultdict(list)
    for r in runs:
        by_bench[r["benchmark"]].append(r)

    # ---------------------------------------------------------- cross-benchmark
    md.append("## Summary (cold cache, best variant per benchmark)\n")
    md.append("| benchmark | variant | headline | total wall s | download+load s | compute s | io s | peak GiB | cache GB |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for b, rs in sorted(by_bench.items()):
        key, label, lower = HEADLINE.get(b, (None, "", True))
        cold = [r for r in rs if r["extra"].get("cache_mode", r["variant"].get("cache")) == "cold"] or rs
        scored = [r for r in cold if key and isinstance(r["extra"].get(key), (int, float))]
        best = (min if lower else max)(scored, key=lambda r: r["extra"][key]) if scored else cold[0]
        bt = bucket_totals(best)
        head = f"{fmt(best['extra'].get(key))} {label}" if key else ""
        cache_gb = best["extra"].get("cache_bytes")
        md.append(f"| {b} | {variant_label(best)} | {head} | {fmt(best['total_wall_s'],1)} | "
                  f"{fmt(bt.get('download',0),1)} | {fmt(bt.get('compute',0),1)} | {fmt(bt.get('io',0),1)} | "
                  f"{fmt(best['peak_gpu_mem_gib'])} | {fmt(cache_gb/1e9 if cache_gb else None)} |")

    # ------------------------------------------------------------ per benchmark
    for b, rs in sorted(by_bench.items()):
        key, label, lower = HEADLINE.get(b, (None, "", True))
        md.append(f"\n## {b}\n")
        env = rs[-1]["env"]
        md.append(f"{len(rs)} runs. GPU {env.get('gpu')}, torch {env.get('torch')}, earth2studio {env.get('earth2studio_sha')}.\n")

        md.append("### Variants\n")
        md.append(f"| variant | cache | {label or 'headline'} | total wall s | peak GiB | " +
                  " | ".join(f"{p} (steady mean s)" for p in sorted({p for r in rs for p in r['phases']})) + " |")
        phases = sorted({p for r in rs for p in r["phases"]})
        md.append("|" + "---|" * (5 + len(phases)))
        for r in sorted(rs, key=lambda r: (variant_label(r), r["extra"].get("cache_mode", ""))):
            cells = [variant_label(r), str(r["extra"].get("cache_mode", r["variant"].get("cache", ""))),
                     fmt(r["extra"].get(key)) if key else "", fmt(r["total_wall_s"], 1), fmt(r["peak_gpu_mem_gib"])]
            for p in phases:
                ph = r["phases"].get(p)
                cells.append("" if ph is None else fmt(ph.get("mean_steady_s", ph["mean_s"]), 3))
            md.append("| " + " | ".join(cells) + " |")

        md.append("\n### Where the time goes (share of total wall)\n")
        md.append("| variant | cache | download+load % | compute % | io % | other/untimed % |")
        md.append("|---|---|---|---|---|---|")
        for r in sorted(rs, key=variant_label):
            tw = r["total_wall_s"] or 1.0
            bt = bucket_totals(r)
            timed = sum(bt.values())
            md.append(f"| {variant_label(r)} | {r['extra'].get('cache_mode','')} | "
                      f"{100*bt.get('download',0)/tw:.0f} | {100*bt.get('compute',0)/tw:.0f} | "
                      f"{100*bt.get('io',0)/tw:.0f} | {100*max(tw-timed,0)/tw:.0f} |")

        # cold vs warm pairs
        pairs = defaultdict(dict)
        for r in rs:
            pairs[variant_label(r)][r["extra"].get("cache_mode", r["variant"].get("cache"))] = r
        both = {k: v for k, v in pairs.items() if "cold" in v and "warm" in v}
        if both:
            md.append("\n### Cold vs warm cache\n")
            md.append("| variant | cold wall s | warm wall s | cold penalty s | cold download+load s | cache GB fetched |")
            md.append("|---|---|---|---|---|---|")
            for k, v in sorted(both.items()):
                c, w = v["cold"], v["warm"]
                cb = c["extra"].get("cache_bytes")
                md.append(f"| {k} | {fmt(c['total_wall_s'],1)} | {fmt(w['total_wall_s'],1)} | "
                          f"{fmt(c['total_wall_s']-w['total_wall_s'],1)} | {fmt(bucket_totals(c).get('download',0),1)} | "
                          f"{fmt(cb/1e9 if cb else None)} |")

        notes = {k: v for r in rs for k, v in r["extra"].items()
                 if k in ("nan_detected", "gpu_shared", "failures", "oom") and v}
        if notes:
            md.append("\n### Flags\n")
            for k, v in notes.items():
                md.append(f"- {k}: {v}")
    return "\n".join(md) + "\n"


def plot(runs: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("matplotlib not available; skipping plot")
        return
    labels, data = [], defaultdict(list)
    for r in sorted(runs, key=lambda r: (r["benchmark"], variant_label(r))):
        labels.append(f"{r['benchmark'][:2]} {variant_label(r)} [{r['extra'].get('cache_mode','')}]")
        bt = bucket_totals(r)
        for k in ("download", "compute", "io", "other"):
            data[k].append(bt.get(k, 0.0))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    left = [0.0] * len(labels)
    for k, color in zip(("download", "compute", "io", "other"), ("#7f8c8d", "#2980b9", "#e67e22", "#bdc3c7")):
        ax.barh(labels, data[k], left=left, label=k, color=color)
        left = [a + b for a, b in zip(left, data[k])]
    ax.set_xlabel("seconds"); ax.invert_yaxis(); ax.legend(loc="lower right"); ax.set_title("Phase time per run")
    fig.tight_layout(); fig.savefig(path, dpi=120); print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.environ.get("BENCH_RESULTS", "results"))
    ap.add_argument("--latest-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    results = Path(a.results)
    runs = load(results, a.latest_only)
    if not runs:
        raise SystemExit(f"no result JSON under {results}")
    (results / "ANALYSIS.md").write_text(report(runs))
    write_csv(runs, results / "all_runs.csv")
    if not a.no_plot:
        plot(runs, results / "phase_breakdown.png")
    print(f"{len(runs)} runs -> {results/'ANALYSIS.md'}, {results/'all_runs.csv'}")


if __name__ == "__main__":
    main()
