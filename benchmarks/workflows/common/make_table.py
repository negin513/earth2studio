"""Collate results/<benchmark>/*.json into results/RESULTS.md."""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

root = Path(os.environ.get("BENCH_RESULTS", "results"))
out = []
for bdir in sorted(p for p in root.iterdir() if p.is_dir()):
    files = sorted(bdir.glob("*.json"))
    if not files:
        continue
    rows = [json.loads(f.read_text()) for f in files]
    out.append(f"\n## {bdir.name}\n")
    env = rows[-1]["env"]
    out.append(f"GPU: {env.get('gpu')}  torch {env['torch']}  earth2studio {env.get('earth2studio_sha')}\n")
    phases = sorted({p for r in rows for p in r["phases"]})
    vkeys = sorted({k for r in rows for k in r["variant"]})
    ekeys = sorted({k for r in rows for k in r["extra"] if isinstance(r["extra"][k], (int, float))})
    hdr = vkeys + ["total_s", "peak_GiB"] + [f"{p} (mean s)" for p in phases] + ekeys
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for r in rows:
        cells = [str(r["variant"].get(k, "")) for k in vkeys]
        cells += [str(r["total_wall_s"]), str(r["peak_gpu_mem_gib"])]
        for p in phases:
            ph = r["phases"].get(p)
            cells.append("" if ph is None else f"{ph.get('mean_steady_s', ph['mean_s']):.3f} (n={ph['count']})")
        cells += [f"{r['extra'][k]:.4g}" if k in r["extra"] else "" for k in ekeys]
        out.append("| " + " | ".join(cells) + " |")
(root / "RESULTS.md").write_text("# Workflow benchmark results\n" + "\n".join(out) + "\n")
print(root / "RESULTS.md")
