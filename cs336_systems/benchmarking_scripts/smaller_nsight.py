#!/usr/bin/env python3
"""Minimal Nsight profile summariser for CS336 A2."""
import csv, json, re, subprocess, sys
from pathlib import Path
from collections import defaultdict

REPORTS = {"nvtx": "nvtx_sum", "kern": "cuda_gpu_kern_sum"}
MODELS  = ["small", "medium", "large", "xl", "2.7B"]
CTXS    = [128, 256, 512, 1024]

def nsys_csv(report, rep_path):
    """Return a list[dict] for the chosen `nsys stats` report."""
    out = subprocess.check_output(
        ["nsys", "stats", "--report", report, "--format", "csv", rep_path],
        text=True,
    )
    rows = (l for l in out.splitlines() if l and l[0] not in "#-=")
    return list(csv.DictReader(rows))

def analyse(rep_path: Path):
    m = re.match(r"(?P<model>.+?)_ctx(?P<ctx>\d+)", rep_path.stem)
    if not m:
        return None
    model, ctx = m["model"], int(m["ctx"])

    nvtx  = nsys_csv(REPORTS["nvtx"],  rep_path)
    kerns = nsys_csv(REPORTS["kern"],  rep_path)

    total_ns = int(nvtx[0]["Total Time (ns)"])         # first row = whole run
    top      = max(kerns, key=lambda r: int(r["Total Time (ns)"]))

    # Bucket kernel time to answer parts (c) & (e)
    buckets = defaultdict(int)
    for k in kerns:
        name = k["Name"].lower()
        ns   = int(k["Total Time (ns)"])
        if any(p in name for p in ("gemm", "cublas", "wmma")):
            buckets["gemm"] += ns
        elif "softmax" in name:
            buckets["softmax"] += ns
        elif any(p in name for p in ("elementwise", "activation", "gelu", "relu")):
            buckets["elementwise"] += ns
        else:
            buckets["other"] += ns

    return {
        "model": model, "ctx": ctx,
        "forward_ms": total_ns / 1e6,
        "top_kernel": top["Name"],
        "top_ms": int(top["Total Time (ns)"]) / 1e6,
        "top_count": int(top["Count"]),
        "fractions": {k: v * 100 / total_ns for k, v in buckets.items()},
    }

def main(root: str):
    reps = Path(root).glob("*.nsys*")
    results = [r for p in reps if (r := analyse(p))]
    # pretty-print table (question a)
    tbl = defaultdict(dict)
    for r in results:
        tbl[r["model"]][r["ctx"]] = f"{r['forward_ms']:.1f}"
    header = "model " + " ".join(f"ctx{c}" for c in CTXS)
    print(header)
    for m in MODELS:
        row = " ".join(tbl[m].get(c, "OOM/NA") for c in CTXS)
        print(f"{m:6} {row}")
    # save machine-readable JSON
    json.dump(results, open("nsight_minimal.json", "w"), indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: minimal_nsys.py <dir_with_.nsys-rep>")
    main(sys.argv[1])
