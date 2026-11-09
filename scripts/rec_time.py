"""Print timing fields of eval records (read only): set, method, n, sec_per_inst, workers, inst_chunk, wall, finished.

Run: .venv/bin/python scripts/rec_time.py 'runs/2026-10-2[89]/r2*_rl/eval/eval_pcb16_*.json'
"""
import glob
import json
import os
import sys

for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat)):
        r = json.load(open(f))
        keys = {k: r.get(k) for k in ("workers", "inst_chunk", "peak_rss_mb", "wall_s", "time", "date", "git") if k in r}
        cnt = r.get("count", [])
        print(f"{r['set']:14s} {r['method']:18s} n={r['n']:3d} sec/inst={r['sec_per_inst']:7.2f} "
              f"mean_count={sum(cnt) / max(1, len(cnt)):6.2f} mtime={os.path.getmtime(f):.0f} {keys}")
