"""Smoke test of train.py --chunk (rl stage, instances split into rollout/backward passes; 10/28 16:23 the
one-pass B=16 x M=8 rl step hit CUDA OOM). Runs the rl stage with --chunk 2 (two passes per step) and
--chunk 0 (one pass) on two small pools, CPU, inline exact checks (CERTIFY_WORKERS=1), so it does not
touch the GPU that the running rung uses. Both must finish and write metrics.json.
Run: .venv/bin/python scripts/smoke_chunk.py
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = [sys.executable, "train.py", "--stage", "rl", "--feat", "coord", "--device", "cpu", "--pool", "runs/pool_v1",
        "--pool_glob", "train_pcb8_0.pkl,train_ip8_0.pkl", "--val_pools", "val_ip10_0", "--d", "32", "--layers", "1",
        "--heads", "4", "--B", "4", "--M", "2", "--iters", "4", "--val_every", "1000", "--max_steps", "16"]

for chunk in (2, 0):
    out = f"runs/smoke_chunk{chunk}"
    shutil.rmtree(os.path.join(ROOT, out), ignore_errors=True)
    r = subprocess.run(BASE + ["--chunk", str(chunk), "--out", out], cwd=ROOT, capture_output=True, text=True,
                       timeout=1800, env=dict(os.environ, CERTIFY_WORKERS="1"))
    print(f"--- chunk {chunk}: rc {r.returncode}", r.stdout[-1500:], r.stderr[-3000:], sep="\n")
    assert r.returncode == 0, r.returncode
    m = json.load(open(os.path.join(ROOT, out, "metrics.json")))
    assert m["iters"] == 4, m["iters"]
print("SMOKE CHUNK OK")
