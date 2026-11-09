"""Smoke test of the rl-step memory guards in train.py (10/28 18:20: r2 rl peaked at 18.3 GB by iter 25
because every --chunk pass added its own padded block of grid points, F 267 -> 511 on one pool).
  1. --grid_growth 0 and a CUDA OOM injected into the first rollout: the step is skipped, chunk 2 -> 1,
     no pool grid grows, the run still finishes.
  2. --grid_growth 0.5: no pool grid grows past 1.5x its initial F.
CPU, two small pools, inline exact checks (CERTIFY_WORKERS=1), so it does not touch the GPU.
Run: .venv/bin/python scripts/smoke_rl_grid.py
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARGS = ["--stage", "rl", "--feat", "coord", "--device", "cpu", "--pool", "runs/pool_v1",
        "--pool_glob", "train_pcb8_0.pkl,train_ip8_0.pkl", "--val_pools", "val_ip10_0", "--d", "32", "--layers", "1",
        "--heads", "4", "--B", "4", "--M", "2", "--iters", "6", "--val_every", "1000", "--max_steps", "16",
        "--chunk", "2"]
INJECT = """
import sys, torch
sys.path.insert(0, {root!r})
import train
orig, n = train.rollout, [0]
def rollout(*a, **k):
    n[0] += 1
    if n[0] == 1:
        raise torch.OutOfMemoryError("injected by smoke_rl_grid")
    return orig(*a, **k)
train.rollout = rollout
sys.argv = ["train.py"] + {args!r}
train.main()
"""

for name, extra, inject in (("oom", ["--grid_growth", "0"], True), ("cap", ["--grid_growth", "0.5"], False)):
    out = f"runs/smoke_rl_grid_{name}"
    shutil.rmtree(os.path.join(ROOT, out), ignore_errors=True)
    args = ARGS + extra + ["--out", out]
    cmd = [sys.executable, "-c", INJECT.format(root=ROOT, args=args)] if inject else [sys.executable, "train.py"] + args
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800,
                       env=dict(os.environ, CERTIFY_WORKERS="1"))
    print(f"--- {name}: rc {r.returncode}", r.stdout[-1500:], r.stderr[-3000:], sep="\n")
    assert r.returncode == 0, r.returncode
    m = json.load(open(os.path.join(ROOT, out, "metrics.json")))
    print(name, "iters", m["iters"], "oom", m["cuda_oom_skipped"], "chunk", m["chunk_final"], "pool_F", m["pool_F"],
          "capped", m["grid_capped_steps"])
    assert m["iters"] == 6, m["iters"]
    if inject:
        assert m["cuda_oom_skipped"] == 1 and m["chunk_final"] == 1, m
        assert all(f == f0 for f0, f in m["pool_F"].values()), m["pool_F"]
    else:
        assert m["cuda_oom_skipped"] == 0 and m["chunk_final"] == 2, m
        assert all(f <= int(1.5 * f0) for f0, f in m["pool_F"].values()), m["pool_F"]
print("SMOKE RL GRID OK")
