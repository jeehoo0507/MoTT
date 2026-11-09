"""Smoke test of train.py restart-resume (QUEUE 2026-10-28: checkpoint every 20 min, continue after a pod restart).

Run A: 6 iters, resume.pt after every iteration. Run B: same --out, --iters 12 -> must log "resumed from"
at iter 6 and finish at 12 with log.csv rows appended. Tiny model, 2 pools, inline exact checks
(CERTIFY_WORKERS=1, so no extra worker processes).
Run: .venv/bin/python scripts/smoke_resume.py
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "runs/smoke_resume"
BASE = [sys.executable, "train.py", "--stage", "rl", "--feat", "modal", "--pool", "runs/pool_v1",
        "--pool_glob", "train_pcb8_0.pkl,train_ip8_0.pkl", "--d", "32", "--layers", "1", "--heads", "4",
        "--B", "4", "--M", "2", "--val_every", "1000", "--ckpt_min", "0.001", "--out", OUT]


def run(iters):
    env = dict(os.environ, CERTIFY_WORKERS="1")
    r = subprocess.run(BASE + ["--iters", str(iters)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
    assert r.returncode == 0, r.returncode
    return r.stdout


shutil.rmtree(os.path.join(ROOT, OUT), ignore_errors=True)
run(6)
os.remove(os.path.join(ROOT, OUT, "metrics.json"))
out = run(12)
assert "resumed from" in out and "iter 6," in out, "no resume"
m = json.load(open(os.path.join(ROOT, OUT, "metrics.json")))
assert m["iters"] == 12, m["iters"]
print("SMOKE RESUME OK")
