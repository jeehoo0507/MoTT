"""Smoke test of train.py --feas_margin (11/02 round 7, diagnosis H2 cheap fix: a grid-feasible sample must clear
the target by a margin to earn the success reward): rl from the r2_mott seed-0 distilled best.pt (so most samples
are grid-feasible and the new path runs), a few iterations on one small training pool, CPU, 1 certify worker.
Checks the run ended and logged margin-fail. On success writes runs/analysis/smoke_feas_margin.ok, which the queue
entries wait for ("after").
Run: .venv/bin/python scripts/smoke_feas_margin.py"""
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK = os.path.join(ROOT, "runs/analysis/smoke_feas_margin.ok")
DELTA = math.log10(1.05)


def main():
    env = dict(os.environ, CERTIFY_WORKERS="1", OMP_NUM_THREADS="1")
    out = "/tmp/smoke_feas_margin"     # outside runs/: not a result
    cmd = [sys.executable, "train.py", "--stage", "rl", "--model", "mott", "--feas_margin", f"{DELTA:.6f}",
           "--device", "cpu", "--init", "runs/2026-10-28/r2_mott_distill/best.pt", "--pool", "runs/pool_v1",
           "--pool_glob", "train_pkg8_0.pkl", "--val_pools", "val_pkg10_0", "--B", "2", "--M", "4", "--chunk", "2",
           "--iters", "2", "--val_every", "25", "--max_steps", "64", "--d", "128", "--layers", "4", "--heads", "8",
           "--out", out]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"smoke failed rc={r.returncode}")
    txt = open(os.path.join(out, "log.txt")).read()
    m = re.findall(r"margin-fail (\d+)", txt)
    if not m:
        raise SystemExit("no margin-fail in the log")
    open(OK, "w").write(f"ok margin-fail {m} delta {DELTA:.6f}\n")
    print("wrote", OK, m)


if __name__ == "__main__":
    main()
