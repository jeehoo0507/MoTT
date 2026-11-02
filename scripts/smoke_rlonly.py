"""Smoke test of the rl-only path (2026-11-02, B-Manoharan / B-Park paper-faithful re-runs): train.py --stage rl
from random init (no --init) with --imit 0, a few iterations on one small training pool, CPU, 1 certify worker.
On success writes runs/analysis/smoke_rlonly.ok, which the queue entries wait for ("after").
Run: .venv/bin/python scripts/smoke_rlonly.py"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK = os.path.join(ROOT, "runs/analysis/smoke_rlonly.ok")


def main():
    env = dict(os.environ, CERTIFY_WORKERS="1", OMP_NUM_THREADS="1")
    for arm in (["--model", "manoharan"], ["--model", "park_am", "--baseline", "rollout"]):
        out = "/tmp/smoke_rlonly_" + arm[1]     # outside runs/: not a result
        cmd = [sys.executable, "train.py", "--stage", "rl", *arm, "--imit", "0", "--device", "cpu", "--pool", "runs/pool_v1",
               "--pool_glob", "train_pcb8_0.pkl", "--val_pools", "val_pcb10_0", "--B", "2", "--M", "2", "--chunk", "2",
               "--iters", "3", "--val_every", "3", "--max_steps", "64", "--d", "128", "--layers", "4", "--heads", "8",
               "--out", out]
        print("$", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=ROOT, env=env, timeout=1800)
        if r.returncode != 0:
            raise SystemExit(f"smoke failed rc={r.returncode}: {arm}")
    open(OK, "w").write("ok\n")
    print("wrote", OK)


if __name__ == "__main__":
    main()
