"""Smoke test of train.py --hard_weight (11/03 round 29, PROPOSALS T3). CPU, 1 certify worker, rl from the r2_mott
seed-0 distilled best.pt, 2 iterations on train_pcb10_1 + train_pcb12_0 (different g+p counts -> spread of weights).
Checks: (1) with --hard_weight 2 the run ends, logs instance weights spanning 1 and writes iteration 2;
(2) with the default (0) the per-iteration log.csv rows equal those of the committed train.py (git HEAD) for the
same seed - the option is a no-op when off, so queued runs (r2tf, lo_pcb_r2tf) are unchanged.
On success writes runs/analysis/smoke_hard_weight.ok, which the r2h / lo_pcb_r2h queue entries wait for ("after").
Run: .venv/bin/python scripts/smoke_hard_weight.py"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK = os.path.join(ROOT, "runs/analysis/smoke_hard_weight.ok")
HEAD = os.path.join(ROOT, "_train_head_smoke.py")   # train.py of git HEAD, removed at the end


def run(script, out, extra):
    env = dict(os.environ, CERTIFY_WORKERS="1", OMP_NUM_THREADS="1")
    cmd = [sys.executable, script, "--stage", "rl", "--model", "mott", "--device", "cpu",
           "--init", "runs/2026-10-28/r2_mott_distill/best.pt", "--pool", "runs/pool_v1",
           "--pool_glob", "train_pcb10_1.pkl,train_pcb12_0.pkl", "--val_pools", "val_pcb10_0", "--B", "2", "--M", "4",
           "--chunk", "2", "--iters", "2", "--val_every", "25", "--max_steps", "64", "--d", "128", "--layers", "4",
           "--heads", "8", "--seed", "0", "--out", out] + extra
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"smoke failed rc={r.returncode}: {script} {extra}")
    rows = open(os.path.join(out, "log.csv")).read().strip().splitlines()
    return open(os.path.join(out, "log.txt")).read(), rows


def main():
    pid = os.getpid()
    txt, rows = run("train.py", f"/tmp/smoke_hw2_{pid}", ["--hard_weight", "2"])
    m = re.search(r"hard_weight 2\.0: instance weights ([0-9.]+)\.\.([0-9.]+)", txt)
    if not m or not float(m.group(1)) < 1.0 < float(m.group(2)) or not rows[-1].startswith("2,"):
        print(txt[-2000:])
        raise SystemExit("smoke: weight line or iteration 2 missing")
    print("weights", m.group(1), m.group(2))
    open(HEAD, "w").write(subprocess.run(["git", "show", "HEAD:train.py"], cwd=ROOT, capture_output=True,
                                         text=True, check=True).stdout)
    try:
        _, rows_new = run("train.py", f"/tmp/smoke_hw0_{pid}", [])
        _, rows_old = run(HEAD, f"/tmp/smoke_head_{pid}", [])
    finally:
        os.remove(HEAD)
    strip = lambda rs: [",".join(c for i, c in enumerate(r.split(",")) if i != 6) for r in rs]   # col 6 = sec
    if strip(rows_new) != strip(rows_old):
        print("new", rows_new, "\nold", rows_old)
        raise SystemExit("smoke: --hard_weight 0 differs from HEAD train.py")
    os.makedirs(os.path.dirname(OK), exist_ok=True)
    open(OK, "w").write("ok\n")
    print("smoke ok ->", OK)


if __name__ == "__main__":
    main()
