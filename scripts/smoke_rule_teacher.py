"""Smoke test of the rule-augmented teacher store (11/03 round 27, PROPOSALS T1): train.py rl reads
data/teacher_rcp/<pool>.npz with the same keys as data/teacher (occ / count / ok / gp_count). rl from the r2_mott
seed-0 distilled best.pt, 2 iterations on train_pcb10_1 (rule teacher used for 0.69 of its instances), CPU, 1 certify
worker. Checks: the run ends, the log reports the teacher file with ok 1.00, and run_rung.py does not append its
default --teacher_dir after an arm that carries one. On success writes runs/analysis/smoke_rule_teacher.ok, which the
r2t / lo_pcb_r2t queue entries wait for ("after").
Run: .venv/bin/python scripts/smoke_rule_teacher.py"""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK = os.path.join(ROOT, "runs/analysis/smoke_rule_teacher.ok")
TD = "data/teacher_rcp"


def main():
    if not os.path.exists(os.path.join(ROOT, TD, "summary.json")):
        raise SystemExit("teacher build not finished: " + TD)
    d = np.load(os.path.join(ROOT, TD, "train_pcb10_1.npz"))
    assert {"occ", "count", "ok", "gp_count"} <= set(d.files), d.files
    assert d["ok"].mean() == 1.0 and (d["count"] == (d["occ"] >= 0).sum(1)).all()
    src = open(os.path.join(ROOT, "scripts/run_rung.py")).read()
    assert 'tdir = "" if "--teacher_dir" in a.arm.split()' in src
    env = dict(os.environ, CERTIFY_WORKERS="1", OMP_NUM_THREADS="1")
    out = f"/tmp/smoke_rule_teacher_{os.getpid()}"     # outside runs/: not a result; fresh (no resume.pt)
    cmd = [sys.executable, "train.py", "--stage", "rl", "--model", "mott", "--teacher_dir", TD,
           "--device", "cpu", "--init", "runs/2026-10-28/r2_mott_distill/best.pt", "--pool", "runs/pool_v1",
           "--pool_glob", "train_pcb10_1.pkl", "--val_pools", "val_pcb10_0", "--B", "2", "--M", "4", "--chunk", "2",
           "--iters", "2", "--val_every", "25", "--max_steps", "64", "--d", "128", "--layers", "4", "--heads", "8",
           "--out", out]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env, timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"smoke failed rc={r.returncode}")
    txt = open(os.path.join(out, "log.txt")).read()
    m = re.search(r"teacher train_pcb10_1\.npz: teacher file, ok ([0-9.]+)", txt)
    rows = open(os.path.join(out, "log.csv")).read().strip().splitlines()
    if not m or float(m.group(1)) != 1.0 or not rows[-1].startswith("2,"):   # log.csv: logged iterations only
        print(txt[-2000:])
        raise SystemExit("smoke: teacher line or iteration 2 missing")
    os.makedirs(os.path.dirname(OK), exist_ok=True)
    open(OK, "w").write("ok\n")
    print("smoke ok ->", OK)


if __name__ == "__main__":
    main()
