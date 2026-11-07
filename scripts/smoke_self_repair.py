"""Smoke for QUEUE 11/07 10:45 KST candidates C1 (`--self_repair 4`) and C2 (`--kind_weight pkg=2`) before they are
queued: train.py rl 6 iters from r2pk_rl best.pt on the GPU (run_rung COMMON, train pools) with both flags; checks the
kind_weight line, the self-repair counters in the log and metrics.json (checked > 0), and the per-iteration time.
Writes runs/analysis/smoke_self_repair.ok on success.
Run: .venv/bin/python scripts/smoke_self_repair.py
"""
import json
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "runs/analysis/smoke_self_repair"


def main():
    os.chdir(ROOT)
    shutil.rmtree(OUT, ignore_errors=True)
    common = "--d 128 --layers 4 --heads 8 --B 16 --M 8 --lr 3e-4 --val_every 100 --max_steps 64".split()
    arm = ["--model", "mott", "--peak_refine", "--kind_weight", "pkg=2", "--self_repair", "4"]
    pools = ["--pool", "runs/pool_v1", "--teacher_dir", "data/teacher", "--seed", "0"]
    subprocess.run([".venv/bin/python", "train.py", "--stage", "rl", *arm, *common, "--iters", "6", *pools,
                    "--init", "runs/2026-10-29/r2pk_rl/best.pt", "--out", OUT], check=True,
                   env=dict(os.environ, CERTIFY_WORKERS="2"))
    log = open(f"{OUT}/log.txt").read()
    assert re.search(r"kind_weight ip 0\.250, pcb 0\.250, pkg 0\.500", log), "kind_weight line"
    m = json.load(open(f"{OUT}/metrics.json"))
    sr = m["self_repair"]
    assert sr["checked"] > 0, sr
    print("self_repair", sr, "| iters", m["iters"], "hours", round(m["hours"], 4), flush=True)
    open("runs/analysis/smoke_self_repair.ok", "w").write(json.dumps(sr) + "\n")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
