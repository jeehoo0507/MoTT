"""Side-16 greedy+prune (the count reference) and modal rule in the CPU slot the eval-pool builder frees.

10/29 01:40 UTC: the driver's 2 eval workers spend hours per rung stage on side 16 (> 75 min per policy job),
and greedy+prune on side 16 was last in its order, so the side-16 count ratios would stay empty for a day.
This waits until scripts/build_pools.py (1 worker left, eval_ip16_1) has exited, then runs greedy+prune and
then the modal rule on every side-16 eval pool with 1 worker: same eval_policy.py, same records
(runs/<day>/baselines/eval/<set>__<method>.json); existing records are skipped by eval_policy and the driver.
Concurrent CPU workers stay 4 (driver 2 + side_evals.py 1 + this 1; QUEUE limit). Stops before a step if
side-20 pools are being built.
Run: .venv/bin/python scripts/bg.py agent/logs/side16_gp.log '.venv/bin/python scripts/side16_gp.py'
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import mem_line, wait_mem  # noqa: E402
from pipeline_day1 import alive  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUT = "runs/2026-10-28/baselines"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


log(f"wait for scripts/build_pools.py to exit | {mem_line()}")
while alive("scripts/build_pools.py"):
    time.sleep(60)
for methods in ("gp", "modal_rule"):
    if alive("--eval_sizes 20"):
        log("side-20 pools building, stop")
        break
    wait_mem(log)
    log(f"START --methods {methods} side 16 | {mem_line()}")
    r = subprocess.run(f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*16_*' --methods {methods} "
                       f"--workers 1 --out {OUT}", shell=True, cwd=ROOT)
    log(f"END rc={r.returncode} | {mem_line()}")
log("side16 gp done")
