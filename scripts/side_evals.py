"""Side-10 evals on the 4th CPU slot while the driver's early evals (2 workers) and the last side-16 pool
(1 worker) run — 10/29 00:20 UTC: one eval_pcb16 policy job takes > 30 min, so the driver's order (every set
of one policy, biggest first) would leave the side-10 ladder empty for many hours.

Same eval_policy.py, same records (<run>/eval/<set>__<method>.json), one worker; eval_policy skips a record
that already exists, and the driver skips the ones written here. Order: greedy+prune (the count reference),
then the rl rungs, r3 at test-time R, then the distill rungs (r0 distill last: the driver is on it now).
Stops before a step if side-20 pools are being built (that step is 2 workers of its own; QUEUE limit 4).
Run: .venv/bin/python scripts/bg.py agent/logs/side_evals.log '.venv/bin/python scripts/side_evals.py'
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
DAY = "runs/2026-10-28"
SETS = "eval_*10_*"
R = f"{DAY}/r3_loop_rl/best.pt"
STEPS = [("--methods gp", f"{DAY}/baselines")]
STEPS += [(f"--policy {n}_rl={DAY}/{n}_rl/best.pt", f"{DAY}/{n}_rl") for n in ("r3_loop", "r2_mott", "r1_modal", "r0_coord")]
STEPS += [("--policy " + " ".join(f"r3_loop_rl@R{k}={R}" for k in (2, 4, 16)), f"{DAY}/r3_loop_rl")]
STEPS += [(f"--policy {n}_distill={DAY}/{n}_distill/best.pt", f"{DAY}/{n}_distill") for n in ("r3_loop", "r2_mott", "r1_modal", "r0_coord")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


for arg, out in STEPS:
    if alive("--eval_sizes 20"):
        log("side-20 pools building, stop")
        break
    wait_mem(log)
    log(f"START {arg} | {mem_line()}")
    r = subprocess.run(f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets '{SETS}' {arg} --workers 1 --out {out}",
                       shell=True, cwd=ROOT)
    log(f"END rc={r.returncode} | {mem_line()}")
log("side evals done")
