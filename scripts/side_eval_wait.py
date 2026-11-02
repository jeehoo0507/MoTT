"""Side-10 eval of named trainings on the 4th CPU slot, as soon as each one's rl stage ends (11/02).

The driver's CPU lane runs one job at a time and a 16-cell E0+ job holds it for 4-7 h, so a training that ends
meanwhile would wait that long for its eval. This runs the driver's exact eval command (pipeline_day2.eval_jobs:
same records, same --inst_chunk), one worker; the driver attaches to it if it reaches the job while it lives
and skips it once the records exist. Slots: 2 exact-check workers (training) + driver lane 1 + this 1 = 4
(side16_gp is not running).
Run: .venv/bin/python scripts/bg.py agent/logs/side_eval_wait.log '.venv/bin/python scripts/side_eval_wait.py lo_pcb_r3 ...'
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import mem_line, wait_mem  # noqa: E402
from pipeline_day1 import alive  # noqa: E402
from pipeline_day2 import ROOT, PY, eval_jobs, exists, rl_done  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


for name in sys.argv[1:]:
    log(f"wait {name}_rl")
    while not rl_done(name):
        time.sleep(120)
    job = next((j for j in eval_jobs() if j["name"] == f"eval10 {name}_rl"), None)
    if job is None:
        log(f"{name} is not in the train queue, skip")
        continue
    if all(exists(p) for p in job["done"]):
        log(f"{name}_rl records exist, skip")
        continue
    if alive(job["cmd"].replace("'", "")):
        log(f"driver runs {name}_rl eval, skip")
        continue
    wait_mem(log)
    log(f"START {job['name']} | {mem_line()}")
    with open(os.path.join(ROOT, job["log"]), "a") as f:
        rc = subprocess.run(job["cmd"], shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"END {job['name']} rc={rc} | {mem_line()}")
    subprocess.run(f"{PY} scripts/collect_results.py", shell=True, cwd=ROOT,
                   stdout=open(os.path.join(ROOT, "agent/logs/collect_results.log"), "a"), stderr=subprocess.STDOUT)
log("done")
