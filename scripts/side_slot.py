"""4th CPU slot (11/02 round 4): side_eval_wait plus filler jobs between training evals.

Priority 1: the driver's exact 10-cell eval of each named training as soon as its rl stage ends (same as
side_eval_wait.py). Priority 2, only while no named training has a pending eval: the first unfinished CPU-queue
entry whose name starts with one of the --fill prefixes (default "ls10 ", the per-set 10-cell local search
baseline). One job at a time, never preempted, so a training eval waits at most one filler job.
Slots: 2 exact-check workers (training) + driver lane 1 + this 1 = 4. The driver attaches to a live identical job
and skips a job once its records exist.
Run: .venv/bin/python scripts/bg.py agent/logs/side_slot.log '.venv/bin/python scripts/side_slot.py r2c lo_pcb_r2c'
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import mem_line, wait_mem  # noqa: E402
from pipeline_day1 import alive  # noqa: E402
from pipeline_day2 import ROOT, PY, eval_jobs, exists, queue, rl_done  # noqa: E402

FILL = ("ls10 ", "ga10 ")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def pending(j):
    return not all(exists(p) for p in j["done"]) and not alive(j["cmd"].replace("'", ""))


def run(j):
    wait_mem(log)
    log(f"START {j['name']} | {mem_line()}")
    with open(os.path.join(ROOT, j.get("log", "agent/logs/side_slot_jobs.log")), "a") as f:
        rc = subprocess.run(j["cmd"], shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"END {j['name']} rc={rc} | {mem_line()}")
    subprocess.run(f"{PY} scripts/collect_results.py", shell=True, cwd=ROOT,
                   stdout=open(os.path.join(ROOT, "agent/logs/collect_results.log"), "a"), stderr=subprocess.STDOUT)
    return rc


targets = sys.argv[1:]
failed = set()
while True:
    evals = {j["name"]: j for j in eval_jobs()}
    mine = [evals[f"eval10 {n}_rl"] for n in targets if f"eval10 {n}_rl" in evals]
    mine += [j for j in queue().get("cpu", []) if j["name"].startswith(FILL)]
    if any(alive(j["cmd"].replace("'", "")) for j in mine):   # a restart mid-job, or the driver runs one: one at a time
        time.sleep(120)
        continue
    todo = [evals[f"eval10 {n}_rl"] for n in targets if rl_done(n) and f"eval10 {n}_rl" in evals]
    todo = [j for j in todo if pending(j) and j["name"] not in failed]
    if not todo:
        todo = [j for j in queue().get("cpu", []) if j["name"].startswith(FILL) and pending(j)
                and j["name"] not in failed]
    if todo:
        if run(todo[0]) != 0 or not all(exists(p) for p in todo[0]["done"]):
            failed.add(todo[0]["name"])
            log(f"{todo[0]['name']} failed or incomplete, skipped")
        continue
    evaluated = all(rl_done(n) and all(exists(p) for p in evals.get(f"eval10 {n}_rl", {"done": []})["done"])
                    for n in targets)
    fill_left = any(j["name"].startswith(FILL) and not all(exists(p) for p in j["done"]) and j["name"] not in failed
                    for j in queue().get("cpu", []))
    if evaluated and not fill_left:
        break
    time.sleep(120)
log("done")
