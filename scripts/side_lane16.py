"""Heavy-job CPU lane while the GPU is idle (11/03 round 27).

Runs the `heavy` list of agent/day2_queue.json (6 `ls16 *` LS 16-cell files, 6 `base20 * greedy+prune` 20-cell g+p
files; moved out of the driver's `cpu` list by scripts/queue_heavy.py so the two lanes never pick the same job), one
job at a time, in list order. With no training running the 2 exact-check worker slots are free:
driver lane 1 + side_slot 1 + this 1 = 3 <= 4 (QUEUE 10/28 14:20).
Guards: never starts a job while any `train.py` is alive (its 2 certify workers would make 5), while the pod is above
START_GIB (16-cell g+p/LS peak rss 19-34 GB), or before the pool file of its `--sets` exists (20-cell files are still
being built). A "solo" job also waits until no other eval_policy / build_pools process is alive. Never preempted.
Run: .venv/bin/python scripts/bg.py agent/logs/side_lane16.log '.venv/bin/python scripts/side_lane16.py'
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import mem_line, wait_mem  # noqa: E402
from pipeline_day1 import alive  # noqa: E402
from pipeline_day2 import ROOT, exists, queue  # noqa: E402

START_GIB = 20.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def pool_ready(j):
    m = re.search(r"--pool (\S+) --sets (\S+)", j["cmd"])
    return m is None or exists(os.path.join(m.group(1), m.group(2) + ".pkl"))


def others_busy(j):
    me = j["cmd"].replace("'", "")
    return any(alive(p) and alive(p) != alive(me) for p in ("scripts/eval_policy.py", "scripts/build_pools.py"))


def blocked(j):
    return alive("train.py --stage") or not pool_ready(j) or (j.get("solo") and others_busy(j))


failed = set()
while True:
    jobs = [j for j in queue().get("heavy", []) if j["name"] not in failed and not all(exists(p) for p in j["done"])]
    if not jobs:
        break
    live = [j for j in jobs if alive(j["cmd"].replace("'", ""))]   # a restart of this lane mid-job: wait for it
    ready = [j for j in jobs if not blocked(j)]
    if live or not ready:
        time.sleep(120)
        continue
    j = ready[0]
    wait_mem(log, START_GIB)
    if blocked(j):
        continue
    log(f"START {j['name']} | {mem_line()}")
    with open(os.path.join(ROOT, j.get("log", "agent/logs/side_lane16_jobs.log")), "a") as f:
        rc = subprocess.run(j["cmd"], shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"END {j['name']} rc={rc} | {mem_line()}")
    if rc != 0 or not all(exists(p) for p in j["done"]):
        failed.add(j["name"])
        log(f"{j['name']} failed or incomplete, skipped")
    subprocess.run(f"{sys.executable} scripts/collect_results.py", shell=True, cwd=ROOT,
                   stdout=open(os.path.join(ROOT, "agent/logs/collect_results.log"), "a"), stderr=subprocess.STDOUT)
log("done")
