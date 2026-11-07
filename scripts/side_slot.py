"""4th CPU slot (11/02 round 4): side_eval_wait plus filler jobs between training evals.

Priority 1: the driver's exact 10-cell eval of each named training as soon as its rl stage ends (same as
side_eval_wait.py). Priority 2, only while no named training has a pending eval: the first unfinished CPU-queue
entry whose name starts with one of the FILL prefixes ("ls10 "/"ga10 ": per-set 10-cell LS/GA baselines,
"samp10 ": best-of-8 policy sampling). One job at a time, never preempted, so a training eval waits at most one filler job.
Slots: 2 exact-check workers (training) + driver lane 1 + this 1 = 4. The driver attaches to a live identical job
and skips a job once its records exist.
Run: .venv/bin/python scripts/bg.py agent/logs/side_slot.log '.venv/bin/python scripts/side_slot.py r2c lo_pcb_r2c'
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import mem_line, wait_mem  # noqa: E402
from pipeline_day1 import alive  # noqa: E402
from pipeline_day2 import DAY, ROOT, PY, eval_jobs, exists, queue, rl_done  # noqa: E402

FILL = ("ls10 ", "ga10 ", "samp10 ")   # samp10: best-of-8 policy sampling (PROGRESS 11/02 round 6)
# 11/07 round 99: F1 coord pkg20 beside the driver's F1 pkg16 (QUEUE 11/08 01:45 KST, allowed below 45 GiB) put the
# pod at 45.7 GiB (pkg16 repair worker 22 GB) -> not a filler; the driver runs it in queue order
FILL_GAP_S = 3600   # no filler job when a target's rl stage has less than this left (11/02: ls pcb10_1 took the
                    # slot 1 min before bm_manoharan_rlonly_l20 ended)
LS16 = ("--methods ls --ls_chunk",)   # 11/05 round 50: a 16/20-cell "scale-eval" filler start (~+22 GiB in 30 s) put the
                                      # pod over 45 GiB and mem_guard killed ls16c pcb16_0 (4 h in) and pcb16_1 -> no
                                      # scale-eval filler while a driver LS16 job is alive
LS16_EXEMPT = ("r2pkfc_rl", "r2pkfc_s1_rl", "r2pkfc_s2_rl")   # QUEUE 11/07 13:40 / 19:14 KST: these scale-evals run
                                                               # while LS16 is alive; mem_guard stops LS16
SCALE_GAP_S = 1800   # scale-eval filler takes 7-15 min (side_slot.log 11/06), so a 30 min gap keeps the G1 eval first


def ls16_alive():
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                continue
            if "eval_policy.py" in cmd and any(s in cmd for s in LS16) and "16_" in cmd:
                return True
    return False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def rl_left_s(name):
    """seconds the named training's live rl stage has left (time budget minus the last logged elapsed), else None"""
    d = os.path.join(ROOT, f"runs/{DAY}/{name}_rl")
    if rl_done(name) or not os.path.exists(os.path.join(d, "log.txt")):
        return None
    try:
        hours = json.load(open(os.path.join(d, "config.json")))["args"]["hours"]
        el = [int(m) for m in re.findall(r"\| (\d+)s \|", open(os.path.join(d, "log.txt")).read())]
    except (OSError, ValueError, KeyError):
        return None
    return hours * 3600 - (el[-1] if el else 0)


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
    left = [s for s in (rl_left_s(n) for n in targets) if s is not None]
    if not todo:   # a filler (<= ~50 min; scale-eval <= ~15 min) must not hold a training's eval
        todo = [j for j in queue().get("cpu", []) if j["name"].startswith(FILL) and pending(j)
                and j["name"] not in failed
                and not any(s < (SCALE_GAP_S if "scale-eval" in j["name"] else FILL_GAP_S) for s in left)
                and not ("scale-eval" in j["name"] and not any(x in j["name"] for x in LS16_EXEMPT)
                         and ls16_alive())]
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
