"""Relaunch the detached day jobs if a workspace restart killed them (happened 06:41 and ~13:45 UTC on 10/28).

Every job it starts skips finished outputs, so running this when the jobs are alive is a no-op and
running it after a restart resumes where the files left off. Run at the start of every round:
  .venv/bin/python scripts/ensure_running.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = ".venv/bin/python"
JOBS = [  # (pattern in cmdline, log, command)
    # 10/29 03:10 QUEUE day 2: the day-2 driver replaces pipeline_day1.py (its remaining side-16 rung evals are
    # "cpu" entries of agent/day2_queue.json). The last side-16 eval pool, then side-16 greedy+prune, keep their
    # own slot: side16_gp.py waits for build_pools.py, which only runs if a pool file is missing.
    ("scripts/pipeline_day2.py", "agent/logs/pipeline_day2.log", f"{PY} scripts/pipeline_day2.py"),
    ("scripts/side16_gp.py", "agent/logs/side16_gp.log", f"{PY} scripts/side16_gp.py"),
    # QUEUE 10/29 14:20: commits RESULTS.md + new eval records every 30 min after the loop stops (exits by itself
    # once the lo_* rungs are evaluated and pushed; a relaunch after that does one cycle and exits)
    ("scripts/report_watch.py", "agent/logs/report_watch.log", f"{PY} scripts/report_watch.py"),
    # QUEUE 10/31 11:40: restarts the agent loop when the GPU queue is done, or after a restart if STOP is gone
    ("scripts/resume_agent.py", "agent/logs/resume_agent.log", f"{PY} scripts/resume_agent.py"),
]
POOL16 = [f"runs/pool_v1/eval_{k}16_{i}.json" for k in ("pcb", "pkg", "ip") for i in (0, 1)]
GP16 = [f"runs/2026-10-28/baselines/eval/eval_{k}16_{i}__{m}.json" for k in ("pcb", "pkg", "ip") for i in (0, 1)
        for m in ("greedy+prune", "modal_rule")]
BUILD16 = f"{PY} scripts/build_pools.py --out runs/pool_v1 --workers 1 --only eval --eval_sizes 16"


def running(pat):
    me = str(os.getpid())
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if pid == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pat in cmd and "bash -c" not in cmd:
            return pid
    return None


def have(paths):
    return all(os.path.exists(os.path.join(ROOT, p)) for p in paths)


def main():
    if not have(POOL16) and not running("scripts/build_pools.py"):
        out = subprocess.check_output([sys.executable, "scripts/bg.py", "agent/logs/build_pools_v1_r3.log", BUILD16],
                                      cwd=ROOT, text=True).strip()
        print(f"started build_pools side 16 (pid {out})")
    for pat, log, cmd in JOBS:
        if pat == "scripts/side16_gp.py" and have(GP16):
            continue
        pid = running(pat)
        if pid:
            print(f"alive   {pat} (pid {pid})")
            continue
        out = subprocess.check_output([sys.executable, "scripts/bg.py", log, cmd], cwd=ROOT, text=True).strip()
        print(f"started {pat} (pid {out}) -> {log}")


if __name__ == "__main__":
    main()
