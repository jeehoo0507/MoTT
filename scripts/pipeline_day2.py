"""Day-2 driver (QUEUE 2026-10-29: architecture-only improvements, prior-work baselines, seeds), launched
detached (scripts/bg.py); scripts/ensure_running.py restarts it after a pod restart.

The work list is agent/day2_queue.json, re-read before every step, so a round appends entries without
restarting this process:
  "train": [{"name", "arm", "seed"?, "after"?}]  -> scripts/run_rung.py (distill 0.75 h + rl 1.25 h, the ladder's
            COMMON args) into runs/<DAY>/<name>_{distill,rl}; one at a time on the GPU, CERTIFY_WORKERS=2.
            "after": a path that must exist first (e.g. code smoke record); missing -> entry skipped for now.
            "rl_only": true -> no distill stage, rl from random init for distill + rl hours (B-Manoharan, 11/02).
            "init": path -> no distill stage, rl (rl hours) from that best.pt: a one-change rl re-run of a rung (11/02).
  "cpu":   [{"name", "cmd", "done": [paths]}]      -> run in order when every "done" path is missing.
CPU lane (1 worker): first the side-10 certified eval of every finished training in "train" (rl best.pt,
eval_*10_*, --inst_chunk 5), then the "cpu" entries in order. It waits while an external job named in
EXTERNAL is alive (the side-10 single-type greedy+prune analysis holds that slot until it ends).
CPU workers: 2 exact-check workers while training + 1 here + 1 eval-pool builder / scripts/side16_gp.py = 4
(QUEUE 10/28 14:20); no new process above 45 GiB (mem.wait_mem). Every step skips existing outputs.
Run: .venv/bin/python scripts/bg.py agent/logs/pipeline_day2.log '.venv/bin/python scripts/pipeline_day2.py'
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import mem_line, wait_mem  # noqa: E402
from pipeline_day1 import alive  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
POOL = "runs/pool_v1"
DAY = "2026-10-29"
QUEUE = os.path.join(ROOT, "agent/day2_queue.json")
DISTILL_H, RL_H = 0.75, 1.25
SETS10 = "eval_*10_*"
INST_CHUNK = 5
EXTERNAL = ("analyze_gp_one_type.py --sets eval_pcb10_0",)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def queue():
    try:
        return json.load(open(QUEUE))
    except (OSError, ValueError) as e:
        log(f"queue unreadable ({e}), retry")
        return {"train": [], "cpu": []}


def run(cmd, logfile, stage, env=None):
    wait_mem(log)
    log(f"START {stage} | {mem_line()}")
    log(f"$ {cmd}  (log {logfile})")
    os.makedirs(os.path.dirname(os.path.join(ROOT, logfile)), exist_ok=True)
    with open(os.path.join(ROOT, logfile), "a") as f:
        r = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, **(env or {})))
    log(f"END {stage} rc={r.returncode} | {mem_line()}")
    return r.returncode


def exists(p):
    return os.path.exists(os.path.join(ROOT, p))


def rl_done(name):
    return exists(f"runs/{DAY}/{name}_rl/metrics.json")


FAILED = set()


def gpu_lane():
    while True:
        todo = [t for t in queue()["train"] if not rl_done(t["name"]) and t["name"] not in FAILED
                and (not t.get("after") or exists(t["after"]))]
        if not todo:
            time.sleep(120)
            continue
        t = todo[0]
        name = t["name"]
        attach = f"run_rung.py --name {name} "
        if alive(attach):
            log(f"ATTACH train {name}")
            while alive(attach):
                time.sleep(60)
        else:
            os.makedirs(os.path.join(ROOT, f"runs/{DAY}/{name}"), exist_ok=True)
            # "rl_only": no distillation, rl from random init for the whole budget (same total hours)
            hours = (f"--distill_hours 0 --rl_hours {DISTILL_H + RL_H} --skip_distill" if t.get("rl_only")
                     else f"--distill_hours 0 --rl_hours {RL_H} --init_from {t['init']}" if t.get("init")
                     else f"--distill_hours {DISTILL_H} --rl_hours {RL_H}")
            rc = run(f"{PY} scripts/run_rung.py --name {name} --arm '{t['arm']}' --day {DAY} --pool {POOL} "
                     f"--seed {t.get('seed', 0)} {hours} --skip_eval",
                     f"runs/{DAY}/{name}/rung_stdout.log", f"train {name} (CERTIFY_WORKERS=2)",
                     env={"CERTIFY_WORKERS": "2"})
            if rc != 0:
                FAILED.add(name)   # not retried by this process (a restart retries; resume.pt continues it)
                log(f"train {name} failed rc={rc}, skipped until driver restart")


def eval_jobs():
    """side-10 certified eval of every finished training, in queue order."""
    jobs = []
    for t in queue()["train"]:
        n = t["name"]
        run_dir = f"runs/{DAY}/{n}_rl"
        if not rl_done(n):
            continue
        sets = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, POOL, SETS10 + ".pkl")))
        done = [f"{run_dir}/eval/{s}__{n}_rl.json" for s in sets]
        cmd = (f"{PY} scripts/eval_policy.py --pool {POOL} --sets '{SETS10}' --policy {n}_rl={run_dir}/best.pt "
               f"--workers 1 --inst_chunk {INST_CHUNK} --out {run_dir}")
        jobs.append(dict(name=f"eval10 {n}_rl", cmd=cmd, done=done, log=f"{run_dir}/eval.log"))
    return jobs


def external_busy():
    for _ in range(3):
        if any(alive(p) for p in EXTERNAL):
            return True
        time.sleep(30)
    return False


def cpu_lane():
    while True:
        if external_busy():
            time.sleep(120)
            continue
        jobs = eval_jobs() + queue().get("cpu", [])
        todo = [j for j in jobs if not all(exists(p) for p in j["done"]) and j["name"] not in FAILED]
        if not todo:
            time.sleep(120)
            continue
        j = todo[0]
        attach = j["cmd"].replace("'", "")     # the child's command line (the shell strips the quotes)
        if alive(attach):                      # a driver restarted while this job lives: wait, don't rerun it
            log(f"ATTACH {j['name']}")
            while alive(attach):
                time.sleep(60)
            rc = 0
        else:
            rc = run(j["cmd"], j.get("log", "agent/logs/day2_cpu.log"), j["name"])
        if rc != 0:
            FAILED.add(j["name"])
            log(f"{j['name']} failed rc={rc}, skipped until driver restart")
        elif not all(exists(p) for p in j["done"]):
            FAILED.add(j["name"])
            log(f"{j['name']} finished without all outputs {j['done']}, skipped until driver restart")
        run(f"{PY} scripts/collect_results.py", "agent/logs/collect_results.log", "collect_results")


def main():
    log(f"driver day2 start | {mem_line()}")
    lanes = [threading.Thread(target=gpu_lane, daemon=True), threading.Thread(target=cpu_lane, daemon=True)]
    for t in lanes:
        t.start()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
