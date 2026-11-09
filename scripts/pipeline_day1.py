"""Day-1 driver, launched detached (scripts/bg.py) so it survives the agent round ending.

Limits follow agent/QUEUE.md (2026-10-28 14:20 UTC, pod OOM restarts): at most 4 concurrent CPU worker
processes (pool build + exact checks + teacher), no new process while the pod is above 45 GiB, memory
logged at every stage start/end, the 20x20 eval pools last with 2 processes.
Rungs train one at a time: two train.py processes with all 27 pools dense on the GPU do not fit in
24 GB (CUDA OOM at load, 15:15 UTC 10/28).

  0. teacher: make_teacher.py, 4 workers (skips finished files)          -> data/teacher/
  GPU lane : r0 coord -> r1 modal -> r2 mott -> r3 loop, each distill -> rl, CERTIFY_WORKERS=2   (2 workers)
             (r3 only after scripts/smoke_pipeline.py --arms loop passes, run in this lane: still 2 workers)
  CPU lane : eval pools side 10/16 (once the GPU lane is done: rung evals on side 16, rl rungs first;
             side 10 by scripts/side_evals.py, side-16 gp/modal_rule by scripts/side16_gp.py, 1 worker each)
             -> E0 (greedy+prune, modal rule; LS side 10) -> certified eval of each rung
             as soon as it finishes, side-16 LS one kind at a time while no rung is waiting (r3 rl also at test-time R = 2, 4, 16) -> collect_results -> eval pools side 20 -> everything on side 20
             -> collect_results                                                     (2 workers)
Every step skips work whose output already exists (pool/teacher files, eval records, finished
training stages; a killed training stage continues from its resume.pt), and a step whose process is
still alive from a previous driver is waited for instead of started twice, so after a pod restart
scripts/ensure_running.py just starts this again.
Run: .venv/bin/python scripts/bg.py agent/logs/pipeline_day1.log '.venv/bin/python scripts/pipeline_day1.py'
"""
import glob
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mem import mem_line, wait_mem  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
POOL = "runs/pool_v1"
DAY = "2026-10-28"
N_TRAINVAL, N_EVAL20 = 30, 6     # side-20 eval pools: 3 kinds x 2 files
DISTILL_H, RL_H = 0.75, 1.25
RUNGS = (("r0_coord", "--feat coord"), ("r1_modal", "--feat modal"), ("r2_mott", "--model mott"), ("r3_loop", "--model loop"))
LOOP_R = (2, 4, 16)              # rung 3 extra test-time loop counts (training R ~ U{2..8}, default eval R = 8)
INST_CHUNK = 5                   # eval_policy --inst_chunk for the driver's policy evals (memory, same solutions)
W16 = 1                          # side-16 rung-eval workers until a record's peak_rss_mb shows 2 fit (10/29)
PREV_EVAL = ("--workers 2 --out runs/", f"--inst_chunk {INST_CHUNK} --out runs/")
DONE = {n: threading.Event() for n, _ in RUNGS}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def n_pools(pattern):
    return len(glob.glob(os.path.join(ROOT, POOL, pattern)))


def alive(pat):
    """pid of a live process whose command line contains pat (a driver restarted while its children live)."""
    me = str(os.getpid())
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if pid == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pat in cmd and "bash -c" not in cmd and "/bin/sh -c" not in cmd:
            return pid
    return None


def run(cmd, logfile, stage, attach=None, env=None):
    """Run cmd to completion; if a process matching `attach` is already alive, wait for it instead."""
    pid = alive(attach) if attach else None
    if pid:
        log(f"ATTACH {stage} (pid {pid} still running)")
        while alive(attach):
            time.sleep(60)
        log(f"END {stage} (attached) | {mem_line()}")
        return 0
    wait_mem(log)
    log(f"START {stage} | {mem_line()}")
    log(f"$ {cmd}  (log {logfile})")
    with open(os.path.join(ROOT, logfile), "a") as f:
        r = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, **(env or {})))
    log(f"END {stage} rc={r.returncode} | {mem_line()}")
    return r.returncode


def gpu_lane():
    # A rung whose run_rung.py is still alive (driver restarted) goes first, so two pool-loaded train.py never
    # share the GPU; the rest keep ladder order. A finished distill is kept and an unfinished rl stage is redone
    # (r0 rl hit CUDA OOM at 16:23 10/28 before train.py --chunk; it reruns after r1).
    for name, arm in sorted(RUNGS, key=lambda r: alive(f"run_rung.py --name {r[0]} ") is None):
        if os.path.exists(os.path.join(ROOT, f"runs/{DAY}/{name}_rl/metrics.json")):
            log(f"{name}: rl stage finished, skip training")
        elif name == "r3_loop" and not os.path.exists(os.path.join(ROOT, f"runs/{DAY}/{name}_distill")) and \
                run(f"{PY} scripts/smoke_pipeline.py --arms loop --workers 2", "agent/logs/smoke_loop.log",
                    "smoke r3 loop (tiny boards)", env={"CERTIFY_WORKERS": "2"}) != 0:
            log("r3_loop smoke failed, skip rung 3 (see agent/logs/smoke_loop.log)")
        else:
            os.makedirs(os.path.join(ROOT, f"runs/{DAY}/{name}"), exist_ok=True)
            run(f"{PY} scripts/run_rung.py --name {name} --arm '{arm}' --day {DAY} --pool {POOL} "
                f"--distill_hours {DISTILL_H} --rl_hours {RL_H} --skip_eval", f"runs/{DAY}/{name}/rung_stdout.log",
                f"train {name} (CERTIFY_WORKERS=2)", attach=f"run_rung.py --name {name} ", env={"CERTIFY_WORKERS": "2"})
        DONE[name].set()


def eval_methods(stage, methods="gp,ls,modal_rule", sets="eval_*"):
    run(f"{PY} scripts/eval_policy.py --pool {POOL} --sets '{sets}' --methods {methods} --workers 2 "
        f"--out runs/{DAY}/baselines", f"runs/{DAY}/baselines_eval.log", f"E0 {stage}",
        attach=f"--workers 2 --out runs/{DAY}/baselines")


def eval_rung(name, sets="eval_*", tags=("distill", "rl"), loop_r=True, workers=2):
    # --inst_chunk 5: side-16 policy jobs peaked at ~22 GB per worker with all 25 instances in one rollout
    # (10/29 01:50, pod 51.6 GiB with 2 of them); greedy decoding per instance, so the same solutions
    for tag in tags:
        run_dir = f"runs/{DAY}/{name}_{tag}"
        if os.path.exists(os.path.join(ROOT, run_dir, "metrics.json")):
            pols = f"{name}_{tag}={run_dir}/best.pt"
            if name == "r3_loop" and tag == "rl" and loop_r:
                pols += "".join(f" {name}_{tag}@R{R}={run_dir}/best.pt" for R in LOOP_R)
            run(f"{PY} scripts/eval_policy.py --pool {POOL} --sets '{sets}' --policy {pols} "
                f"--workers {workers} --inst_chunk {INST_CHUNK} --out {run_dir}", f"{run_dir}/eval.log",
                f"eval {name}_{tag}" + ("" if sets == "eval_*" else f" ({sets}, {workers} workers)"),
                attach=f"--inst_chunk {INST_CHUNK} --out {run_dir}")
        else:
            log(f"{run_dir} not finished (no metrics.json), skip eval")


def build_pools(sizes):
    run(f"{PY} scripts/build_pools.py --out {POOL} --workers 2 --only eval --eval_sizes {sizes}",
        "agent/logs/build_pools_v1_r3.log", f"build eval pools side {sizes}", attach="scripts/build_pools.py")


def collect():
    run(f"{PY} scripts/collect_results.py", "agent/logs/collect_results.log", "collect_results")


def built_eval_sets(pattern="eval_*.json"):
    """eval pool files whose .json is written (build_pools writes it after the .pkl), comma-joined for --sets."""
    return ",".join(sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(ROOT, POOL, pattern))
                           if os.path.exists(f[:-5] + ".pkl")))


def cpu_lane():
    # While the last side-10/16 pools build (2 workers), the GPU lane's exact checks hold the other 2 slots.
    # Once every rung has trained those slots are idle (10/28: r3 done ~00:25, ip16 pools ~05:00), so use them
    # for the rung evals and E0 gp/modal_rule on the pools already built. Every record is skipped when it
    # exists, so the full passes below only add the missing pools.
    builder = threading.Thread(target=build_pools, args=("10,16",))
    builder.start()
    while builder.is_alive() and not all(e.is_set() for e in DONE.values()):
        time.sleep(60)
    pid = next(filter(None, map(alive, PREV_EVAL)), None)   # an eval started by the previous driver keeps its slots
    if pid:
        log(f"ATTACH eval from previous driver (pid {pid})")
        while any(map(alive, PREV_EVAL)):
            time.sleep(60)
    if all(e.is_set() for e in DONE.values()):
        # 10/29 01:40: a side-16 policy job takes > 75 min, so one rung stage on the side-16 pools is hours.
        # Side 10 is covered by scripts/side_evals.py (1 slot), greedy+prune and modal rule on side 16 by
        # scripts/side16_gp.py (the slot the pool builder frees), so here: side 16 only, rl rungs first (the
        # ladder and figure 2 use them), then distill, then r3 at test-time R.
        sets = built_eval_sets("eval_*16_*.json")
        log(f"rung evals on side 16 (side 10 in side_evals.py, gp in side16_gp.py): {sets}")
        for tags, loop_r in ((("rl",), False), (("distill",), False), (("rl",), True)):
            for name, _ in RUNGS:
                if loop_r and name != "r3_loop":
                    continue
                eval_rung(name, sets, tags, loop_r, workers=W16)
        collect()
    builder.join()
    # LS on side 16 (4x32 restarts inside cutting-plane) is by far the slowest eval; run it one kind at a
    # time only while no finished rung is waiting, so rung evals are not queued behind it.
    eval_methods("side 10/16 gp,modal_rule", "gp,modal_rule")
    eval_methods("side 10 ls", "ls", "eval_*10_*")
    ls16 = ["eval_pcb16_*", "eval_pkg16_*", "eval_ip16_*"]
    pending = [n for n, _ in RUNGS]
    while pending or ls16:
        ready = [n for n in pending if DONE[n].is_set()]
        if ready:
            eval_rung(ready[0])
            pending.remove(ready[0])
            collect()
        elif ls16:
            eval_methods(f"side 16 ls {ls16[0]}", "ls", ls16.pop(0))
            collect()
        else:
            time.sleep(60)
    build_pools("20")
    if n_pools("eval_*20_*.pkl") < N_EVAL20:
        log("side-20 eval pools incomplete")
    eval_methods("side 20")
    for name, _ in RUNGS:
        eval_rung(name)
    collect()


def main():
    log(f"driver start | {mem_line()}")
    while n_pools("train_*.pkl") + n_pools("val_*.pkl") < N_TRAINVAL:
        time.sleep(60)
    run(f"{PY} scripts/make_teacher.py --pool {POOL} --out data/teacher --workers 4", "agent/logs/make_teacher.log",
        "teacher", attach="scripts/make_teacher.py")
    if len(glob.glob(os.path.join(ROOT, "data/teacher/train_*.npz"))) < n_pools("train_*.pkl"):
        log("teacher incomplete, stop (training would silently fall back to greedy+prune)")
        return
    lanes = [threading.Thread(target=gpu_lane), threading.Thread(target=cpu_lane)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    log("PIPELINE DAY1 DONE")


if __name__ == "__main__":
    main()
