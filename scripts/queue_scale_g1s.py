"""Round 86 (11/07): QUEUE 11/07 19:14 KST / PROPOSALS "11/07 G1 보충 2" -- report-only 16 / 20-cell evals of the G1
confirmation runs `r2pkfc_s1_rl`, `r2pkfc_s2_rl` (12 files each), NOT part of the G1 verdict. Same command as the
scale-curve evals (--workers 1 --inst_chunk 5, pool_v1, best.pt), one file per entry, human order (pcb first).

The CPU queue has no "after" gate, so a run's entries are appended only once its training is done AND its 10-cell
eval (G1 verdict, always first) has all its records: with --wait the script polls (120 s) until then for each named
run in turn, appends, and exits after the last. Names start with "ls10 scale-eval" (the 4th slot picks them;
scripts/side_slot.py LS16_EXEMPT lets them run while an LS16 job is alive, mem_guard stops LS16 -- human 13:40 rule).
Idempotent; atomic queue write.
Run: .venv/bin/python scripts/bg.py agent/logs/queue_scale_g1s.log \
       '.venv/bin/python scripts/queue_scale_g1s.py --wait r2pkfc_s1 r2pkfc_s2'
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
DAY = "2026-10-29"
SETS = [f"eval_{t}{n}_{k}" for t in ("pcb", "ip", "pkg") for n in (16, 20) for k in (0, 1)]


def ready(run):
    """training done and every 10-cell eval record (the driver's eval10 job) present"""
    d = os.path.join(ROOT, f"runs/{DAY}/{run}_rl")
    sets = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, "runs/pool_v1/eval_*10_*.pkl")))
    return (os.path.exists(f"{d}/metrics.json") and os.path.exists(f"{d}/best.pt") and len(sets) == 6
            and all(os.path.exists(f"{d}/eval/{s}__{run}_rl.json") for s in sets))


def job(run, s):
    n, d = f"{run}_rl", f"runs/{DAY}/{run}_rl"
    return dict(name=f"ls10 scale-eval {s} {n}",
                cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets {s} --policy {n}={d}/best.pt "
                    f"--workers 1 --inst_chunk 5 --out {d}",
                done=[f"{d}/eval/{s}__{n}.json"], log=f"{d}/eval.log")


def append(run):
    q = json.load(open(Q))
    names = {e["name"] for e in q["cpu"]}
    new = [e for e in (job(run, s) for s in SETS) if e["name"] not in names]
    q["cpu"] += new
    tmp = Q + ".tmp"
    open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, Q)
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {run}: appended {len(new)}", flush=True)


def main():
    wait = sys.argv[1] == "--wait"
    for run in sys.argv[2:] if wait else sys.argv[1:]:
        while not ready(run):
            if not wait:
                sys.exit(f"{run}: training or its 10-cell eval not done")
            time.sleep(120)
        append(run)


if __name__ == "__main__":
    main()
