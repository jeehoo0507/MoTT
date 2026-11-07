"""Run one ladder rung end to end: stage 2 (distill) -> stage 3 (rl, init from the distilled best.pt)
-> certified evaluation of both checkpoints on the eval pools. One process; meant to be launched in
the background, one per rung (several rungs can share the GPU).

Run: .venv/bin/python scripts/run_rung.py --name r1_modal --arm "--feat modal" --day 2026-10-28 \
         --distill_hours 0.75 --rl_hours 1.25 [--common "..."] [--pool runs/pool_v1] [--skip_eval]
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
COMMON = "--d 128 --layers 4 --heads 8 --B 16 --M 8 --lr 3e-4 --iters 1000000 --val_every 100 --max_steps 64"


def sh(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n")
        f.flush()
        t = time.time()
        r = subprocess.run(shlex.split(cmd), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"[rc={r.returncode} {time.time()-t:.0f}s]\n")
    if r.returncode != 0:
        raise SystemExit(f"failed: {cmd} (see {log})")


def fresh(run_dir):
    """True if the stage must run. A finished stage (metrics.json) is kept; a partial one with a
    resume.pt (train.py, every 20 min) is continued in place; a partial one without it is moved aside
    so its stale best.pt cannot be mistaken for a result."""
    path = os.path.join(ROOT, run_dir)
    if os.path.exists(os.path.join(path, "metrics.json")):
        return False
    if os.path.exists(path) and not os.path.exists(os.path.join(path, "resume.pt")):
        os.rename(path, f"{path}_killed_{time.strftime('%H%M%S', time.gmtime())}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", required=True, help='e.g. "--feat coord" | "--feat modal" | "--model mott" | "--model loop"')
    ap.add_argument("--day", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--pool", default="runs/pool_v1")
    ap.add_argument("--teacher_dir", default="data/teacher")
    ap.add_argument("--common", default=COMMON)
    ap.add_argument("--distill_hours", type=float, default=0.75)
    ap.add_argument("--rl_hours", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_sets", default="eval_*")
    ap.add_argument("--eval_workers", type=int, default=6)
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_distill", action="store_true")
    ap.add_argument("--init_from", default="", help="no distill stage; rl starts from this best.pt (an existing distill)")
    a = ap.parse_args()
    base = f"runs/{a.day}/{a.name}"
    os.makedirs(base, exist_ok=True)
    log = os.path.join(base, "driver.log")
    # 11/03 round 27: an arm may carry its own --teacher_dir (teacher-store change); argparse keeps the last value,
    # so the default must not follow it
    tdir = "" if "--teacher_dir" in a.arm.split() else f"--teacher_dir {a.teacher_dir} "
    # 11/03 round 34 (PROPOSALS D2, diagnosis): an arm token "--distill_iters N" ends the distill stage at N
    # iterations instead of the 0.75 h limit (MoTT gets ~1600 iterations in 0.75 h, the grid rungs ~2200); it is
    # removed from the arm, so the rl stage is unchanged
    dist_extra = ""
    toks = a.arm.split()
    if "--distill_iters" in toks:
        i = toks.index("--distill_iters")
        dist_extra = f" --iters {int(toks[i + 1])}"
        a.distill_hours = max(a.distill_hours, 3.0)   # the iteration cap ends it, not the clock
        a.arm = " ".join(toks[:i] + toks[i + 2:])
    # PROPOSALS 11/07 R: an arm token "--rl_pool_glob G" gives the rl stage its own --pool_glob (appended after the
    # arm, argparse keeps the last value); the distill stage keeps the arm's --pool_glob (teacher pools unchanged)
    rl_extra = ""
    toks = a.arm.split()
    if "--rl_pool_glob" in toks:
        i = toks.index("--rl_pool_glob")
        rl_extra = f" --pool_glob {toks[i + 1]}"
        a.arm = " ".join(toks[:i] + toks[i + 2:])
    pools = f"--pool {a.pool} {tdir}--seed {a.seed}"
    dist, rl = f"{base}_distill", f"{base}_rl"
    if a.init_from:   # 11/02: a one-change rl re-run paired with an existing rung shares its distilled init
        a.skip_distill = True
    if not a.skip_distill and fresh(dist):
        sh(f"{PY} train.py --stage distill {a.arm} {a.common}{dist_extra} {pools} --hours {a.distill_hours} --out {dist}", log)
    if fresh(rl):
        init = "" if a.skip_distill else f"--init {dist}/best.pt "   # --skip_distill: rl from random init
        if a.init_from:
            init = f"--init {a.init_from} "
        sh(f"{PY} train.py --stage rl {a.arm} {a.common} {pools} --hours {a.rl_hours} {init}--out {rl}{rl_extra}", log)
    if not a.skip_eval:
        for tag, run in ((("distill", dist),) if not a.skip_distill else ()) + (("rl", rl),):
            sh(f"{PY} scripts/eval_policy.py --pool {a.pool} --sets {a.eval_sets} --policy {a.name}_{tag}={run}/best.pt "
               f"--workers {a.eval_workers} --out {run}", log)
    with open(log, "a") as f:
        f.write("RUNG DONE\n")


if __name__ == "__main__":
    main()
