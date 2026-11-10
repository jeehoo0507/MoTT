"""PROPOSALS 11/09 U: build fresh 16-cell speed problems (never solved by any method before this run).
eval_{tech}16_{70+k}.pkl in runs/pool_speed (own dir: the driver's eval globs never see it), k = 0..9, 5 problems each (50 per kind), seed = SEED[tech] + 70000 + k, same generator,
difficulty and greedy+prune certification as build_pools.py. Every built problem is used.
  .venv/bin/python scripts/speed_fresh.py --out runs/pool_speed --workers 6"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pools import SEED, build   # noqa: E402

TECHS, K, N, SIZE = ("pkg", "pcb", "ip"), 10, 5, 16

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pool_speed")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="build just this one file in this process (no Pool; 11/10: the Pool "
                    "build hung at 0 %% CPU on server 2)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)   # 11/10: the first run died here (save_bank into a missing dir)
    js = [(a.out, f"eval_{t}{SIZE}_{70 + k}", t, SIZE, N, SEED[t] + 70000 + k) for t in TECHS for k in range(K)]
    t0 = time.time()
    if a.only:
        name, status, dt, meta = build(next(j for j in js if j[1] == a.only))
        extra = "" if meta is None else f"gp ok {meta['gp_ok']:.2f} rounds {meta['rounds']}"
        print(f"[{time.time()-t0:6.0f}s] {name} {status} {dt:.0f}s {extra}", flush=True)
        sys.exit(0)
    with Pool(a.workers) as p:   # pkg first (slowest)
        for name, status, dt, meta in p.imap_unordered(build, js):
            extra = "" if meta is None else f"gp ok {meta['gp_ok']:.2f} rounds {meta['rounds']}"
            print(f"[{time.time()-t0:6.0f}s] {name} {status} {dt:.0f}s {extra}", flush=True)
    print("built", flush=True)
