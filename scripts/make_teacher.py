"""Stage 1 of the fixed methodology: offline teacher solutions for every training pool.

For each train_*.pkl in the pool directory: greedy+prune (stored in the pool, certified) -> local
search (destroy-and-repair, baselines.local_search) -> exact certification of the LS solution.
Saved to data/teacher/<pool name>.npz:
    gp_occ [B,n], gp_count [B], gp_ok [B]        greedy+prune (from the pool file)
    ls_occ [B,n], ls_count [B], ls_ok [B]        local search, exact-checked (worst <= 1% over Zt)
    occ    [B,n], count [B], ok [B]              the teacher: LS where certified and not worse, else g+p
    worst_ls [B]                                 certified log10 worst |Z|/Zt of the LS solution
One process per pool file, single-threaded torch (decap_loop/safe_verify.py).

Run: .venv/bin/python scripts/make_teacher.py [--pool runs/pool_v1] [--out data/teacher] [--glob 'train_*.pkl'] [--ls 4x32]
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def job(args):
    path, out, passes, trials, seed = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from decap_loop.bank import load_bank
    from decap_loop.baselines import local_search
    from decap_loop.verify import certify
    name = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(out, name + ".npz")
    if os.path.exists(dst):
        return name, "exists", 0.0, None
    t = time.time()
    b, res = load_bank(path)
    gp = res["greedy+prune"]
    gp_occ = torch.from_numpy(gp["occ"])
    gp_ok = torch.from_numpy(np.asarray(gp["ok"]))
    ls_occ = local_search(b, b.cfg, gp_occ, gp_ok, passes=passes, trials=trials, seed=seed)
    t_ls = time.time() - t
    inst = torch.arange(b.B)
    worst, _, _ = certify(b, inst, ls_occ, b.cfg)
    ls_ok = worst <= np.log10(1.01)
    ls_count = (ls_occ >= 0).sum(1).numpy()
    gp_count = np.asarray(gp["count"])
    use_ls = ls_ok & (ls_count <= gp_count)
    occ = np.where(use_ls[:, None], ls_occ.numpy(), gp["occ"])
    count = np.where(use_ls, ls_count, gp_count)
    ok = np.where(use_ls, True, np.asarray(gp["ok"]))
    np.savez(dst, gp_occ=gp["occ"], gp_count=gp_count, gp_ok=np.asarray(gp["ok"]), ls_occ=ls_occ.numpy(),
             ls_count=ls_count, ls_ok=ls_ok, worst_ls=worst, occ=occ, count=count, ok=ok)
    both = ls_ok & np.asarray(gp["ok"])
    meta = dict(name=name, B=int(b.B), H=b.H, W=b.W, gp_ok=float(np.mean(gp["ok"])), ls_ok=float(ls_ok.mean()),
                ls_over_gp=float(np.mean(ls_count[both] / np.maximum(gp_count[both], 1))) if both.any() else None,
                improved_frac=float(np.mean(ls_count[both] < gp_count[both])) if both.any() else None,
                gp_count_mean=float(gp_count[np.asarray(gp["ok"])].mean()), teacher_count_mean=float(count[ok].mean()),
                t_ls=t_ls, t_total=time.time() - t, passes=passes, trials=trials, seed=seed)
    json.dump(meta, open(os.path.join(out, name + ".json"), "w"), indent=1)
    return name, "built", time.time() - t, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="runs/pool_v1")
    ap.add_argument("--out", default="data/teacher")
    ap.add_argument("--glob", default="train_*.pkl")
    ap.add_argument("--ls", default="4x32", help="local search passes x trials")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    passes, trials = map(int, a.ls.split("x"))
    files = sorted(glob.glob(os.path.join(a.pool, a.glob)))
    files.sort(key=lambda f: -os.path.getsize(f))
    print(f"{len(files)} pool files -> {a.out}, LS {passes}x{trials}, {a.workers} workers", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for name, status, dt, meta in pool.imap_unordered(job, [(f, a.out, passes, trials, i) for i, f in enumerate(files)]):
            extra = "" if meta is None else (f"g+p ok {meta['gp_ok']:.2f} LS ok {meta['ls_ok']:.2f} "
                                             f"LS/g+p {meta['ls_over_gp']} improved {meta['improved_frac']} "
                                             f"teacher count {meta['teacher_count_mean']:.2f}")
            print(f"[{time.time()-t0:6.0f}s] {name:16s} {status:6s} {dt:6.0f}s {extra}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
