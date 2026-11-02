"""Certified problem pools for three board kinds (PCB / PKG / IP), built in parallel processes.

Files (runs/pool_v1/):
  train_{tech}{size}_{k}.pkl   sizes 8/10/12, k = 0..2, 16 instances each       (training mixture)
  val_{tech}10_0.pkl           16 instances, size 10                             (checkpoint selection)
  eval_{tech}{size}_{k}.pkl    sizes 10/16/20, k = 0..1, 25 instances each      (held-out evaluation)

Every file: sample_batch -> cutting_plane(greedy+prune, <= 10 rounds) so that the working grid
is certified for the greedy+prune solution; the greedy+prune result (certified) is stored.
Seeds are fixed per file, so the pools are reproducible. Each worker runs single-threaded torch
(see decap_loop/safe_verify.py for why).

Run: .venv/bin/python scripts/build_pools.py [--out runs/pool_v1] [--workers 14] [--only eval,val,train]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_SIZES, TRAIN_K, TRAIN_N = (8, 10, 12), 3, 16
EVAL_SIZES, EVAL_K, EVAL_N = (10, 16, 20), 2, 25
VAL_N = 16
SEED = {"pcb": 0, "pkg": 100000, "ip": 200000}


def jobs(out, only, eval_sizes=EVAL_SIZES, val_k=1):
    from decap_loop.techs import TECHS
    js = []
    for t in TECHS:
        if "eval" in only:
            for s in eval_sizes:
                for k in range(EVAL_K):
                    js.append((out, f"eval_{t}{s}_{k}", t, s, EVAL_N, SEED[t] + 9000 + 10 * s + k))
        if "val" in only:
            for k in range(val_k):   # 11/02 (diagnosis H1): extra val files _1, _2 for less noisy best.pt selection
                js.append((out, f"val_{t}10_{k}", t, 10, VAL_N, SEED[t] + 5000 + k))
        if "train" in only:
            for s in TRAIN_SIZES:
                for k in range(TRAIN_K):
                    js.append((out, f"train_{t}{s}_{k}", t, s, TRAIN_N, SEED[t] + 100 * s + k))
    # biggest first so the slow ones do not trail at the end
    js.sort(key=lambda j: -(j[3] ** 2 * j[4]))
    return js


def build(job):
    out, name, tech, size, n, seed = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from decap_loop.bank import save_bank
    from decap_loop.baselines import greedy
    from decap_loop.pdn import sample_batch
    from decap_loop.techs import DIFFICULTY, TECHS
    from decap_loop.verify import cutting_plane
    path = os.path.join(out, name + ".pkl")
    if os.path.exists(path):
        return name, "exists", 0.0, None
    cfg = TECHS[tech]
    t = time.time()
    torch.manual_seed(seed)
    batch = sample_batch(n, size, size, cfg, np.random.default_rng(seed), difficulty=DIFFICULTY[tech])
    t_gen = time.time() - t
    r, info = cutting_plane(lambda b: greedy(b, cfg), batch, cfg, max_rounds=10)
    ok = info["worst"] <= np.log10(1.01)
    cnt = (r["occ"] >= 0).sum(1).cpu().numpy()
    res = {"greedy+prune": dict(occ=r["occ"].cpu().numpy(), worst=info["worst"], ok=ok, count=cnt)}
    save_bank(path, batch, res)
    meta = dict(name=name, tech=tech, size=size, n=n, seed=seed, difficulty=DIFFICULTY[tech], F=int(batch.freqs.shape[1]),
                rounds=int(info["rounds"]), gp_ok=float(ok.mean()), gp_count_mean=float(cnt[ok].mean()) if ok.any() else None,
                gp_count=cnt.tolist(), shapes=batch.shapes, targets=batch.targets, probes=int(batch.P),
                t_gen=t_gen, t_total=time.time() - t)
    json.dump(meta, open(os.path.join(out, name + ".json"), "w"), indent=1)
    return name, "built", time.time() - t, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pool_v1")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--only", default="eval,val,train")
    ap.add_argument("--eval_sizes", default=",".join(map(str, EVAL_SIZES)),
                    help="subset of eval board sides (20 last, 2 workers: QUEUE 2026-10-28 memory rule)")
    ap.add_argument("--val_k", type=int, default=1, help="val files per kind (val_{tech}10_0 .. _{k-1})")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    js = jobs(a.out, a.only.split(","), tuple(map(int, a.eval_sizes.split(","))), a.val_k)
    print(f"{len(js)} pool files -> {a.out}, {a.workers} workers", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for name, status, dt, meta in pool.imap_unordered(build, js):
            extra = "" if meta is None else (f"F={meta['F']} rounds {meta['rounds']} g+p ok {meta['gp_ok']:.2f} "
                                             f"count {meta['gp_count_mean']}")
            print(f"[{time.time()-t0:6.0f}s] {name:16s} {status:6s} {dt:6.0f}s {extra}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
