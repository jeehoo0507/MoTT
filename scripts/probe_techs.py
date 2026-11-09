"""Quick look at each kind: greedy+prune success / count on a few boards per size (working grid
only, no certification) to choose difficulty ranges before building pools. One process per
(kind, difficulty, size), single-threaded torch (see decap_loop/safe_verify.py).

Run: .venv/bin/python scripts/probe_techs.py [--n 4] [--sizes 10,16,20]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def job(args):
    name, diff_name, diff, s, n = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from decap_loop.baselines import greedy
    from decap_loop.pdn import sample_batch
    from decap_loop.techs import TECHS
    cfg = TECHS[name]
    t = time.time()
    b = sample_batch(n, s, s, cfg, np.random.default_rng(1000 + s), difficulty=diff)
    r = greedy(b, cfg)
    ok = r["feasible"].cpu().numpy()
    cnt = r["count"].cpu().numpy()
    return (f"{name} {diff_name} {s}x{s}: g+p feasible {ok.mean():.2f} count {cnt.tolist()} "
            f"F={b.freqs.shape[1]} targets {b.targets} ({time.time()-t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--sizes", default="10,16,20")
    a = ap.parse_args()
    from decap_loop.techs import HARD, EASY, TECHS, describe
    for k in TECHS:
        print(describe(k), flush=True)
    js = [(t, dn, d, s, a.n) for t in TECHS for dn, d in (("hard", HARD), ("easy", EASY))
          for s in map(int, a.sizes.split(","))]
    with Pool(min(16, len(js))) as pool:
        for line in pool.imap_unordered(job, js):
            print("  " + line, flush=True)


if __name__ == "__main__":
    main()
