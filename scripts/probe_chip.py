"""Difficulty check for the zero-shot kind `chip` (decap_loop/techs.py), like probe_techs.py but one
process (CPU rule: <= 4 heavy processes) and 10-cell only. Working grid greedy+prune, no
certification -- only to pick the target range before building the eval files; not a report number.

Run: .venv/bin/python scripts/probe_chip.py [--n 4]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=10)
    a = ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from decap_loop.baselines import greedy
    from decap_loop.pdn import sample_batch
    from decap_loop.techs import EASY, HARD, CHIP, describe
    print(describe("chip"), flush=True)
    for dn, d in (("hard", HARD), ("easy", EASY)):
        t = time.time()
        b = sample_batch(a.n, a.size, a.size, CHIP, np.random.default_rng(1000 + a.size), difficulty=d)
        r = greedy(b, CHIP)
        ok = r["feasible"].cpu().numpy()
        cnt = r["count"].cpu().numpy()
        types = np.bincount(r["occ"][r["occ"] >= 0].cpu().numpy().ravel(), minlength=CHIP.n_types)
        print(f"  chip {dn} {a.size}x{a.size}: g+p feasible {ok.mean():.2f} count {cnt.tolist()} types {types.tolist()} "
              f"F={b.freqs.shape[1]} targets {b.targets} ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
