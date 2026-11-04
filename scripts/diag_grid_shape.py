"""D8 side check (11/04 round 37): shape of the sorted working-grid max_probe level curve that MoTT mode tokens are
picked from (sim.features_mott rule). Bare boards of eval_pkg10_0 / eval_pcb10_0: grid size, unique frequencies,
number of local maxima (>= both neighbours, the sim rule), and a strict-local-maximum count for comparison.
Run: .venv/bin/python scripts/diag_grid_shape.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from diag_fail_freq import tokens_and_grid  # noqa: E402


def main():
    torch.set_num_threads(1)
    from decap_loop.bank import load_bank
    for tech in ("pkg", "pcb"):
        b, _ = load_bank(os.path.join(ROOT, f"runs/pool_v1/eval_{tech}10_0.pkl"))
        ib = torch.arange(min(b.B, 5))
        occ = torch.full((len(ib), b.n), -1, dtype=torch.long)
        lf, lv, toks, tj = tokens_and_grid(b, ib, occ, allpk=[])
        for e in range(len(ib)):
            g, v = lf[e], lv[e]
            d = np.diff(v)
            strict = int(np.sum((d[:-1] > 0) & (d[1:] < 0)))
            loose = int(np.sum(np.r_[True, v[1:] <= v[:-1]] & np.r_[v[:-1] >= v[1:], True]))
            sign_flips = int(np.sum(np.diff(np.sign(d)) != 0))
            print(f"{tech} board {e}: F={len(g)} unique_f={len(np.unique(np.round(g, 9)))} "
                  f"loose_max={loose} strict_max={strict} slope_sign_flips={sign_flips} "
                  f"min_df={np.min(np.diff(g)):.2e} med_df={np.median(np.diff(g)):.3f} lv range [{v.min():.2f},{v.max():.2f}]")
            if e == 0:
                print("  first 30 (log10 f, level):", " ".join(f"({a:.3f},{c:.3f})" for a, c in zip(g[:30], v[:30])))
                print("  tokens log10 f:", np.round(toks[e], 3).tolist(), "levels:", np.round(v[tj[e]], 3).tolist())


if __name__ == "__main__":
    main()
