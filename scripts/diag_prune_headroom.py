"""Prune headroom of a method's certified solutions (round 32 diagnosis, analysis only).

For each problem a method solved (exact check passed in its eval record), greedily remove placed decaps
one at a time while the exact dense check still passes (same verify.exact_worst and tolerance as the
eval). Reports how many decaps were removable. Question (written before running): does the policy leave
removable decaps in its own solutions on pcb/pkg (T4 showed r2_mott never removes from a met-target
state)? If the removable count is ~0 on pcb/pkg, a "learn to remove" RL variant has no headroom there.

Usage: .venv/bin/python scripts/diag_prune_headroom.py <eval_dir> <method> [sets_glob] [max_problems]
  -> runs/analysis/prune_headroom_<method>.json
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)

from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.verify import _inst_data, exact_worst  # noqa: E402

TOL = 1e-3


def worst(b, i, occ, cfg):
    mask, ent, P = _inst_data(b, i)
    w, _, _ = exact_worst(mask, ent, P, occ, cfg, b.zt_par[i:i + 1])
    return np.log10(w)


def main():
    eval_dir, method = sys.argv[1], sys.argv[2]
    sets = sys.argv[3] if len(sys.argv) > 3 else "eval_*10_*"
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    out = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, f"{sets}__{method}.json"))):
        r = json.load(open(f))
        b, _ = load_bank(f"runs/pool_v1/{r['set']}.pkl")
        cfg = b.cfg
        rows, t0 = [], time.time()
        for i, (ok, occ) in enumerate(zip(r["ok"], r["occ"])):
            if not ok or len(rows) >= cap:
                continue
            occ = np.asarray(occ).copy()
            removed, changed = 0, True
            while changed:
                changed = False
                for p in np.nonzero(occ >= 0)[0]:
                    t = occ[p]
                    occ[p] = -1
                    if worst(b, i, occ, cfg) <= TOL:
                        removed += 1
                        changed = True
                    else:
                        occ[p] = t
            rows.append((r["count"][i], removed))
        c = np.array(rows, float)
        out[r["set"]] = {"n_solved": len(rows), "mean_count": float(c[:, 0].mean()),
                         "mean_removable": float(c[:, 1].mean()), "frac_with_removable": float((c[:, 1] > 0).mean()),
                         "sec": round(time.time() - t0, 1)}
        print(r["set"], out[r["set"]], flush=True)
    os.makedirs("runs/analysis", exist_ok=True)
    json.dump(out, open(f"runs/analysis/prune_headroom_{method}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
