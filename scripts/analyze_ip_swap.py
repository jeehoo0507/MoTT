"""Is the ip greedy+prune failure decided by the decap types or by the positions? (recorded occ, no search)

Follows scripts/analyze_ip_gp_fail.py (same exact 700-point check via verify._probe_z_exact). For every
instance where greedy+prune fails the exact check, keep the positions of a recorded solution and set every
placed decap to one library type t (t = 0, 1, 2), then recompute the exact worst |Z|/Zt:
  gp@t  : greedy+prune positions, all decaps type t
  pol@t : policy positions (only where the policy passed), all decaps type t
Also prints the greedy+prune record's cutting-plane rounds: 0 means no solution the search called feasible
was refuted by the exact check, i.e. the failures are the search ending infeasible on its own working grid
(not a grid artefact).

Run: .venv/bin/python scripts/analyze_ip_swap.py --sets eval_ip10_0,eval_ip10_1 \
       --gp runs/2026-10-28/baselines/eval --policy runs/2026-10-28/r3_loop_rl/eval:r3_loop_rl \
       --out runs/analysis/ip_swap.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ip_gp_fail import curve  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decap_loop.bank import load_bank  # noqa: E402


def swap(occ, t):
    occ = np.asarray(occ).copy()
    occ[occ >= 0] = t
    return occ


def main():
    import torch
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="runs/pool_v1")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--gp", required=True, help="dir with <set>__greedy+prune.json")
    ap.add_argument("--policy", required=True, help="<dir>:<method>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pdir, pm = a.policy.split(":")
    rows = []
    for s in a.sets.split(","):
        batch, _ = load_bank(os.path.join(a.pool, s + ".pkl"))
        T = len(batch.cfg.decaps)
        g = json.load(open(os.path.join(a.gp, f"{s}__greedy+prune.json")))
        p = json.load(open(os.path.join(pdir, f"{s}__{pm}.json")))
        for b in range(len(g["ok"])):
            if g["ok"][b]:
                continue
            go = np.asarray(g["occ"][b])
            row = dict(set=s, b=b, gp_count=int(g["count"][b]), gp_rounds=int(g["rounds"]),
                       gp=curve(batch, b, go)[0], gp_types=np.bincount(go[go >= 0], minlength=T).tolist(),
                       gp_at=[curve(batch, b, swap(go, t))[0] for t in range(T)], pol_ok=bool(p["ok"][b]))
            if p["ok"][b]:
                po = np.asarray(p["occ"][b])
                row.update(pol_count=int(p["count"][b]), pol=curve(batch, b, po)[0],
                           pol_types=np.bincount(po[po >= 0], minlength=T).tolist(),
                           pol_at=[curve(batch, b, swap(po, t))[0] for t in range(T)])
            rows.append(row)
            print(f"{s} #{b:2d} gp {row['gp_count']:2d} {row['gp']:5.2f} types {row['gp_types']} @t "
                  + " ".join(f"{x:5.2f}" for x in row["gp_at"]) + f" | {pm} "
                  + (f"{row['pol_count']:2d} {row['pol']:5.2f} types {row['pol_types']} @t "
                     + " ".join(f"{x:5.2f}" for x in row["pol_at"]) if p["ok"][b] else "fail"), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(args=vars(a), rows=rows), open(a.out, "w"), indent=1)
    T = len(rows[0]["gp_at"])
    print(f"n {len(rows)} | greedy+prune record cutting-plane rounds: {sorted({r['gp_rounds'] for r in rows})}")
    for t in range(T):
        print(f"gp positions, all type {t}: pass {np.mean([r['gp_at'][t] <= 1.01 for r in rows]):.2f}, "
              f"worst median {np.median([r['gp_at'][t] for r in rows]):.2f}")
    pol = [r for r in rows if r["pol_ok"]]
    for t in range(T):
        print(f"{pm} positions ({len(pol)}), all type {t}: pass {np.mean([r['pol_at'][t] <= 1.01 for r in pol]):.2f}, "
              f"worst median {np.median([r['pol_at'][t] for r in pol]):.2f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
