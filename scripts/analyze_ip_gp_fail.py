"""Where does greedy+prune fail on interposer-like boards? (recorded occ only, no training, no new search)

For every instance of the given sets where greedy+prune fails the exact check, recompute on the exact
700-point check grid (verify._probe_z_exact, same as scripts/fig_board.py):
  bare  : frequency of the bare board's worst |Z|/Zt
  gp    : worst |Z|/Zt and its frequency with the greedy+prune decaps
  pol   : the same for a policy record on the same instance (if it passed)
and the total placed capacitance per method. Writes a JSON next to --out and prints a table.

Run: .venv/bin/python scripts/analyze_ip_gp_fail.py --sets eval_ip10_0,eval_ip10_1 \
       --gp runs/2026-10-28/baselines/eval --policy runs/2026-10-28/r3_loop_rl/eval:r3_loop_rl \
       --out runs/analysis/ip_gp_fail.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.pdn import target_at  # noqa: E402
from decap_loop.verify import _inst_data, _probe_z_exact  # noqa: E402


def curve(batch, b, occ):
    cfg = batch.cfg
    mask, ent, P = _inst_data(batch, b)
    f = cfg.check_freqs()
    zt = target_at(batch.zt_par[b:b + 1], f)[0]
    r = (_probe_z_exact(mask, ent, P, occ, cfg, f) / zt[:, None]).max(1)
    i = int(np.argmax(r))
    return float(r[i]), float(f[i])


def cap(cfg, occ):
    occ = np.asarray(occ)
    return float(sum(cfg.decaps[t][0] for t in occ[occ >= 0]))


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
        g = json.load(open(os.path.join(a.gp, f"{s}__greedy+prune.json")))
        p = json.load(open(os.path.join(pdir, f"{s}__{pm}.json")))
        for b in range(len(g["ok"])):
            if g["ok"][b]:
                continue
            bw, bf = curve(batch, b, None)
            gw, gf = curve(batch, b, np.asarray(g["occ"][b]))
            row = dict(set=s, b=b, bare_worst=bw, bare_f=bf, gp_count=int(g["count"][b]), gp_worst=gw, gp_f=gf,
                       gp_cap_pF=cap(batch.cfg, g["occ"][b]) * 1e12, pol_ok=bool(p["ok"][b]))
            if p["ok"][b]:
                pw, pf = curve(batch, b, np.asarray(p["occ"][b]))
                row.update(pol_count=int(p["count"][b]), pol_worst=pw, pol_f=pf, pol_cap_pF=cap(batch.cfg, p["occ"][b]) * 1e12)
            rows.append(row)
            print(f"{s} #{b:2d} bare {bw:6.2f}@{bf/1e9:5.2f}G | gp {row['gp_count']:2d} {gw:5.2f}@{gf/1e9:5.2f}G "
                  f"C {row['gp_cap_pF']:6.0f}p | {pm} " + (f"{row['pol_count']:2d} {pw:5.2f}@{pf/1e9:5.2f}G C {row['pol_cap_pF']:6.0f}p"
                                                         if p["ok"][b] else "fail"), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(args=vars(a), rows=rows), open(a.out, "w"), indent=1)
    gf = np.array([r["gp_f"] for r in rows]); bf = np.array([r["bare_f"] for r in rows])
    print(f"n {len(rows)} | gp worst f / bare worst f: median {np.median(gf / bf):.2f}, "
          f"below bare peak {np.mean(gf < bf):.2f} | gp worst median {np.median([r['gp_worst'] for r in rows]):.2f}")
    pol = [r for r in rows if r["pol_ok"]]
    if pol:
        print(f"{pm} passes {len(pol)}: C gp/{pm} median {np.median([r['gp_cap_pF'] / r['pol_cap_pF'] for r in pol]):.2f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
