"""Greedy+prune with the decap library restricted to one type (analysis only, not a reported baseline).

scripts/analyze_ip_swap.py: on the 30 ip side-10 instances where greedy+prune fails, the same positions with
every decap set to type 0 (100 pF) pass the exact check (30/30), with all type 1 or 2 none pass. So the
failure is the type the greedy rule picks, not where it places. This asks the next question: if the rule may
only use type t, does it pass, and with how many decaps? Same greedy rule (baselines.heuristic_rollout,
adds restricted with its `only` argument), same cutting plane / exact check / rounds as scripts/eval_policy.py.
Records -> <out>/<set>__greedy+prune@type<t>.json (ok, count, worst, occ, rounds, sec_per_inst).

Run: .venv/bin/python scripts/analyze_gp_one_type.py --sets eval_ip10_0,eval_ip10_1 --type 0 \
       --out runs/analysis/gp_one_type
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import torch
    torch.set_num_threads(1)
    from decap_loop.bank import load_bank
    from decap_loop.baselines import Env, _result, heuristic_rollout
    from decap_loop.verify import cutting_plane
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="runs/pool_v1")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--type", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.sets.split(","):
        dst = os.path.join(a.out, f"{s}__greedy+prune@type{a.type}.json")
        if a.skip_existing and os.path.exists(dst):
            print(f"{s} type {a.type} exists", flush=True)
            continue
        b, _ = load_bank(os.path.join(a.pool, s + ".pkl"))
        cfg = b.cfg
        box = {}

        def fn(bb):
            t = time.time()
            inst = torch.arange(bb.B, device=bb.valid.device)
            env = Env(bb, inst, cfg)
            only = torch.full_like(env.occ, a.type)
            r = _result(env, heuristic_rollout(env, True, only=only))
            box["last"] = time.time() - t
            return r
        t0 = time.time()
        r, info = cutting_plane(fn, b, cfg, max_rounds=a.rounds)
        ok = info["worst"] <= np.log10(1.01)
        cnt = (r["occ"] >= 0).sum(1).cpu().numpy()
        rec = dict(set=s, method=f"greedy+prune@type{a.type}", decap=list(cfg.decaps[a.type]), n=int(b.B),
                   ok=ok.tolist(), count=cnt.tolist(), worst=[float(x) for x in info["worst"]],
                   sec_per_inst=box["last"] / b.B, sec_total=time.time() - t0, rounds=int(info["rounds"]),
                   F_final=int(b.freqs.shape[1]), occ=r["occ"].cpu().numpy().tolist(),
                   time=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
        json.dump(rec, open(dst, "w"))
        print(f"{s} type {a.type} ok {ok.mean():.2f} count {cnt[ok].mean() if ok.any() else float('nan'):.2f} "
              f"rounds {info['rounds']} {rec['sec_total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
