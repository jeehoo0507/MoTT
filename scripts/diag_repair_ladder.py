"""PROPOSALS 11/03 R4: under the same repair (eval_policy --repair 4), rung 2 (r2_mott) vs rung 0 (r0_coord), seeds 0/1/2,
10-cell pcb and pkg. Per seed: paired success difference on the same 50 problems + bootstrap CI; pooled: 150 pairs,
bootstrap resampling problems within each seed (stratified). Count ratio r2/r0 = sum over problems both solve.
Usage: .venv/bin/python scripts/diag_repair_ladder.py  -> runs/analysis/diag_repair_ladder.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "runs/repair10/eval")
SEEDS = {0: ("r2_mott_rl", "r0_coord_rl"), 1: ("r2_mott_s1_rl", "r0_coord_s1_rl"), 2: ("r2_mott_s2_rl", "r0_coord_s2_rl")}
B = 2000


def load(tech, m):
    ok, cnt = [], []
    for k in (0, 1):
        r = json.load(open(os.path.join(D, f"eval_{tech}10_{k}__{m}+rep4.json")))
        ok += list(map(bool, r["ok"]))
        cnt += list(map(float, r["count"]))
    return np.array(ok), np.array(cnt)


def main():
    rng = np.random.default_rng(0)
    out = {}
    for tech in ("pcb", "pkg"):
        per, diffs, pairs = {}, [], []
        for s, (a, b) in SEEDS.items():
            oa, ca = load(tech, a)
            ob, cb = load(tech, b)
            d = oa.astype(float) - ob.astype(float)
            bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
            both = oa & ob
            per[f"s{s}"] = dict(r2=float(oa.mean()), r0=float(ob.mean()), diff=float(d.mean()),
                                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                count_ratio=float(ca[both].sum() / cb[both].sum()), n_both=int(both.sum()))
            diffs.append(d)
            pairs.append((ca, cb, both))
        pooled = np.concatenate(diffs)
        boots = np.array([np.concatenate([d[rng.integers(0, len(d), len(d))] for d in diffs]).mean() for _ in range(B)])
        rboot = []
        for _ in range(B):
            na = nb = 0.0
            for ca, cb, both in pairs:
                i = rng.integers(0, len(ca), len(ca))
                m = both[i]
                na += ca[i][m].sum()
                nb += cb[i][m].sum()
            rboot.append(na / nb)
        num = sum(ca[both].sum() for ca, cb, both in pairs)
        den = sum(cb[both].sum() for ca, cb, both in pairs)
        out[tech] = dict(per_seed=per, pooled=dict(diff=float(pooled.mean()),
                                                   ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                                   count_ratio=float(num / den),
                                                   count_ratio_ci=[float(np.percentile(rboot, 2.5)),
                                                                   float(np.percentile(rboot, 97.5))]))
        all_pos = all(v["diff"] > 0 for v in per.values())
        lo = out[tech]["pooled"]["ci"][0]
        out[tech]["verdict"] = "r2 > r0 under the same repair" if all_pos and lo > 0 else "no difference"
        print(tech, json.dumps(out[tech]))
    json.dump(out, open(os.path.join(ROOT, "runs/analysis/diag_repair_ladder.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
