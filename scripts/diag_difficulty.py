"""Diagnosis (11/03 round 29, QUEUE 11/03 11:40 KST observation 2: learning loses on the hard problems).
Question: are the hard eval problems (where r2_mott_rl fails) inside the training pool's difficulty range but rare,
or outside it?  Difficulty proxies, all from the instance definition / stored greedy+prune result only (no method
output): bare-board worst log10(|Z|/Zt) on the working grid, number of violating peaks, g+p count.
Train pools: runs/pool_v1/train_*.pkl (sizes 8/10/12); eval: eval_*10_* (bare features + g+p count) and eval_*16_*
(g+p count only, from the eval record - loading a 16-cell bank densely is ~8 GB, not done here).
For each tech: train / eval quantiles, the train-percentile of every eval failure of the policy, and the policy's
success by train-difficulty tercile.  Also the teacher coverage per train tercile (data/teacher_rcp/*.npz ok).
Run: .venv/bin/python scripts/diag_difficulty.py [run=runs/2026-10-28/r2_mott_rl] [method=r2_mott_rl]
Writes runs/analysis/diag_difficulty_<method>.json.  CPU, 1 thread, ~1-2 min."""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
BASE = "runs/2026-10-28/baselines/eval"


def bare(path):
    from problem_features import features
    import pickle
    rows = features(path)
    g = pickle.load(open(path, "rb"))["results"]["greedy+prune"]
    for r, c, ok in zip(rows, g["count"], g["ok"]):
        r["gp_count"], r["gp_ok"] = int(c), bool(ok)
    return rows


def pct(train, x):
    """percentile of x within the train sample (0..1)"""
    t = np.sort(np.asarray(train))
    return float(np.searchsorted(t, x, side="right") / len(t))


def main():
    torch.set_num_threads(1)
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-10-28/r2_mott_rl"
    meth = sys.argv[2] if len(sys.argv) > 2 else "r2_mott_rl"
    out = {}
    for tech in ("pcb", "pkg", "ip"):
        tr = []
        for p in sorted(glob.glob(os.path.join(ROOT, f"runs/pool_v1/train_{tech}*.pkl"))):
            name = os.path.basename(p)[:-4]
            rows = bare(p)
            tz = os.path.join(ROOT, "data/teacher_rcp", name + ".npz")
            t0 = os.path.join(ROOT, "data/teacher", name + ".npz")
            for key, f in (("teach_rcp_ok", tz), ("teach_ok", t0)):
                if os.path.exists(f):
                    ok = np.load(f)["ok"]
                    for r, o in zip(rows, ok):
                        r[key] = bool(o)
            for r in rows:
                r["file"] = name
            tr += rows
        res = dict(n_train=len(tr))
        keys = ("bare_worst", "n_viol_peaks", "gp_count")
        q = lambda a: [round(float(x), 3) for x in np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9, 1.0])]
        res["train_q"] = {k: q([r[k] for r in tr]) for k in keys}
        for size in (8, 10, 12):
            sub = [r for r in tr if f"{tech}{size}_" in r["file"]]
            res[f"train{size}_q"] = {k: q([r[k] for r in sub]) for k in keys}
        cut = {k: np.quantile([r[k] for r in tr], [1 / 3, 2 / 3]) for k in keys}
        for k in keys:   # teacher coverage per train tercile
            ter = np.digitize([r[k] for r in tr], cut[k])
            res[f"train_teach_ok_by_{k}_tercile"] = [
                [round(float(np.mean([r.get(t, np.nan) for r, z in zip(tr, ter) if z == j])), 3)
                 for j in range(3)] for t in ("teach_ok", "teach_rcp_ok")]
        for size in (10, 16):
            ev, okp = [], []
            for p in sorted(glob.glob(os.path.join(ROOT, f"runs/pool_v1/eval_{tech}{size}_*.pkl"))):
                s = os.path.basename(p)[:-4]
                rec = os.path.join(ROOT, run, "eval", f"{s}__{meth}.json")
                gp = os.path.join(ROOT, BASE, f"{s}__greedy+prune.json")
                if not os.path.exists(rec):
                    continue
                if size == 10:
                    rows = bare(p)
                else:
                    g = json.load(open(gp))
                    rows = [dict(gp_count=int(c), gp_ok=bool(o)) for c, o in zip(g["count"], g["ok"])]
                ev += rows
                okp += json.load(open(rec))["ok"]
            if not ev:
                continue
            okp = np.array(okp, bool)
            kk = keys if size == 10 else ("gp_count",)
            d = dict(n=len(ev), success=round(float(okp.mean()), 3))
            for k in kk:
                x = np.array([r[k] for r in ev], float)
                d[f"{k}_q"] = q(x)
                pf = np.array([pct([r[k] for r in tr], v) for v in x])
                d[f"{k}_fail_train_pct"] = sorted(round(float(v), 3) for v in pf[~okp])
                d[f"{k}_share_above_train_max"] = round(float((x > max(r[k] for r in tr)).mean()), 3)
                d[f"{k}_fail_share_above_train_q75"] = round(float((pf[~okp] > 0.75).mean()), 3) if (~okp).any() else None
                ter = np.digitize(x, cut[k])
                d[f"{k}_success_by_train_tercile"] = [
                    (round(float(okp[ter == j].mean()), 3) if (ter == j).any() else None, int((ter == j).sum()))
                    for j in range(3)]
            res[f"eval{size}"] = d
        out[tech] = res
        print(tech, json.dumps(res, indent=None)[:3000], flush=True)
    os.makedirs(os.path.join(ROOT, "runs/analysis"), exist_ok=True)
    path = os.path.join(ROOT, f"runs/analysis/diag_difficulty_{meth}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
