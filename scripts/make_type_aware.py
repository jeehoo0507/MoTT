"""E0+ "type-aware greedy" baseline (QUEUE 2026-10-29 03:10, item 1), derived from recorded runs only.

scripts/analyze_gp_one_type.py runs greedy+prune with the library restricted to one decap type t, inside the
same cutting plane and exact check as every reported method (records runs/analysis/gp_one_type/
<set>__greedy+prune@type<t>.json). Type-aware greedy runs it once per type and keeps, per instance, the
solution that passes the exact check with the fewest decaps (ties -> lower type index); if no type passes,
the instance fails and the record keeps the type with the lowest exact-check worst |Z|/Zt.
Time per instance = sum over the T runs (it has to run all of them).
Written as an ordinary eval record (runs/<DAY>/baselines_e0plus/eval/<set>__gp_type_aware.json), so
collect_results.py tables it like any method. Only sets where every type has a record are written.
Run: .venv/bin/python scripts/make_type_aware.py   (collect_results.py calls it first)
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "runs/analysis/gp_one_type")
OUT = os.path.join(ROOT, "runs/2026-10-29/baselines_e0plus/eval")
METHOD = "gp_type_aware"
T = 3


def main():
    by = {}
    for f in glob.glob(os.path.join(SRC, "*__greedy+prune@type*.json")):
        s, t = re.match(r"(.+)__greedy\+prune@type(\d+)\.json", os.path.basename(f)).groups()
        by.setdefault(s, {})[int(t)] = f
    os.makedirs(OUT, exist_ok=True)
    wrote = []
    for s, fs in sorted(by.items()):
        if sorted(fs) != list(range(T)):
            continue
        recs = [json.load(open(fs[t])) for t in range(T)]
        ok = np.array([r["ok"] for r in recs], bool)                    # [T,B]
        cnt = np.array([r["count"] for r in recs], float)
        worst = np.array([r["worst"] for r in recs], float)
        key = np.where(ok, cnt, np.inf)
        pick = np.where(ok.any(0), key.argmin(0), worst.argmin(0))       # [B]
        b = np.arange(ok.shape[1])
        tech, size = re.match(r"eval_([a-z]+)(\d+)_\d+", s).groups()
        rec = dict(set=s, method=METHOD, tech=tech, size=int(size), n=int(ok.shape[1]),
                   ok=ok[pick, b].tolist(), count=cnt[pick, b].astype(int).tolist(), worst=worst[pick, b].tolist(),
                   type_picked=pick.tolist(), occ=[recs[t]["occ"][i] for i, t in enumerate(pick)],
                   sec_per_inst=float(sum(r["sec_per_inst"] for r in recs)),
                   rounds=[r["rounds"] for r in recs], sources=[os.path.relpath(fs[t], ROOT) for t in range(T)],
                   note="derived from the single-type greedy+prune records (scripts/make_type_aware.py)")
        json.dump(rec, open(os.path.join(OUT, f"{s}__{METHOD}.json"), "w"))
        wrote.append(s)
    print(f"type-aware greedy records: {wrote}")


if __name__ == "__main__":
    main()
