"""Cutting-plane rounds / final grid size per (pool file, method) from existing eval records (no new runs).

Question (QUEUE 11/03 obs. 4 follow-up): do the policy's far misses on side 16 come with the round cap
(every round the re-run finds a new grid gap) or with a converged grid?
Run: .venv/bin/python scripts/diag_rounds.py
"""
import glob
import json
import os

import numpy as np

RUNS = {"r2_mott_rl": "runs/2026-10-28/r2_mott_rl", "r0_coord_rl": "runs/2026-10-28/r0_coord_rl",
        "greedy+prune": "runs/2026-10-28/baselines", "modal_rule": "runs/2026-10-28/baselines"}


def main():
    rows = {}
    for meth, d in RUNS.items():
        for f in sorted(glob.glob(os.path.join(d, "eval", f"eval_*__{meth}.json"))):
            r = json.load(open(f))
            ok = np.array(r["ok"])
            rows.setdefault(r["set"], {})[meth] = dict(
                ok=round(float(ok.mean()), 2), rounds=r["rounds"], F_final=r["F_final"],
                count=round(float(np.mean(r["count"])), 2))
    for s in sorted(rows):
        print(s, json.dumps(rows[s]))
    os.makedirs("runs/analysis", exist_ok=True)
    json.dump(rows, open("runs/analysis/diag_rounds.json", "w"), indent=1)


if __name__ == "__main__":
    main()
