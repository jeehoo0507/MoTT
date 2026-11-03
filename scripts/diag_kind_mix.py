"""Kind mix per ladder rung and seed on the 10-cell eval boards (round 32 diagnosis, no repair).

For each tech (pcb, pkg, ip), seed (0, 1, 2) and rung (r0_coord, r1_modal, r2_mott, r3_loop), takes
the problems where all four rungs of that seed pass the exact check and reports mean count and mean
decaps per kind (kind 0 = largest C, kind 2 = smallest C / highest SRF). modal_rule (deterministic,
seed-free) is reported on the same problems for reference.

Question (written before running): is the extra kind-2 use of r2_mott over r0_coord under repair
(diag_count_gap) also present without repair, in every seed, and does r1_modal (which has the 1-step
exact-effect features but no mode tokens) share it?

Usage: .venv/bin/python scripts/diag_kind_mix.py  -> runs/analysis/diag_kind_mix.json
"""
import json
import os

import numpy as np

RUNS = {0: {"r0": "runs/2026-10-28/r0_coord_rl", "r1": "runs/2026-10-28/r1_modal_rl",
            "r2": "runs/2026-10-28/r2_mott_rl", "r3": "runs/2026-10-28/r3_loop_rl"}}
for s in (1, 2):
    RUNS[s] = {r: f"runs/2026-10-29/{n}_s{s}_rl" for r, n in
               [("r0", "r0_coord"), ("r1", "r1_modal"), ("r2", "r2_mott"), ("r3", "r3_loop")]}
RULE = "runs/2026-10-28/baselines/eval"


def load(path):
    r = json.load(open(path))
    mix = np.array([[int((np.asarray(o) == t).sum()) for t in range(3)] for o in r["occ"]], float)
    return np.array(r["ok"]), np.array(r["count"], float), mix


def rec(run_dir, tech):
    name = os.path.basename(run_dir)
    parts = [load(f"{run_dir}/eval/eval_{tech}10_{k}__{name}.json") for k in (0, 1)]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def main():
    out = {}
    for tech in ["pcb", "pkg", "ip"]:
        rule = tuple(np.concatenate([load(f"{RULE}/eval_{tech}10_{k}__modal_rule.json")[i] for k in (0, 1)])
                     for i in range(3))
        for s, runs in RUNS.items():
            recs = {r: rec(d, tech) for r, d in runs.items()}
            both = np.all([v[0] for v in recs.values()], axis=0)
            row = {"n_all_ok": int(both.sum())}
            line = f"{tech} s{s} all-ok {both.sum():2d}/50 |"
            for r, (ok, cnt, mix) in list(recs.items()) + [("rule", rule)]:
                pk = mix[both].mean(0)
                row[r] = {"mean_count": round(float(cnt[both].mean()), 2),
                          "per_kind": [round(float(x), 2) for x in pk]}
                line += f" {r} {cnt[both].mean():5.2f} [{pk[0]:.2f} {pk[1]:.2f} {pk[2]:.2f}] |"
            row["success"] = {r: round(float(v[0].mean()), 2) for r, v in list(recs.items()) + [("rule", rule)]}
            out[f"{tech}_s{s}"] = row
            print(line, "success", row["success"])
    os.makedirs("runs/analysis", exist_ok=True)
    json.dump(out, open("runs/analysis/diag_kind_mix.json", "w"), indent=1)


if __name__ == "__main__":
    main()
