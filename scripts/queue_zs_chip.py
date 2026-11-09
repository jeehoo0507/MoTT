"""PROGRESS 11/03 round 25: zero-shot to the unseen fourth kind `chip` (AGENT_PROMPT bonus; plan and verdict rule
fixed in PROPOSALS 11/03 before any result). No training: the ladder runs trained on pcb+pkg+ip (seed 0/1/2) and the
classical baselines on two new 10-cell eval files.

Pools go to runs/pool_zs (not runs/pool_v1), so the driver's eval10 glob and every training's val never see chip.
Records go to runs/zs_chip/<tag>/eval/ (and runs/analysis/gp_one_type/ for the E0+ per-type runs, which
make_type_aware.py turns into gp_type_aware records). Names start "ls10 " so side_slot.py runs them in the 4th CPU
slot one at a time (10-cell, light); they are appended at the end of the CPU queue, so the driver lane reaches them
only after everything else. Order: pool -> g+p/rule -> ladder seed 0, 1, 2 -> E0+ types.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
POOL, OUT, LOG = "runs/pool_zs", "runs/zs_chip", "agent/logs/zs_chip.log"
SETS = ["eval_chip10_0", "eval_chip10_1"]
RUNG = ("r0_coord", "r1_modal", "r2_mott", "r3_loop", "r2a")   # r2a = final candidate (2단 + --fc_bias, QUEUE 10/29 item 3)

new = [dict(name="ls10 zs-chip pool",
            cmd=f"{PY} scripts/build_pools.py --only eval --eval_sizes 10 --techs chip --out {POOL} --workers 1",
            done=[f"{POOL}/{s}.pkl" for s in SETS], log=LOG),
       dict(name="ls10 zs-chip gp,modal_rule",
            cmd=f"{PY} scripts/eval_policy.py --pool {POOL} --sets 'eval_chip10_*' --methods gp,modal_rule --workers 1 "
                f"--out {OUT}/baselines",
            done=[f"{OUT}/baselines/eval/{s}__{m}.json" for s in SETS for m in ("greedy+prune", "modal_rule")], log=LOG)]
for seed in (0, 1, 2):
    names = [f"{r}_rl" if seed == 0 else f"{r}_s{seed}_rl" for r in RUNG]
    runs = [(n, next(d for d in (f"runs/2026-10-28/{n}", f"runs/2026-10-29/{n}") if os.path.exists(os.path.join(ROOT, d, "best.pt"))))
            for n in names]
    pol = " ".join(f"{n}={d}/best.pt" for n, d in runs)
    new.append(dict(name=f"ls10 zs-chip ladder s{seed}",
                    cmd=f"{PY} scripts/eval_policy.py --pool {POOL} --sets 'eval_chip10_*' --policy {pol} --workers 1 "
                        f"--inst_chunk 5 --out {OUT}/s{seed}",
                    done=[f"{OUT}/s{seed}/eval/{s}__{n}.json" for s in SETS for n, _ in runs], log=LOG))
for t in range(3):
    new.append(dict(name=f"ls10 zs-chip e0plus type{t}",
                    cmd=f"{PY} scripts/analyze_gp_one_type.py --pool {POOL} --sets {','.join(SETS)} --type {t} "
                        f"--skip_existing --out runs/analysis/gp_one_type",
                    done=[f"runs/analysis/gp_one_type/{s}__greedy+prune@type{t}.json" for s in SETS], log=LOG))

q = json.load(open(Q))
mine = {e["name"]: e for e in new}
started = [e["name"] for e in q["cpu"] if e["name"] in mine and e["cmd"] != mine[e["name"]]["cmd"]
           and any(os.path.exists(os.path.join(ROOT, p)) for p in e["done"])]
assert not started, f"changed entries already have records: {started}"
q["cpu"] = [mine.get(e["name"], e) for e in q["cpu"]]     # re-run: refresh entries in place (11/03: + r2a)
names = {e["name"] for e in q["cpu"]}
new = [e for e in new if e["name"] not in names]
q["cpu"] += new
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print("appended", len(new), "entries:", [e["name"] for e in new], "| refreshed", len(mine) - len(new))
