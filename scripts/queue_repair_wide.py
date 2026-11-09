"""PROPOSALS 11/03 T2, rule R1 met on r2_mott_rl seed 0 (round 29: pcb +0.12 [+0.04, +0.20], pkg +0.26
[+0.14, +0.38]) -> widen as pre-registered: seeds 1/2 and the two 16-cell pkg files.
- 10-cell, seeds 1/2 of r2_mott and (fairness, rule R3) r0_coord: names start "ls10 " -> side_slot.py 4th-slot filler
  (10-cell policy evals are light). Records -> runs/repair10/eval/.
- 16-cell pkg (eval_pkg16_0/1): r2_mott_rl, modal_rule, r0_coord_rl, greedy+prune (seed 0). A 16-cell bank eval peaks at
  ~22 GB rss (g+p ~34 GB), so these go into the driver lane (one heavy job at a time, OOM note 11/03 round 27), right
  after the running 20-cell pool build and before the 20-cell evals. Records -> runs/repair16/eval/.
Run: .venv/bin/python scripts/queue_repair_wide.py   (idempotent)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
K = 4
SETS10 = [f"eval_{t}10_{k}" for t in ("ip", "pcb", "pkg") for k in (0, 1)]
SETS16 = ["eval_pkg16_0", "eval_pkg16_1"]
SEEDS = [(f"{r}_s{s}_rl", f"--policy {r}_s{s}_rl=runs/2026-10-29/{r}_s{s}_rl/best.pt")
         for r in ("r2_mott", "r0_coord") for s in (1, 2)]
M16 = [("r2_mott_rl", "--policy r2_mott_rl=runs/2026-10-28/r2_mott_rl/best.pt"),
       ("modal_rule", "--methods modal_rule"),
       ("r0_coord_rl", "--policy r0_coord_rl=runs/2026-10-28/r0_coord_rl/best.pt"),
       ("greedy+prune", "--methods gp")]

side = [dict(name=f"ls10 repair {m}",
             cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*10_*' {arg} --repair {K} --workers 1 "
                 f"--out runs/repair10",
             done=[f"runs/repair10/eval/{s}__{m}+rep{K}.json" for s in SETS10], log="agent/logs/repair10.log")
        for m, arg in SEEDS]
heavy = [dict(name=f"repair16 pkg {m}",
              cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_pkg16_*' {arg} --repair {K} "
                  f"--workers 1 --out runs/repair16",
              done=[f"runs/repair16/eval/{s}__{m}+rep{K}.json" for s in SETS16], log="agent/logs/repair16.log")
         for m, arg in M16]
q = json.load(open(Q))
names = {e["name"] for e in q["cpu"]}
q["cpu"] += [e for e in side if e["name"] not in names]
at = next(i for i, e in enumerate(q["cpu"]) if e["name"] == "pool20 eval k8") + 1
for e in reversed(heavy):
    if e["name"] not in names:
        q["cpu"].insert(at, e)
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print([e["name"] for e in q["cpu"]][at - 1:at + 5])
print([e["name"] for e in q["cpu"]][-6:])
