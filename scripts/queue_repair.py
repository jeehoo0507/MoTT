"""PROPOSALS 11/03 T2 (round 28): the repair stage (eval_policy.py --repair 4) on the six 10-cell eval files for
r2_mott_rl, modal_rule, greedy+prune and r0_coord_rl (seed 0), verdict rule fixed in PROPOSALS before any result.
Smoke: scripts/smoke_repair.py -> runs/analysis/smoke_repair.ok. Records -> runs/repair10/eval/<set>__<method>+rep4.json.
Names start "ls10 " so side_slot.py runs them in the 4th CPU slot one at a time between training evals; appended at
the end of the CPU queue, so the driver lane reaches them only after everything else.
Run: .venv/bin/python scripts/queue_repair.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
OUT, LOG, K = "runs/repair10", "agent/logs/repair10.log", 4
SETS = [f"eval_{t}10_{k}" for t in ("ip", "pcb", "pkg") for k in (0, 1)]
METHODS = [("r2_mott_rl", "--policy r2_mott_rl=runs/2026-10-28/r2_mott_rl/best.pt"),
           ("modal_rule", "--methods modal_rule"),
           ("greedy+prune", "--methods gp"),
           ("r0_coord_rl", "--policy r0_coord_rl=runs/2026-10-28/r0_coord_rl/best.pt")]

assert os.path.exists(os.path.join(ROOT, "runs/analysis/smoke_repair.ok")), "run scripts/smoke_repair.py first"
new = [dict(name=f"ls10 repair {m}",
            cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*10_*' {arg} --repair {K} --workers 1 "
                f"--out {OUT}",
            done=[f"{OUT}/eval/{s}__{m}+rep{K}.json" for s in SETS], log=LOG) for m, arg in METHODS]
q = json.load(open(Q))
names = {e["name"] for e in q["cpu"]}
new = [e for e in new if e["name"] not in names]
q["cpu"] += new
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print("appended", len(new), "entries:", [e["name"] for e in new])
