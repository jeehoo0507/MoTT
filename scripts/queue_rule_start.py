"""PROPOSALS 11/03 T4 (rule start, eval only): 10-cell 6 files x {r2_mott_rl, r0_coord_rl} seed 0 with
eval_policy.py --start rule (records 'modal_rule><policy>'). Names start "ls10 " -> side_slot.py 4th-slot filler
(10-cell policy evals are light). Inserted after the seed-0 repair entries (T2 R2/R3) and before the seed 1/2 repair
widening. Records -> runs/rulestart10/eval/.
Run: .venv/bin/python scripts/queue_rule_start.py   (idempotent; needs runs/analysis/smoke_rule_start.ok)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
SETS10 = [f"eval_{t}10_{k}" for t in ("ip", "pcb", "pkg") for k in (0, 1)]
POL = [("r2_mott_rl", "runs/2026-10-28/r2_mott_rl/best.pt"), ("r0_coord_rl", "runs/2026-10-28/r0_coord_rl/best.pt")]

assert os.path.exists(os.path.join(ROOT, "runs/analysis/smoke_rule_start.ok")), "smoke first"
side = [dict(name=f"ls10 rulestart {m}",
             cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*10_*' --policy {m}={p} --start rule "
                 f"--workers 1 --out runs/rulestart10",
             done=[f"runs/rulestart10/eval/{s}__modal_rule>{m}.json" for s in SETS10], log="agent/logs/rulestart10.log")
        for m, p in POL]
q = json.load(open(Q))
names = [e["name"] for e in q["cpu"]]
at = names.index("ls10 repair r0_coord_rl") + 1
for e in reversed(side):
    if e["name"] not in names:
        q["cpu"].insert(at, e)
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print([e["name"] for e in q["cpu"]][at - 2:at + 4])
