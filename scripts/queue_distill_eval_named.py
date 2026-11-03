"""Queue the 10-cell exact eval of named distill checkpoints as 4th-slot fillers (same command as
queue_distill_eval.py: eval10, inst_chunk 5, workers 1; names "ls10 distill-eval <run>").
Only after <run>/best.pt exists (side_slot skips a filler whose checkpoint is missing as failed).
Run: .venv/bin/python scripts/queue_distill_eval_named.py r2pk_distill [r2pk_s1_distill ...]   (idempotent)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
SETS10 = [f"eval_{t}10_{k}" for t in ("ip", "pcb", "pkg") for k in (0, 1)]

jobs = []
for n in sys.argv[1:]:
    d = f"runs/2026-10-29/{n}"
    if not os.path.exists(os.path.join(ROOT, d, "best.pt")):
        print("skip (no best.pt yet):", n)
        continue
    jobs.append(dict(name=f"ls10 distill-eval {n}",
                     cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*10_*' --policy {n}={d}/best.pt "
                         f"--workers 1 --inst_chunk 5 --out {d}",
                     done=[f"{d}/eval/{x}__{n}.json" for x in SETS10], log="agent/logs/distill_eval.log"))
q = json.load(open(Q))
names = {e["name"] for e in q["cpu"]}
q["cpu"] += [e for e in jobs if e["name"] not in names]
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print([e["name"] for e in q["cpu"]][-4:])
