"""Round 49 (11/05): the scale curve (QUEUE 11/04 18:20 KST, RESULTS "규모 곡선") has the adopted structure r2pk at
16 / 20 cells for seed 0 and 1 only (r2pk_s2 was trained later). Queue the same 12 evaluations for `r2pk_s2_rl` --
same command as the seed 0 / 1 "scale eval ..." entries (--workers 1 --inst_chunk 5, pool_v1, best.pt) -- so the
final structure has 3 seeds at every size. Evaluation only (no training, no new pools, no rule change).

Names start with "ls10 " so the live 4th slot (scripts/side_slot.py FILL prefixes, same precedent as
"ls10 distill-eval ...") runs them between training evals; appended at the end of the CPU list so the driver lane
keeps running the LS16 entries first. Idempotent; atomic queue write.
Run: .venv/bin/python scripts/queue_scale_r2pk_s2.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
N = "r2pk_s2_rl"
D = f"runs/2026-10-29/{N}"
assert os.path.exists(os.path.join(ROOT, D, "best.pt")), D

jobs = []
for n in (16, 20):
    for t in ("pcb", "pkg", "ip"):
        for k in (0, 1):
            s = f"eval_{t}{n}_{k}"
            jobs.append(dict(name=f"ls10 scale-eval {s} {N}",
                             cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets {s} --policy {N}={D}/best.pt "
                                 f"--workers 1 --inst_chunk 5 --out {D}",
                             done=[f"{D}/eval/{s}__{N}.json"], log=f"{D}/eval.log"))
q = json.load(open(Q))
names = {e["name"] for e in q["cpu"]}
q["cpu"] += [e for e in jobs if e["name"] not in names]
tmp = Q + ".tmp"
open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
os.replace(tmp, Q)
print([e["name"] for e in q["cpu"]][-13:])
