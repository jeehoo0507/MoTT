"""Round 33 diagnosis (eval only, no training): 10-cell exact eval of the distill-only checkpoints of seeds 1/2
for r0_coord, r1_modal, r2_mott (seed 0 already has them: pkg r0 0.90 / r1 0.90 / r2 0.70).
Question (written before any seed 1/2 result): is r2's pkg weakness already present after distillation in every
seed? Use (pre-registered in PROPOSALS 11/03 D1): if r2_distill pkg success is below both r0_distill and r1_distill
in all 3 seeds, "MoTT imitates the teacher worse on pkg" goes into RESULTS as a finding and the distill stage becomes
the target of the next MoTT candidate; otherwise the seed-0 gap is written down as one seed, not a finding.
Same command as the driver's eval10 (inst_chunk 5, workers 1); names start "ls10 " -> side_slot.py 4th-slot filler.
Run: .venv/bin/python scripts/queue_distill_eval.py   (idempotent)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
SETS10 = [f"eval_{t}10_{k}" for t in ("ip", "pcb", "pkg") for k in (0, 1)]

jobs = []
for s in (1, 2):
    for r in ("r2_mott", "r0_coord", "r1_modal"):
        n = f"{r}_s{s}_distill"
        d = f"runs/2026-10-29/{n}"
        jobs.append(dict(name=f"ls10 distill-eval {n}",
                         cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*10_*' --policy {n}={d}/best.pt "
                             f"--workers 1 --inst_chunk 5 --out {d}",
                         done=[f"{d}/eval/{x}__{n}.json" for x in SETS10], log="agent/logs/distill_eval.log"))
for j in jobs:
    assert os.path.exists(os.path.join(ROOT, j["cmd"].split("=")[-1].split()[0])), j["name"]
q = json.load(open(Q))
names = {e["name"] for e in q["cpu"]}
q["cpu"] += [e for e in jobs if e["name"] not in names]
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print([e["name"] for e in q["cpu"]][-8:])
