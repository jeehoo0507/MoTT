"""Append the difficulty-weighted rl runs (11/03 round 29, QUEUE 11/03 11:40 KST observation 2; PROPOSALS 11/03
"T3") to the GPU queue. One change: `--hard_weight 2` (rl loss terms weighted by 1 + 2 * g+p-count percentile of the
instance in its board kind, mean 1). Same pairs as r2t / r2g: rl 1.25 h only from the same distilled best.pt as
r2_mott (seed 0) and lo_pcb_r2, same seed, default teacher. "after": scripts/smoke_hard_weight.py passed.
Idempotent (skips names already queued)."""
import json

P = "agent/day2_queue.json"
ARM = "--model mott --hard_weight 2"
LO_PCB = " --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0"
AFTER = "runs/analysis/smoke_hard_weight.ok"
NEW = [
    {"name": "r2h", "arm": ARM, "seed": 0, "init": "runs/2026-10-28/r2_mott_distill/best.pt", "after": AFTER},
    {"name": "lo_pcb_r2h", "arm": ARM + LO_PCB, "seed": 0, "init": "runs/2026-10-29/lo_pcb_r2_distill/best.pt",
     "after": AFTER},
]

q = json.load(open(P))
names = {e["name"] for e in q["train"]}
base = next(e for e in q["train"] if e["name"] == "lo_pcb_r2g")
assert base["arm"] == "--model mott --grid_growth 1.0" + LO_PCB, base
for e in NEW:
    if e["name"] not in names:
        q["train"].append(e)
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
print([e["name"] for e in q["train"]][-4:])
