"""Append the --feas_margin runs (11/02 round 7, diagnosis H2 cheap fix) to the GPU queue, after lo_pkg_r2a.
Same pairs as r2c: rl 1.25 h only from the same distilled best.pt as r2_mott (seed 0) and lo_pcb_r2, so the rl reward's
success margin is the only change. Idempotent (skips names already in the queue)."""
import json

P = "agent/day2_queue.json"
ARM = "--model mott --feas_margin 0.021189"     # log10(1.05)
LO_PCB = " --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0"
NEW = [
    {"name": "r2m", "arm": ARM, "seed": 0, "init": "runs/2026-10-28/r2_mott_distill/best.pt",
     "after": "runs/analysis/smoke_feas_margin.ok"},
    {"name": "lo_pcb_r2m", "arm": ARM + LO_PCB, "seed": 0, "init": "runs/2026-10-29/lo_pcb_r2_distill/best.pt",
     "after": "runs/analysis/smoke_feas_margin.ok"},
]

q = json.load(open(P))
names = {e["name"] for e in q["train"]}
base = next(e for e in q["train"] if e["name"] == "lo_pcb_r2c")
assert base["arm"] == "--model mott --certify 2" + LO_PCB, base
for e in NEW:
    if e["name"] not in names:
        q["train"].append(e)
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
print([e["name"] for e in q["train"]][-6:])
