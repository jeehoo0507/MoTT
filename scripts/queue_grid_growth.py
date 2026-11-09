"""Append the --grid_growth 1.0 runs (11/02 round 14, diagnosis H2: the other fix named in round 1, "relax the grid
cap") to the GPU queue, after lo_pcb_r2m, so the GPU does not go idle when the queue ends (~01:00 UTC).
Same pairs as r2c / r2m: rl 1.25 h only from the same distilled best.pt as r2_mott (seed 0) and lo_pcb_r2, so the
working grid's growth cap (0.5 -> 1.0 of the initial F) is the only change. Idempotent (skips names already queued)."""
import json

P = "agent/day2_queue.json"
ARM = "--model mott --grid_growth 1.0"
LO_PCB = " --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0"
NEW = [
    {"name": "r2g", "arm": ARM, "seed": 0, "init": "runs/2026-10-28/r2_mott_distill/best.pt"},
    {"name": "lo_pcb_r2g", "arm": ARM + LO_PCB, "seed": 0, "init": "runs/2026-10-29/lo_pcb_r2_distill/best.pt"},
]

q = json.load(open(P))
names = {e["name"] for e in q["train"]}
base = next(e for e in q["train"] if e["name"] == "lo_pcb_r2m")
assert base["arm"] == "--model mott --feas_margin 0.021189" + LO_PCB, base
assert q["train"][-1]["name"] == "lo_pcb_r2m", q["train"][-1]["name"]
for e in NEW:
    if e["name"] not in names:
        q["train"].append(e)
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
print([e["name"] for e in q["train"]][-6:])
