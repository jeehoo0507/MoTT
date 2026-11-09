"""Append the RL-only replicate runs (11/03 round 32, PROPOSALS 11/03 "T5") to the GPU queue. No change in the arm:
the same distilled best.pt as r2_mott (seed 0) and lo_pcb_r2, rl 1.25 h only, `--seed 1` instead of 0. Measures how
far two rl runs from the same init differ on the paired eval (training noise the bootstrap CI does not contain).
Idempotent (skips names already queued)."""
import json

P = "agent/day2_queue.json"
ARM = "--model mott"
LO_PCB = " --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0"
NEW = [
    {"name": "r2s1", "arm": ARM, "seed": 1, "init": "runs/2026-10-28/r2_mott_distill/best.pt"},
    {"name": "lo_pcb_r2s1", "arm": ARM + LO_PCB, "seed": 1, "init": "runs/2026-10-29/lo_pcb_r2_distill/best.pt"},
    # T5b (PROPOSALS 11/03, round 33): second replicate, fallback at the queue end
    {"name": "r2s2", "arm": ARM, "seed": 2, "init": "runs/2026-10-28/r2_mott_distill/best.pt"},
    {"name": "lo_pcb_r2s2", "arm": ARM + LO_PCB, "seed": 2, "init": "runs/2026-10-29/lo_pcb_r2_distill/best.pt"},
]

q = json.load(open(P))
names = {e["name"] for e in q["train"]}
base = next(e for e in q["train"] if e["name"] == "lo_pcb_r2h")
assert base["arm"] == "--model mott --hard_weight 2" + LO_PCB, base
for e in NEW:
    if e["name"] not in names:
        q["train"].append(e)
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
print([e["name"] for e in q["train"]][-4:])
