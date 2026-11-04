"""Append the P2 runs (11/04 round 37, PROPOSALS 11/04 P2, rule written before training) to the GPU queue, behind P1.
One change: `--dedup_modes` (MoTT mode tokens on distinct working-grid frequencies, decap_loop/dedup_env.py). Input
change, so the full pipeline (distill 0.75 h + rl 1.25 h); pairs r2dd / r2dd_s1 vs r2_mott / r2_mott_s1, lo_pcb_r2dd /
lo_pcb_r2dd_s1 vs lo_pcb_r2 / lo_pcb_r2_s1. "after": scripts/smoke_dedup_modes.py passed. Idempotent."""
import json

P = "agent/day2_queue.json"
ARM = "--model mott --dedup_modes"
LO_PCB = " --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0"
AFTER = "runs/analysis/smoke_dedup_modes.ok"
NEW = [
    {"name": "r2dd", "arm": ARM, "seed": 0, "after": AFTER},
    {"name": "lo_pcb_r2dd", "arm": ARM + LO_PCB, "seed": 0, "after": AFTER},
    {"name": "r2dd_s1", "arm": ARM, "seed": 1, "after": AFTER},
    {"name": "lo_pcb_r2dd_s1", "arm": ARM + LO_PCB, "seed": 1, "after": AFTER},
]

q = json.load(open(P))
names = [e["name"] for e in q["train"]]
assert names.index("lo_pcb_r2pk_s1") == len(names) - 1, names[-5:]     # P1 stays in front
base = next(e for e in q["train"] if e["name"] == "lo_pcb_r2_s1")
assert base["arm"] == "--model mott" + LO_PCB and base["seed"] == 1, base
for e in NEW:
    if e["name"] not in names:
        q["train"].append(e)
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
print([e["name"] for e in q["train"]][-9:])
