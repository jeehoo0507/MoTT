"""Append the P1 runs (11/03 round 36, PROPOSALS 11/03 P1, rule written before training) to the GPU queue.
One change: `--peak_refine` (two extra channels per MoTT mode token, decap_loop/peak_env.py). Input change, so the
full pipeline (distill 0.75 h + rl 1.25 h); pairs r2pk / r2pk_s1 vs r2_mott / r2_mott_s1, lo_pcb_r2pk / lo_pcb_r2pk_s1
vs lo_pcb_r2 / lo_pcb_r2_s1. "after": scripts/smoke_peak_refine.py passed. Idempotent (skips names already queued)."""
import json

P = "agent/day2_queue.json"
ARM = "--model mott --peak_refine"
LO_PCB = " --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0"
AFTER = "runs/analysis/smoke_peak_refine.ok"
NEW = [
    {"name": "r2pk", "arm": ARM, "seed": 0, "after": AFTER},
    {"name": "lo_pcb_r2pk", "arm": ARM + LO_PCB, "seed": 0, "after": AFTER},
    {"name": "r2pk_s1", "arm": ARM, "seed": 1, "after": AFTER},
    {"name": "lo_pcb_r2pk_s1", "arm": ARM + LO_PCB, "seed": 1, "after": AFTER},
]

q = json.load(open(P))
names = {e["name"] for e in q["train"]}
base = next(e for e in q["train"] if e["name"] == "lo_pcb_r2_s1")
assert base["arm"] == "--model mott" + LO_PCB and base["seed"] == 1, base
for e in NEW:
    if e["name"] not in names:
        q["train"].append(e)
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
print([e["name"] for e in q["train"]][-5:])
