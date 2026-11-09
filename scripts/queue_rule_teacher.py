"""Append the rule-augmented teacher runs (11/03 round 27, QUEUE 11/03 11:40 KST observation 3; PROPOSALS 11/03
"T1") to the GPU queue. One change: the teacher store (`--teacher_dir data/teacher_rcp`: per training instance the
certified solution with fewer decaps among {stored LS/g+p teacher, modal_rule inside 4-round cutting-plane},
scripts/make_rule_teacher.py --cp 4). Same pairs as r2c / r2m / r2g: rl 1.25 h only from the same distilled best.pt
as r2_mott (seed 0) and lo_pcb_r2, same seed. "after": the teacher files must be complete and pass a 1-iteration CPU
train.py smoke (and those CPU processes be gone, so training's 2 exact-check workers keep the CPU count at 4).
Idempotent (skips names already queued)."""
import json

P = "agent/day2_queue.json"
ARM = "--model mott --teacher_dir data/teacher_rcp"
LO_PCB = " --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0"
AFTER = "runs/analysis/smoke_rule_teacher.ok"   # written after the teacher build + a CPU train.py smoke on it
NEW = [
    {"name": "r2t", "arm": ARM, "seed": 0, "init": "runs/2026-10-28/r2_mott_distill/best.pt", "after": AFTER},
    {"name": "lo_pcb_r2t", "arm": ARM + LO_PCB, "seed": 0, "init": "runs/2026-10-29/lo_pcb_r2_distill/best.pt",
     "after": AFTER},
    # PROPOSALS T1': the same teacher in distill + rl (0.75 h + 1.25 h), vs r2_mott_rl / lo_pcb_r2_rl (full runs)
    {"name": "r2tf", "arm": ARM, "seed": 0, "after": AFTER},
    {"name": "lo_pcb_r2tf", "arm": ARM + LO_PCB, "seed": 0, "after": AFTER},
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
