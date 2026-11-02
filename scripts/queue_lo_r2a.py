"""Append the leave-one-kind-out runs of r2a (rung 2 + frequency-compatibility bias, --fc_bias) to the GPU queue.
PROGRESS 11/02 round 5: r2a beats r2_mott on all three seeds (same-kind), but has no leave-out run; the arms are the
lo_<t>_r2 arms plus --fc_bias, seed 0. Idempotent (skips names already in the queue)."""
import json

P = "agent/day2_queue.json"
LO = {"pcb": "train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0",
      "ip": "train_pcb*.pkl,train_pkg*.pkl --val_pools val_pcb10_0,val_pkg10_0",
      "pkg": "train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0"}

q = json.load(open(P))
names = {e["name"] for e in q["train"]}
for t in ("pcb", "ip", "pkg"):           # held-out kinds used for the claim first
    base = next(e for e in q["train"] if e["name"] == f"lo_{t}_r2")
    assert base["arm"] == f"--model mott --pool_glob {LO[t]}", base
    name = f"lo_{t}_r2a"
    if name not in names:
        q["train"].append({"name": name, "arm": f"--model mott --fc_bias --pool_glob {LO[t]}", "seed": 0})
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
print([e["name"] for e in q["train"]][-6:])
