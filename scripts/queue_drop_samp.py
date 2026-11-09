"""PROGRESS 11/02 round 6: keep only the pcb best-of-8 sampling jobs of r2a_rl (finish the 50-problem pcb row) and drop
the other samp10 entries (pcb10_0 measured 33.6 s/problem = 84x greedy and 25 GB peak; pkg would exceed the 45 GiB rule)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
KEEP = {"samp10 r2a_rl eval_pcb10_0", "samp10 r2a_rl eval_pcb10_1"}
q = json.load(open(Q))
drop = [e["name"] for e in q["cpu"] if e["name"].startswith("samp10 ") and e["name"] not in KEEP]
q["cpu"] = [e for e in q["cpu"] if e["name"] not in drop]
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print("dropped", drop)
