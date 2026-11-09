"""PROGRESS 11/02 round 6: 16-cell eval of the current best candidate r2a_rl (seed 0) for Fig 2 (QUEUE 10/29 03:10
item 4: g+p, E0+, ladder rl and the final candidate). Same command as the ladder's eval16 entries; inserted right after
the last ladder eval16 entry so the driver's CPU lane runs them next."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
RUN = "runs/2026-10-29/r2a_rl"
q = json.load(open(Q))
names = [e["name"] for e in q["cpu"]]
new = []
for s in ("eval_pcb16_0", "eval_pcb16_1", "eval_pkg16_0", "eval_pkg16_1", "eval_ip16_0", "eval_ip16_1"):
    n = f"eval16 {s} r2a_rl"
    if n in names:
        continue
    new.append(dict(name=n, cmd=f".venv/bin/python scripts/eval_policy.py --pool runs/pool_v1 --sets {s} --policy r2a_rl={RUN}/best.pt "
                                f"--workers 1 --inst_chunk 5 --out {RUN}",
                    done=[f"{RUN}/eval/{s}__r2a_rl.json"], log=f"{RUN}/eval.log"))
at = max(i for i, n in enumerate(names) if n.startswith("eval16 ")) + 1
q["cpu"][at:at] = new
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print("inserted", [e["name"] for e in new], "at", at, "after", names[at - 1])
