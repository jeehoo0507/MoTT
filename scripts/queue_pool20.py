"""PROGRESS 11/02 round 9: 20-cell eval pools (QUEUE 10/29 03:10 item 5: 8 instances per file = PROPOSALS (c); QUEUE
10/28 14:20: 20-cell boards last). One CPU-lane entry, 1 worker (driver lane slot, CPU limit 4 kept), inserted right
after the last eval16 entry so it runs once the 16-cell evals are done; build_pools skips files that exist, so a pod
restart only loses the file in progress. Policy / baseline evals on these pools are queued after the files exist."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
q = json.load(open(Q))
names = [e["name"] for e in q["cpu"]]
NAME = "pool20 eval k8"
if NAME in names:
    print("already queued")
else:
    done = [f"runs/pool_v1/eval_{t}20_{k}.pkl" for t in ("pcb", "pkg", "ip") for k in (0, 1)]
    e = dict(name=NAME, cmd=".venv/bin/python scripts/build_pools.py --only eval --eval_sizes 20 --eval_n 8 --workers 1",
             log="agent/logs/build_pool20.log", done=done)
    at = max(i for i, n in enumerate(names) if n.startswith("eval16 ")) + 1
    q["cpu"].insert(at, e)
    json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
    print("inserted", NAME, "at", at, "after", names[at - 1])
