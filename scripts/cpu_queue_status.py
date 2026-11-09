"""List the driver CPU lane (agent/day2_queue.json "cpu") with done / pending status.
Usage: .venv/bin/python scripts/cpu_queue_status.py [substring]  (pending entries only unless --all)"""
import json
import os
import sys

q = json.load(open("agent/day2_queue.json"))
sub = [a for a in sys.argv[1:] if not a.startswith("--")]
for i, e in enumerate(q["cpu"]):
    done = all(os.path.exists(p) for p in e["done"])
    if done and "--all" not in sys.argv:
        continue
    if sub and not any(s in e["name"] for s in sub):
        continue
    miss = [p for p in e["done"] if not os.path.exists(p)]
    print(i, "DONE" if done else f"pending {len(miss)}/{len(e['done'])}", e["name"], "|", e["cmd"][:200])
