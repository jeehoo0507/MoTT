"""PROGRESS 11/02 round 15: LS 16-cell timing smoke (scripts/smoke_ls16.py) as a 4th-slot filler at the end of the CPU
queue ("ls10 " prefix so side_slot.py picks it up, like the GA smoke). Fig 2 (AGENT_PROMPT) names LS but no 16-cell
LS record exists; the LS16 eval entries are decided from this timing."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
q = json.load(open(Q))
NAME = "ls10 ls16-timing smoke"
if NAME in [e["name"] for e in q["cpu"]]:
    print("already queued")
else:
    q["cpu"].append(dict(name=NAME, cmd=".venv/bin/python scripts/smoke_ls16.py",
                         done=["runs/analysis/smoke_ls16.json"], log="agent/logs/smoke_ls16.log"))
    json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
    print("appended", NAME)
