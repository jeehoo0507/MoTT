"""PROGRESS 11/03 round 24: LS on the 6 side-16 eval files for Fig 2 (AGENT_PROMPT: model vs g+p vs LS).

smoke_ls16 (runs/analysis/smoke_ls16.json, first instance per type, working grid): local_search 88 s (pcb16) /
229 s (pkg16) on top of g+p (54 / 146 s), 0 s on ip16_0 inst 0 (g+p infeasible -> nothing to improve); peak rss
22.7 GB at load. The g+p 16-cell records peak at 19~34 GB, so an LS16 file must not run beside another 16/20-cell
job: these go on the driver CPU lane (one job at a time), appended at the end, i.e. after the 20-cell smoke and the
20-cell evals that scripts/queue_eval20.py inserts right after the smoke. Names start "ls16 " (not "ls10 ") so
side_slot.py never takes them as 4th-slot fillers. Same command as the ls10 entries. Order: pcb, ip, pkg (cheap first).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
BASE = "runs/2026-10-28/baselines"
q = json.load(open(Q))
names = [e["name"] for e in q["cpu"]]
new = []
for s in [f"eval_{t}16_{k}" for t in ("pcb", "ip", "pkg") for k in (0, 1)]:
    new.append(dict(name=f"ls16 {s}",
                    cmd=f".venv/bin/python scripts/eval_policy.py --pool runs/pool_v1 --sets {s} --methods ls --workers 1 --out {BASE}",
                    done=[f"{BASE}/eval/{s}__local_search.json"], log="agent/logs/base16_ls.log"))
new = [e for e in new if e["name"] not in names]
q["cpu"] += new
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print("appended", len(new), "entries:", [e["name"] for e in new])
