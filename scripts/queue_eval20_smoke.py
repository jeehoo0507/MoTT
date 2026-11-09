"""PROGRESS 11/02 round 16: first 20-cell eval record (QUEUE 10/29 03:10 item 5), one entry only, as a memory / time
measurement before the rest of the 20-cell evals are queued (QUEUE 10/28 14:20: 20-cell boards were the suspected
OOM cause; pod limit 60 GiB, no new process above 45 GiB). Current best candidate r2a_rl on eval_pcb20_0 (the only
20-cell file that exists), same command as its eval16 entries (--workers 1 --inst_chunk 5). Inserted right after
`pool20 eval k8` so the driver's CPU lane runs it once the pool files are built; the record is a normal eval record."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
RUN = "runs/2026-10-29/r2a_rl"
S = "eval_pcb20_0"
q = json.load(open(Q))
names = [e["name"] for e in q["cpu"]]
NAME = f"eval20 {S} r2a_rl"
if NAME in names:
    print("already queued")
else:
    e = dict(name=NAME, cmd=f".venv/bin/python scripts/eval_policy.py --pool runs/pool_v1 --sets {S} --policy r2a_rl={RUN}/best.pt "
                            f"--workers 1 --inst_chunk 5 --out {RUN}",
             done=[f"{RUN}/eval/{S}__r2a_rl.json"], log=f"{RUN}/eval.log")
    at = names.index("pool20 eval k8") + 1
    q["cpu"].insert(at, e)
    json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
    print("inserted", NAME, "at", at, "after", names[at - 1])
