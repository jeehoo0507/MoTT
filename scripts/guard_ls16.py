"""Round 45 (11/05): `ls16 eval_pcb16_0` was killed by the kernel OOM killer twice (11/03 03:55 next to pool20 + a GPU
diag; 11/05 between 02:24 and 03:27 UTC with only RL training + 2 certify workers beside it, pod 14.8 GiB at 02:24).
Same problem twice -> STUCK (AGENT_PROMPT rule 0): give up on that file and say "메모리로 생략".

  - wrap every pending `ls16 ...` driver entry in scripts/mem_guard.py (as base20 greedy+prune): the guard kills the
    job and writes `<record>.json.memskip` if the pod goes above the 45 GiB rule, instead of the kernel picking a victim
  - write the memskip marker for eval_pcb16_0 now (reason "kernel oom_kill x2"), so a driver restart does not run it a
    third time (mem_guard exits 3 at once when the marker exists)
Commands are otherwise unchanged. Idempotent; atomic queue write."""
import json
import os
import time

P = "agent/day2_queue.json"
GUARD = ".venv/bin/python scripts/mem_guard.py "
q = json.load(open(P))
for e in q["cpu"]:
    if not e["name"].startswith("ls16 ") or e["cmd"].startswith(GUARD):
        continue
    marker = e["done"][0] + ".memskip"             # not *.json: eval-record globs must not see it
    e["cmd"] = GUARD + marker + " " + e["cmd"]
tmp = P + ".tmp"
open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
os.replace(tmp, P)

rec = "runs/2026-10-28/baselines/eval/eval_pcb16_0__local_search.json"
marker = rec + ".memskip"
if not os.path.exists(rec) and not os.path.exists(marker):
    json.dump(dict(reason="memory", detail="kernel oom_kill x2 (2026-11-03 03:55 UTC; 2026-11-05 02:24-03:27 UTC)",
                   limit_gib=60.0, cmd=[e["cmd"] for e in q["cpu"] if e["name"] == "ls16 eval_pcb16_0"],
                   time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())),
              open(marker, "w"), indent=1)
for e in q["cpu"]:
    if e["name"].startswith("ls16 "):
        print(e["name"], "|", e["cmd"][:90], "| memskip" if os.path.exists(e["done"][0] + ".memskip") else "")
