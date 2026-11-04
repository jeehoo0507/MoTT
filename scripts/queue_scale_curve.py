"""QUEUE 11/04 18:20 KST (scale curve) + 18:15 KST (loop R sweep at 16/20 cells): reorder / extend the driver CPU lane.
Pending order after this script:
  1. base20 modal_rule (6)  2. remaining eval20 policy evals  3. base20 greedy+prune (6), each wrapped in
  scripts/mem_guard.py (killed + `<record>.json.memskip` marker if the pod exceeds 45 GiB)  4. r2pk_rl / r2pk_s1_rl on
  eval_{pcb,pkg,ip}{16,20}_{0,1}  5. r3_loop_rl @R2/@R4/@R16 on the same 12 sets  6. every other pending entry
  (ls16 ...) in its old order.
Done entries keep their place in front. Idempotent (re-running gives the same order, no duplicates).
Optional: --r2dd also appends r2dd_rl on the 16/20 list (only after the P2 rule says "이김")."""
import json
import os
import re
import sys

P = "agent/day2_queue.json"
POOL = "runs/pool_v1"
SETS = [f"eval_{t}{n}_{k}" for n in (16, 20) for t in ("pcb", "pkg", "ip") for k in (0, 1)]
q = json.load(open(P))
cpu = q["cpu"]
names = {e["name"] for e in cpu}


def done(e):
    return all(os.path.exists(p) for p in e["done"])


def pol_entry(run, day, s, pols, log):
    spec = " ".join(f"{p}=runs/{day}/{run}/best.pt" for p in pols)
    return dict(name=f"scale {s} {' '.join(pols)}",
                cmd=f".venv/bin/python scripts/eval_policy.py --pool {POOL} --sets {s} --policy {spec} --workers 1 "
                    f"--inst_chunk 5 --out runs/{day}/{run}",
                done=[f"runs/{day}/{run}/eval/{s}__{p}.json" for p in pols], log=log)


new = []
for run in ("r2pk_rl", "r2pk_s1_rl"):
    assert os.path.exists(f"runs/2026-10-29/{run}/best.pt"), run
for s in SETS:
    for run in ("r2pk_rl", "r2pk_s1_rl"):
        new.append(pol_entry(run, "2026-10-29", s, [run], f"runs/2026-10-29/{run}/eval.log"))
if "--r2dd" in sys.argv:
    m = re.search(r"## P2 `--dedup_modes` seed 0·1 묶음 판정.*?\n판정: ([^\n]*)", open("RESULTS.md").read(), re.S)
    assert m and m.group(1).startswith("**이김**"), f"P2 verdict is not a win: {m.group(1) if m else None}"
    for s in SETS:
        new.append(pol_entry("r2dd_rl", "2026-10-29", s, ["r2dd_rl"], "runs/2026-10-29/r2dd_rl/eval.log"))
# all three R share the one best.pt (as the 10-cell "day1 eval10 r3@R" entry)
new += [pol_entry("r3_loop_rl", "2026-10-28", s, ["r3_loop_rl@R2", "r3_loop_rl@R4", "r3_loop_rl@R16"],
                  "runs/2026-10-28/r3_loop_rl/eval.log") for s in SETS]

old_new = [e for e in cpu if e["name"].startswith("scale ")]
cpu = [e for e in cpu if not e["name"].startswith("scale ")]
keep_new = {e["name"]: e for e in old_new}
new = [keep_new.get(e["name"], e) for e in new]
for e in old_new:
    if e["name"] not in {n["name"] for n in new}:
        new.append(e)

front = [e for e in cpu if done(e)]
pend = [e for e in cpu if not done(e)]
mr20 = [e for e in pend if e["name"].startswith("base20 ") and e["name"].endswith(" modal_rule")]
ev20 = [e for e in pend if e["name"].startswith("eval20 ")]
gp20 = [e for e in pend if e["name"].startswith("base20 ") and e["name"].endswith(" greedy+prune")]
for e in gp20:
    cmd = e["cmd"]
    if cmd.startswith(".venv/bin/python scripts/mem_guard.py "):
        cmd = cmd.split(" ", 3)[3]                 # drop "<py> mem_guard.py <marker> "
    marker = e["done"][0] + ".memskip"             # not *.json: eval-record globs must not see it
    e["cmd"] = f".venv/bin/python scripts/mem_guard.py {marker} " + cmd
picked = {id(e) for e in mr20 + ev20 + gp20}
rest = [e for e in pend if id(e) not in picked]
q["cpu"] = front + mr20 + ev20 + gp20 + new + rest
assert len({e["name"] for e in q["cpu"]}) == len(q["cpu"]), "duplicate names"
tmp = P + ".tmp"
open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
os.replace(tmp, P)
for i, e in enumerate(q["cpu"]):
    if not done(e):
        print(i, e["name"])
