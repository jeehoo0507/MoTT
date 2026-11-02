"""PROGRESS 11/02 round 16: the rest of the 20-cell evals (QUEUE 10/29 03:10 item 5), in the round-11 order: the 5
policies (r2a, r0~r3) -> modal_rule -> g+p, file by file. LS / E0+ wait for the g+p 20-cell memory. Run only after the
smoke record (scripts/queue_eval20_smoke.py) exists: it prints that record's peak rss / time so the round can check it
against the 45 GiB rule before queueing. Same commands as the eval16 entries; inserted after the smoke entry."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
SMOKE = os.path.join(ROOT, "runs/2026-10-29/r2a_rl/eval/eval_pcb20_0__r2a_rl.json")
if not os.path.exists(SMOKE):
    sys.exit("smoke record missing: " + SMOKE)
r = json.load(open(SMOKE))
print(f"smoke: peak_rss {r['peak_rss_mb'] / 1024:.1f} GB, {r['sec_per_inst']:.1f} s/inst, total {r['sec_total']:.0f} s")
POL = dict(r2a_rl="runs/2026-10-29/r2a_rl", r0_coord_rl="runs/2026-10-28/r0_coord_rl",
           r1_modal_rl="runs/2026-10-28/r1_modal_rl", r2_mott_rl="runs/2026-10-28/r2_mott_rl",
           r3_loop_rl="runs/2026-10-28/r3_loop_rl")
BASE = "runs/2026-10-28/baselines"
SETS = [f"eval_{t}20_{k}" for t in ("pcb", "pkg", "ip") for k in (0, 1)]
q = json.load(open(Q))
names = [e["name"] for e in q["cpu"]]
new = []
for s in SETS:
    for n, run in POL.items():
        new.append(dict(name=f"eval20 {s} {n}",
                        cmd=f".venv/bin/python scripts/eval_policy.py --pool runs/pool_v1 --sets {s} --policy {n}={run}/best.pt "
                            f"--workers 1 --inst_chunk 5 --out {run}",
                        done=[f"{run}/eval/{s}__{n}.json"], log=f"{run}/eval.log"))
for m, rec in (("modal_rule", "modal_rule"), ("gp", "greedy+prune")):
    for s in SETS:
        new.append(dict(name=f"base20 {s} {rec}",
                        cmd=f".venv/bin/python scripts/eval_policy.py --pool runs/pool_v1 --sets {s} --methods {m} --workers 1 --out {BASE}",
                        done=[f"{BASE}/eval/{s}__{rec}.json"], log="agent/logs/base20.log"))
new = [e for e in new if e["name"] not in names]
at = names.index("eval20 eval_pcb20_0 r2a_rl") + 1
q["cpu"][at:at] = new
json.dump(q, open(Q, "w"), indent=1, ensure_ascii=False)
print("inserted", len(new), "entries at", at, ":", [e["name"] for e in new][:3], "...")
