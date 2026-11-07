"""Round 74 (11/07): QUEUE 11/07 13:25 KST / PROPOSALS "11/07 G1 보충" -- report-only 16 / 20-cell evals of the G1
candidate `r2pkfc_rl` (seed 0), NOT part of the G1 verdict. Same command as the scale-curve evals
(--workers 1 --inst_chunk 5, pool_v1, best.pt). Order as the human gave it: eval_pkg16_0, pkg16_1, pkg20_0, pkg20_1
first, then pcb / ip 16 and 20.

Names start with "ls10 scale-eval" so the 4th slot (scripts/side_slot.py) runs them only while no training eval is
pending (lo_pkg_r2pkfc 10-cell eval = G1 verdict goes first) and never while a driver LS16 job is alive (16-cell eval
peak ~22 GB + pod ~24 GiB > 45 GiB -> mem_guard kill, 11/05 round 50). The 4 pkg entries go before the older pending
r2pk_s2 filler (human said "바로"); pcb / ip go at the end. Idempotent; atomic queue write.
Run: .venv/bin/python scripts/queue_scale_r2pkfc.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "agent/day2_queue.json")
PY = ".venv/bin/python"
N = "r2pkfc_rl"
D = f"runs/2026-10-29/{N}"
assert os.path.exists(os.path.join(ROOT, D, "best.pt")), D


def job(s):
    return dict(name=f"ls10 scale-eval {s} {N}",
                cmd=f"{PY} scripts/eval_policy.py --pool runs/pool_v1 --sets {s} --policy {N}={D}/best.pt "
                    f"--workers 1 --inst_chunk 5 --out {D}",
                done=[f"{D}/eval/{s}__{N}.json"], log=f"{D}/eval.log")


first = [job(f"eval_pkg{n}_{k}") for n in (16, 20) for k in (0, 1)]
rest = [job(f"eval_{t}{n}_{k}") for n in (16, 20) for t in ("pcb", "ip") for k in (0, 1)]
q = json.load(open(Q))
names = {e["name"] for e in q["cpu"]}
first = [e for e in first if e["name"] not in names]
rest = [e for e in rest if e["name"] not in names]
# before the first pending "ls10 scale-eval" filler (else at the end)
at = next((i for i, e in enumerate(q["cpu"]) if e["name"].startswith("ls10 scale-eval")
           and not all(os.path.exists(os.path.join(ROOT, p)) for p in e["done"])), len(q["cpu"]))
q["cpu"][at:at] = first
q["cpu"] += rest
tmp = Q + ".tmp"
open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
os.replace(tmp, Q)
print([e["name"] for e in q["cpu"]][-18:])
