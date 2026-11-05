"""Dry check of side_slot's decision for given training names (no job is started).
Run: .venv/bin/python scripts/side_slot_dry.py bp_park_s1 ..."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_day1 import alive  # noqa: E402
from pipeline_day2 import eval_jobs, exists, queue, rl_done  # noqa: E402

evals = {j["name"]: j for j in eval_jobs()}
for n in sys.argv[1:]:
    j = evals.get(f"eval10 {n}_rl")
    print(n, "rl_done", rl_done(n), "eval job", j is not None)
    if j:
        print("  done files", [(p, exists(p)) for p in j["done"]])
        print("  alive", alive(j["cmd"].replace("'", "")))
        print("  cmd", j["cmd"][:300])
fill = [j for j in queue().get("cpu", []) if j["name"].startswith(("ls10 ", "ga10 ", "samp10 "))]
print("alive fillers", [j["name"] for j in fill if alive(j["cmd"].replace("'", ""))])
print("eval job names with bp_/bm_", [k for k in evals if "bp_" in k or "bm_" in k])
