"""Check collect_results.load_evals on server-2 paths (PROPOSALS 11/05 L1-L4 / FRESH) in a temp dir, using one real
server-1 record as the template. Expect: FRESH excluded, L3 duplicate of a server-1 (set, method) dropped, other
server-2 records kept with sec_per_inst blanked, server-1 records untouched.

Run: .venv/bin/python scripts/test_s2_guard.py
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_results as cr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = json.load(open(os.path.join(ROOT, "runs/2026-10-29/r2pk_rl/eval/eval_pcb16_0__r2pk_rl.json")))


def put(path, method, sec=1.5, s=None):
    d = dict(src, method=method, sec_per_inst=sec, set=s or src["set"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path, "w"))


with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    put("runs/2026-10-29/r2pk_rl/eval/eval_pcb16_0__r2pk_rl.json", "r2pk_rl", sec=2.0)        # server 1
    put("runs/2026-11-05fresh/r2pk_rl/eval/eval_pcb16_0__r2pk_rl.json", "r2pk_rl", sec=9.0)  # FRESH: excluded
    put("runs/2026-11-05s2/l3_r2pk/eval/eval_pcb16_0__r2pk_rl.json", "r2pk_rl", sec=9.0)     # L3 dup: dropped
    put("runs/2026-11-05s2/r3pk_rl/eval/eval_pcb16_0__r3pk_rl.json", "r3pk_rl", sec=9.0)     # L3 new: kept, sec blank
    put("runs/2026-10-29/r3_loop_s1_rl/eval/eval_pcb16_0__r3_loop_s1_rl.json", "r3_loop_s1_rl", sec=9.0)  # L1
    put("runs/2026-10-29/r3_loop_s1_rl/eval/eval_pcb10_0__r3_loop_s1_rl.json", "r3_loop_s1_rl", sec=3.0,
        s="eval_pcb10_0")                                                                       # server 1 (10-cell)
    recs = cr.load_evals()
    got = {(r["_file"].split("/")[1], r["method"], r["set"]): r["sec_per_inst"] for r in recs}
    print(got, cr.S2_COUNT)
    assert got[("2026-10-29", "r2pk_rl", "eval_pcb16_0")] == 2.0
    assert not any(k[0] == "2026-11-05fresh" for k in got)
    assert sum(k[1] == "r2pk_rl" for k in got) == 1
    assert math.isnan(got[("2026-11-05s2", "r3pk_rl", "eval_pcb16_0")])
    assert math.isnan(got[("2026-10-29", "r3_loop_s1_rl", "eval_pcb16_0")])
    assert got[("2026-10-29", "r3_loop_s1_rl", "eval_pcb10_0")] == 3.0
    assert cr.S2_COUNT == {"excluded": 1, "server2": 2, "dup_dropped": 1}
    m = cr.merged(recs)[("pcb", 16)]
    assert m["r2pk_rl"]["sec"] == 2.0 and math.isnan(m["r3pk_rl"]["sec"])
print("ok")
