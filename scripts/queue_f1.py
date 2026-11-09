"""Append the F1 runs (PROPOSALS 11/04 F1, written round 43 before any of them ran) to the GPU queue, after the P3 runs:
held-out ip / pkg seeds 1-2 of the adopted final structure r2pk (`--peak_refine`). Only if RESULTS.md's P1 extension
verdict adopts it ("이김 (seed 3개 확인)" + held-out ip / pkg above -0.16; refuses otherwise).
Arms = the seed-0 run's arm (lo_ip_r2pk / lo_pkg_r2pk), only the seed changes. Idempotent (skips names already queued)."""
import json
import os
import re

P ="agent/day2_queue.json"
txt = open("RESULTS.md").read()
m = re.search(r"### P1 확장: .*?\n판정: ([^\n]*)", txt, re.S)
assert m and "이김 (seed 3개 확인)" in m.group(1) and "최종 구조 후보" in m.group(1), \
    f"P1 extension verdict does not adopt r2pk: {m.group(1) if m else None}"

RUNS = [("lo_ip_r2pk_s1", "lo_ip_r2pk", 1), ("lo_ip_r2pk_s2", "lo_ip_r2pk", 2),
        ("lo_pkg_r2pk_s1", "lo_pkg_r2pk", 1), ("lo_pkg_r2pk_s2", "lo_pkg_r2pk", 2)]
q = json.load(open(P))
by = {e["name"]: e for e in q["train"]}
assert "lo_pcb_r2pkdd_s1" in by, "P3 runs must be queued first"
for new, ref, seed in RUNS:
    if new in by:
        continue
    b = by[ref]
    assert "--peak_refine" in b["arm"] and "--dedup_modes" not in b["arm"] and b["seed"] == 0, b
    q["train"].append({"name": new, "arm": b["arm"], "seed": seed, "after": b["after"]})
tmp = P + ".tmp"
open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
os.replace(tmp, P)
for e in q["train"][-6:]:
    print(e["name"], e["seed"], e["arm"])
