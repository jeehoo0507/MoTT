"""Append the P1 extension runs (PROPOSALS P1 "확장 계획", written 11/04 05:30 UTC before lo_pcb_r2pk_s1) to the GPU
queue, after the P2 runs. Only if RESULTS.md's P1 seed 0+1 pooled verdict is "이김" (refuses otherwise).
Arms = the paired reference's arm + `--peak_refine`. Idempotent (skips names already queued)."""
import json
import re

P = "agent/day2_queue.json"
txt = open("RESULTS.md").read()
m = re.search(r"## P1 `--peak_refine` seed 0·1 묶음 판정.*?\n판정: ([^\n]*)", txt, re.S)
assert m and m.group(1).startswith("**이김**"), f"P1 verdict is not a win: {m.group(1) if m else None}"

PAIRS = [("r2pk_s2", "r2_mott_s2"), ("lo_pcb_r2pk_s2", "lo_pcb_r2_s2"), ("lo_ip_r2pk", "lo_ip_r2"),
         ("lo_pkg_r2pk", "lo_pkg_r2")]
q = json.load(open(P))
by = {e["name"]: e for e in q["train"]}
assert by["lo_pcb_r2dd_s1"] is q["train"][-1] or "lo_pcb_r2dd_s1" in by
for new, ref in PAIRS:
    if new in by:
        continue
    b = by[ref]
    assert b["arm"].startswith("--model mott") and "--peak_refine" not in b["arm"], b
    arm = b["arm"].replace("--model mott", "--model mott --peak_refine", 1)
    q["train"].append({"name": new, "arm": arm, "seed": b["seed"], "after": "runs/analysis/smoke_peak_refine.ok"})
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
for e in q["train"][-6:]:
    print(e["name"], e["seed"], e["arm"])
