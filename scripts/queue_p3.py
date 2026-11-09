"""Append the P3 runs (PROPOSALS 11/04 P3, written 11/04 08:56 UTC before the P2 rl results) to the GPU queue, after the
P1 extension runs. Only if RESULTS.md's P1 seed 0+1 pooled verdict is "이김" and the P2 verdict exists and is not
"해로움" (refuses otherwise). Arms = the paired P1 run's arm + `--dedup_modes`. Idempotent (skips names already queued)."""
import json
import re
import sys

P = "agent/day2_queue.json"
txt = open("RESULTS.md").read()


def verdict(tag, flag):
    m = re.search(rf"## {tag} `{flag}` seed 0·1 묶음 판정.*?\n판정: ([^\n]*)", txt, re.S)
    return m.group(1) if m else None


v1, v2 = verdict("P1", "--peak_refine"), verdict("P2", "--dedup_modes")
assert v1 and v1.startswith("**이김**"), f"P1 verdict is not a win: {v1}"
assert v2 and not v2.startswith("seed 0·1 네 run이"), f"P2 not judged yet: {v2}"
if v2.startswith("**해로움**"):
    sys.exit(f"P2 verdict is harm -> P3 cancelled (PROPOSALS P3): {v2}")

PAIRS = [("r2pkdd", "r2pk"), ("lo_pcb_r2pkdd", "lo_pcb_r2pk"), ("r2pkdd_s1", "r2pk_s1"),
         ("lo_pcb_r2pkdd_s1", "lo_pcb_r2pk_s1")]
q = json.load(open(P))
by = {e["name"]: e for e in q["train"]}
assert "lo_pkg_r2pk" in by, "P1 extension must be queued first"
for new, ref in PAIRS:
    if new in by:
        continue
    b = by[ref]
    assert "--peak_refine" in b["arm"] and "--dedup_modes" not in b["arm"], b
    arm = b["arm"].replace("--peak_refine", "--peak_refine --dedup_modes", 1)
    q["train"].append({"name": new, "arm": arm, "seed": b["seed"], "after": "runs/analysis/smoke_pkdd.ok"})
open(P, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
for e in q["train"][-8:]:
    print(e["name"], e["seed"], e["arm"])
