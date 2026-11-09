"""Append seeds 1-2 of the four prior-work re-implementations (B-Park, B-Manoharan; PROPOSALS 11/05 B1, written round 46
before any of them ran) to the GPU queue, after the F1 runs. Report-only: no verdict, no change to any earlier verdict.
Arms = the seed-0 run's arm and flags (rl_only / after) unchanged, only the seed changes. Order: the two same-pipeline
or best-performing variants first (bp_park, bm_manoharan_rlonly_l20) at seed 1 then 2, then the other two.
Idempotent (skips names already queued)."""
import json
import os

P = "agent/day2_queue.json"
BASES = ["bp_park", "bm_manoharan_rlonly_l20", "bp_park_rlonly", "bm_manoharan"]
ORDER = [(b, s) for pair in (BASES[:2], BASES[2:]) for s in (1, 2) for b in pair]
q = json.load(open(P))
by = {e["name"]: e for e in q["train"]}
assert "lo_pkg_r2pk_s2" in by, "F1 runs must be queued first"
for base, seed in ORDER:
    new = f"{base}_s{seed}"
    if new in by:
        continue
    e = dict(by[base])
    assert e["seed"] == 0, e
    e.update(name=new, seed=seed)
    q["train"].append(e)
tmp = P + ".tmp"
open(tmp, "w").write(json.dumps(q, indent=1, ensure_ascii=False) + "\n")
os.replace(tmp, P)
for e in q["train"][-10:]:
    print(e["name"], e["seed"], e["arm"], e.get("rl_only", ""))
