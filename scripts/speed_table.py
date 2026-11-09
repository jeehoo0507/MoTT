"""PROPOSALS 11/09 T: per-problem time table from runs/speed_1109/eval (decision = method only in the last
cutting-plane round; total = everything incl. exact checks, re-solves and harness, divided by the problem count)."""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "runs/speed_1109/eval"
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    r = json.load(open(f))
    s, m = os.path.basename(f)[:-5].split("__", 1)
    rows.append((s.replace("_k10", ""), m, r["n"], sum(r["ok"]) / r["n"], r["sec_per_inst"], r["sec_total"] / r["n"]))
print("| set | method | n | success | decision s/problem | total s/problem |")
print("|---|---|---|---|---|---|")
for s, m, n, ok, dec, tot in rows:
    print(f"| {s} | {m} | {n} | {ok:.2f} | {dec:.1f} | {tot:.1f} |")
