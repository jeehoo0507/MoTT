"""PROPOSALS 11/09 T: per-problem time table from runs/speed_1109/eval (decision = method only in the last
cutting-plane round; total = everything incl. exact checks, re-solves and harness, divided by the problem count)."""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "runs/speed_1109/eval"
if "--by_type" in sys.argv:
    # PROPOSALS 11/09 U: pool every file of a type (10 files x 5 problems) per method; ratio = total / greedy+prune total
    agg = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        a = agg.setdefault((r["tech"], r["method"]), [0, 0, 0, 0.0, 0.0, 0])
        a[0] += r["n"]
        a[1] += sum(r["ok"])
        a[2] += sum(r.get("ok_pre", r["ok"]))
        a[3] += r["sec_per_inst"] * r["n"]
        a[4] += r["sec_total"]
        a[5] += 1
    print("| type | method | files | n | success | policy-only success | decision s/problem | total s/problem "
          "| total / greedy+prune |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (t, m), (n, ok, ok0, dec, tot, nf) in sorted(agg.items()):
        gp = agg.get((t, "greedy+prune"))
        ratio = f"{(tot / n) / (gp[4] / gp[0]):.2f}" if gp else "–"
        print(f"| {t} | {m} | {nf} | {n} | {ok / n:.2f} | {ok0 / n:.2f} | {dec / n:.1f} | {tot / n:.1f} | {ratio} |")
    sys.exit(0)
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    r = json.load(open(f))
    s, m = os.path.basename(f)[:-5].split("__", 1)
    rows.append((s.replace("_k10", ""), m, r["n"], sum(r["ok"]) / r["n"], r["sec_per_inst"], r["sec_total"] / r["n"]))
print("| set | method | n | success | decision s/problem | total s/problem |")
print("|---|---|---|---|---|---|")
for s, m, n, ok, dec, tot in rows:
    print(f"| {s} | {m} | {n} | {ok:.2f} | {dec:.1f} | {tot:.1f} |")
