"""Per-instance (ok, count) of eval records, side by side (read only).

Run: .venv/bin/python scripts/rec_counts.py runs/.../eval_pcb16_0__r2pk_rl.json runs/.../eval_pcb16_0__r2pk_s1_rl.json
"""
import json
import sys

recs = [json.load(open(f)) for f in sys.argv[1:]]
print("idx " + " ".join(f"{r['method']:>16s}" for r in recs))
for i in range(min(r["n"] for r in recs)):
    print(f"{i:3d} " + " ".join(f"{('ok' if r['ok'][i] else '--'):>4s} {r['count'][i]:11d}" for r in recs))
