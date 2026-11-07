"""PROPOSALS 11/07 R screening (rule fixed by the human before any R run): held-out pkg, seed 0-2 pooled, policy
alone, paired on the same problems. Candidate X = lo_pkg_rnd{,_s1,_s2} (MoTT+P1, RL pools pcb+ip+rnd, pkg never
seen); pair = lo_pkg_r2pk{,_s1,_s2} (same model, RL pools pcb+ip). Report-only extra: lo_pkg_r0{,_s1,_s2} (coords)
and lo_pkg_rnd_r0{,_s1,_s2} (coords + rnd) for the 2x2 reading.
  (i)   pooled held-out pkg diff X - r2pk >= +0.10 (point estimate; bootstrap CI printed)
  (ii)  pooled held-in pcb and ip diff X - r2pk >= -0.06 each
  (iii) pooled count ratio (problems both solve, per kind) <= 1.05 for pcb, pkg, ip
Seeds that have not finished are listed as waiting; the verdict needs all three seeds.
Run: .venv/bin/python scripts/r_screen.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import boot_ci  # noqa: E402

DAYS = ("runs/2026-10-29", "runs/2026-11-07s2")
SEEDS = ("", "_s1", "_s2")


def load(run, kind):
    recs = [json.load(open(f)) for d in DAYS for f in sorted(glob.glob(f"{d}/{run}_rl/eval/eval_{kind}10_*__{run}_rl.json"))]
    return {r["set"]: r for r in recs}


def pooled(x, y, kind):
    ok_a, ok_b, c_a, c_b, seeds = [], [], [], [], []
    for s in SEEDS:
        a, b = load(f"{x}{s}", kind), load(f"{y}{s}", kind)
        sets = sorted(set(a) & set(b))
        if len(sets) < 2:
            continue
        seeds.append(s or "_s0")
        ok_a += [np.asarray(a[t]["ok"], bool) for t in sets]; ok_b += [np.asarray(b[t]["ok"], bool) for t in sets]
        c_a += [np.asarray(a[t]["count"], float) for t in sets]; c_b += [np.asarray(b[t]["count"], float) for t in sets]
    if not seeds:
        return None
    ok_a, ok_b, c_a, c_b = map(np.concatenate, (ok_a, ok_b, c_a, c_b))
    d = ok_a.astype(float) - ok_b
    both = ok_a & ok_b
    return dict(seeds=seeds, n=len(d), a=ok_a.mean(), b=ok_b.mean(), diff=d.mean(), ci=boot_ci(lambda i: d[i].mean(), len(d)),
                ratio=(c_a[both] / np.maximum(c_b[both], 1)).mean() if both.any() else float("nan"), n_both=int(both.sum()))


def fmt(p):
    return (f"{p['a']:.2f} vs {p['b']:.2f}, diff {p['diff']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}], count ratio "
            f"{p['ratio']:.3f} (both ok {p['n_both']}, seeds {','.join(p['seeds'])})")


def main():
    held = pooled("lo_pkg_rnd", "lo_pkg_r2pk", "pkg")
    same = {k: pooled("lo_pkg_rnd", "lo_pkg_r2pk", k) for k in ("pcb", "ip")}
    print("held-out pkg  rnd vs r2pk : " + (fmt(held) if held else "waiting"))
    for k, p in same.items():
        print(f"held-in {k:3s}   rnd vs r2pk : " + (fmt(p) if p else "waiting"))
    for lab, x, y in (("report: held-out pkg rnd vs r0 (coords)", "lo_pkg_rnd", "lo_pkg_r0"),
                      ("report: 2x2 coords+rnd vs coords", "lo_pkg_rnd_r0", "lo_pkg_r0"),
                      ("report: 2x2 mott+rnd vs coords+rnd", "lo_pkg_rnd", "lo_pkg_rnd_r0")):
        p = pooled(x, y, "pkg")
        print(f"{lab}: " + (fmt(p) if p else "waiting"))
    if not held or len(held["seeds"]) < 3 or any(p is None or len(p["seeds"]) < 3 for p in same.values()):
        print("\nverdict: waiting (needs seeds 0,1,2 for held-out pkg and held-in pcb/ip)")
        return
    c1 = held["diff"] >= 0.10 - 1e-9
    c2 = all(same[k]["diff"] >= -0.06 - 1e-9 for k in ("pcb", "ip"))
    c3 = all(p["ratio"] <= 1.05 for p in (held, *same.values()))
    print(f"\n(i) {c1} (ii) {c2} (iii) {c3} -> {'PASS' if c1 and c2 and c3 else 'fail'}")


if __name__ == "__main__":
    main()
