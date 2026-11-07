"""PROPOSALS 11/07 G1 screening (rule written by the human before any run; this code written 11/07 round 65 before any
G1 result): seed 0, 10-cell eval files, policy alone, paired on the same problems.
  same kind  : X_rl        vs r2pk_rl         on eval_{pcb,pkg,ip}10_{0,1} (50 problems per kind)
  held-out   : lo_pkg_X_rl vs lo_pkg_r2pk_rl  on eval_pkg10_{0,1}
  pass       : (i) same-kind pkg diff >= +0.10 or held-out pkg diff >= +0.10, (ii) same-kind pcb and ip diff >= -0.06,
               (iii) same-kind count ratio (mean per-problem ratio over problems both solve, as scripts/pair2.py)
               <= 1.05 for each of pcb / pkg / ip (reading fixed before results: per kind, the strict one)
  pick       : among passing X the larger held-out pkg diff; none -> G1 ends, r2pk stays.
Point estimates decide; the paired bootstrap 95% CI (collect_results.boot_ci) is printed alongside, not used.
Run: .venv/bin/python scripts/g1_screen.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import boot_ci  # noqa: E402

DAY = "runs/2026-10-29"
CANDS = ("r2pkfc",)   # G1-L r3pk withdrawn by the human before any G1 run (PROPOSALS G1 수정, 11/07 10:50 KST)


def load(run, kind):
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{DAY}/{run}_rl/eval/eval_{kind}10_*__{run}_rl.json"))]
    return {r["set"]: r for r in recs}


def pair(a_run, b_run, kind):
    a, b = load(a_run, kind), load(b_run, kind)
    sets = sorted(set(a) & set(b))
    if len(sets) < 2:
        return None
    ok_a = np.concatenate([np.asarray(a[s]["ok"], bool) for s in sets])
    ok_b = np.concatenate([np.asarray(b[s]["ok"], bool) for s in sets])
    c_a = np.concatenate([np.asarray(a[s]["count"], float) for s in sets])
    c_b = np.concatenate([np.asarray(b[s]["count"], float) for s in sets])
    d = ok_a.astype(float) - ok_b
    both = ok_a & ok_b
    r = c_a[both] / np.maximum(c_b[both], 1)
    return dict(n=len(d), a=ok_a.mean(), b=ok_b.mean(), diff=d.mean(), ci=boot_ci(lambda i: d[i].mean(), len(d)),
                ratio=r.mean() if both.any() else float("nan"), n_both=int(both.sum()))


def fmt(p):
    return (f"{p['a']:.2f} vs {p['b']:.2f}, diff {p['diff']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}], "
            f"count ratio {p['ratio']:.3f} (both ok {p['n_both']})")


def main():
    passed, complete = [], True
    for x in CANDS:
        same = {k: pair(x, "r2pk", k) for k in ("pcb", "pkg", "ip")}
        held = pair(f"lo_pkg_{x}", "lo_pkg_r2pk", "pkg")
        for k, p in same.items():
            print(f"{x} same {k}: " + (fmt(p) if p else "waiting"))
        print(f"lo_pkg_{x} held-out pkg: " + (fmt(held) if held else "waiting"))
        if any(p is None for p in same.values()) or held is None:
            print(f"{x}: screening waiting for records\n")
            complete = False
            continue
        c1 = same["pkg"]["diff"] >= 0.10 - 1e-9 or held["diff"] >= 0.10 - 1e-9
        c2 = all(same[k]["diff"] >= -0.06 - 1e-9 for k in ("pcb", "ip"))
        c3 = all(same[k]["ratio"] <= 1.05 for k in ("pcb", "pkg", "ip"))
        ok = c1 and c2 and c3
        print(f"{x}: (i) {c1} (ii) {c2} (iii) {c3} -> {'PASS' if ok else 'fail'}\n")
        if ok:
            passed.append((held["diff"], x))
    print("pick:", (max(passed)[1] if passed else "none (G1 ends, r2pk stays)") if complete else "waiting")


if __name__ == "__main__":
    main()
