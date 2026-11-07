"""PROPOSALS 11/07 C1 screening (rule written by the human 11/07 14:15 KST before any C1 run; this code written 11/07
round 76 before any C1 record exists): seed 0, 10-cell eval files, policy alone, paired on the same problems.
  runs       : server 2 (human, scripts/server2_c1.sh) -> runs/2026-11-07s2/{r2pk_vsi,r2pk_ctl}_rl/eval/
  pair       : r2pk_vsi_rl vs r2pk_ctl_rl on eval_{pcb,pkg,ip}10_{0,1} (50 problems per kind)
  pass       : (i) pkg diff >= +0.10, (ii) pcb and ip diff >= -0.06, (iii) count ratio (mean per-problem ratio over
               problems both solve, as scripts/pair2.py) <= 1.05 for each of pcb / pkg / ip (same reading as G1)
  pass ->    : confirmation seed 1/2 (4 runs, server placement per the human); 3-seed pooled pkg CI lower bound > 0
Point estimates decide; the paired bootstrap 95% CI is printed alongside, not used. Training iterations of both runs
(metrics.json) are printed too (C1 does fewer iterations in the same 1 h; reported, not used).
Run: .venv/bin/python scripts/c1_screen.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_screen  # noqa: E402

DAY = "runs/2026-11-07s2"
CAND, CTL = "r2pk_vsi", "r2pk_ctl"


def iters(run):
    f = f"{DAY}/{run}_rl/metrics.json"
    if not os.path.exists(f):
        return "-"
    m = json.load(open(f))
    return m.get("iters", m.get("iter", m.get("n_iter", "?")))


def main():
    g1_screen.DAY = DAY   # g1_screen.load reads {DAY}/{run}_rl/eval/eval_{kind}10_*__{run}_rl.json
    same = {k: g1_screen.pair(CAND, CTL, k) for k in ("pcb", "pkg", "ip")}
    for k, p in same.items():
        print(f"{CAND} - {CTL} {k}: " + (g1_screen.fmt(p) if p else "waiting"))
    print(f"train iters: {CAND} {iters(CAND)}, {CTL} {iters(CTL)}")
    if any(p is None for p in same.values()):
        print("C1 screening: waiting for records")
        return
    c1 = same["pkg"]["diff"] >= 0.10 - 1e-9
    c2 = all(same[k]["diff"] >= -0.06 - 1e-9 for k in ("pcb", "ip"))
    c3 = all(same[k]["ratio"] <= 1.05 for k in ("pcb", "pkg", "ip"))
    print(f"C1: (i) {c1} (ii) {c2} (iii) {c3} -> {'PASS' if c1 and c2 and c3 else 'fail'}")


if __name__ == "__main__":
    main()
