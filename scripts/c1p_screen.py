"""PROPOSALS 11/07 C1' G-inf screening (rule written by the human 11/07 14:30 KST before any C1' run; this code written
11/07 round 77 while the G-inf eval was running, before its ip records existed): seed 0, 10-cell eval files, paired.
  G-inf      : r2pk_rl seed 0 checkpoint, no training, eval with --env_feat mott_pkgate
               -> runs/2026-10-29/r2pk_rl/eval/eval_{kind}10_*__r2pk_rl@mott_pkgate.json
  pair       : r2pk_rl (same checkpoint, plain env) on eval_{pcb,pkg,ip}10_{0,1} (50 problems per kind)
  pass       : (i) pkg diff >= +0.10, (ii) pcb and ip diff >= -0.06, (iii) count ratio (mean per-problem ratio over
               problems both solve, as scripts/pair2.py) <= 1.05 for each of pcb / pkg / ip (same reading as G1 / C1)
  pass ->    : recorded as "inference gate alone"; reported with G-tr if G-tr runs (G-tr is not queued: G1 passed,
               QUEUE 14:30 KST item 3 -> human instruction).
Point estimates decide; the paired bootstrap 95% CI is printed alongside, not used. The gated "policy alone" number
includes the refined-peak gate (not the exact check / repair).
Run: .venv/bin/python scripts/c1p_screen.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_screen  # noqa: E402

EVAL = "runs/2026-10-29/r2pk_rl/eval"
CAND, CTL = "r2pk_rl@mott_pkgate", "r2pk_rl"


def load(tag, kind):
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{EVAL}/eval_{kind}10_*__{tag}.json"))]
    return {r["set"]: r for r in recs}


def main():
    g1_screen.load = lambda run, kind: load(run, kind)
    same = {k: g1_screen.pair(CAND, CTL, k) for k in ("pcb", "pkg", "ip")}
    for k, p in same.items():
        print(f"{CAND} - {CTL} {k}: " + (g1_screen.fmt(p) if p else "waiting"))
    if any(p is None for p in same.values()):
        print("C1' G-inf screening: waiting for records")
        return
    c1 = same["pkg"]["diff"] >= 0.10 - 1e-9
    c2 = all(same[k]["diff"] >= -0.06 - 1e-9 for k in ("pcb", "ip"))
    c3 = all(same[k]["ratio"] <= 1.05 for k in ("pcb", "pkg", "ip"))
    print(f"C1' G-inf: (i) {c1} (ii) {c2} (iii) {c3} -> {'PASS' if c1 and c2 and c3 else 'fail'}")


if __name__ == "__main__":
    main()
