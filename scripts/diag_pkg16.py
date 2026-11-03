"""Diagnosis (11/03 round 27, QUEUE 11/03 11:40 KST observation 4: on 16-cell pkg every learned rung is below g+p).
From existing eval records only (no new computation): per board kind and size, for a policy vs greedy+prune on the
same problems, the failure type of the policy's failures -
  near miss   exact worst log10(|Z|/Zt) in (log10 1.01, log10 1.15]
  far miss    above log10 1.15
and the policy's count / g+p count on the failed problems (does the policy stop with fewer decaps than g+p needs?),
plus the episode-length cap: share of failures at the 64-step cap (max_steps; a policy that runs out of steps).
Run: .venv/bin/python scripts/diag_pkg16.py [policy run dir=runs/2026-10-28/r2_mott_rl] [method=r2_mott_rl]
Writes runs/analysis/diag_size_<method>.json and prints a table."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "runs/2026-10-28/baselines/eval"
TOL, NEAR = np.log10(1.01), np.log10(1.15)


def load(path):
    r = json.load(open(os.path.join(ROOT, path)))
    return np.array(r["ok"], bool), np.array(r["count"]), np.array(r["worst"])


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-10-28/r2_mott_rl"
    meth = sys.argv[2] if len(sys.argv) > 2 else "r2_mott_rl"
    out = {}
    for tech in ("pcb", "pkg", "ip"):
        for size in (10, 16):
            ok, cnt, w, gok, gcnt = [], [], [], [], []
            for k in (0, 1):
                s = f"eval_{tech}{size}_{k}"
                p = f"{run}/eval/{s}__{meth}.json"
                g = f"{BASE}/{s}__greedy+prune.json"
                if not (os.path.exists(os.path.join(ROOT, p)) and os.path.exists(os.path.join(ROOT, g))):
                    continue
                a, b, c = load(p)
                d, e, _ = load(g)
                ok += list(a); cnt += list(b); w += list(c); gok += list(d); gcnt += list(e)
            if not ok:
                continue
            ok, cnt, w, gok, gcnt = map(np.array, (ok, cnt, w, gok, gcnt))
            f = ~ok
            row = dict(n=int(ok.size), ok=float(ok.mean()), gp_ok=float(gok.mean()),
                       fail=int(f.sum()), near=int((f & (w <= NEAR)).sum()), far=int((f & (w > NEAR)).sum()),
                       worst_fail_median=float(np.median(w[f])) if f.any() else None,
                       at_cap=int((f & (cnt >= 64)).sum()),
                       fail_gp_ok=int((f & gok).sum()),
                       fail_count_over_gp=float(np.median(cnt[f & gok] / np.maximum(gcnt[f & gok], 1))) if (f & gok).any() else None,
                       ok_count_over_gp=float(np.median(cnt[ok & gok] / np.maximum(gcnt[ok & gok], 1))) if (ok & gok).any() else None,
                       gp_count_median=float(np.median(gcnt[gok])) if gok.any() else None)
            out[f"{tech}{size}"] = row
    os.makedirs(os.path.join(ROOT, "runs/analysis"), exist_ok=True)
    dst = os.path.join(ROOT, f"runs/analysis/diag_size_{meth}.json")
    json.dump(out, open(dst, "w"), indent=1)
    cols = list(next(iter(out.values())).keys())
    print("| board | " + " | ".join(cols) + " |")
    for k, v in out.items():
        print(f"| {k} | " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v.values()) + " |")
    print("wrote", dst)


if __name__ == "__main__":
    main()
