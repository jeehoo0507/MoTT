"""QUEUE 2026-11-01 23:45 (pkg reversal, H4): how close is each board type to the other two in input distribution?
Prints, per type, the bare-board feature quantiles of the 10×10 eval problems (runs/analysis/problem_features.json),
the technology parameters and decap library of its pool, and a 1-D Wasserstein distance between types per feature.
Prints only (analysis aid; RESULTS numbers come from collect_results.py).
Run: .venv/bin/python scripts/type_distance.py"""
import json
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TECHS = ("ip", "pcb", "pkg")
KEYS = ("bare_worst", "log_f_worst", "viol_frac", "n_placeable")


def w1(a, b):
    q = np.linspace(0, 1, 101)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def main():
    F = json.load(open(os.path.join(ROOT, "runs/analysis/problem_features.json")))
    X = {}
    for t in TECHS:
        rows = [r for s in F if s.startswith(f"eval_{t}10") for r in F[s]]
        X[t] = {k: np.array([r[k] for r in rows], float) for k in KEYS}
        print(t, "  ".join(f"{k} med {np.median(v):.2f} p10-p90 {np.percentile(v, 10):.2f}..{np.percentile(v, 90):.2f}"
                           for k, v in X[t].items()))
        d = pickle.load(open(os.path.join(ROOT, f"runs/pool_v1/eval_{t}10_0.pkl"), "rb"))
        c = d["cfg"]
        print("   cfg", [(k, c[k]) for k in ("cell_size", "diel_thickness", "eps_r", "tan_d", "vrm_l", "f_min", "f_max")])
        print("   decaps (C, ESL, ESR)", c["decaps"])
    for k in KEYS:
        sd = np.std(np.concatenate([X[t][k] for t in TECHS]))
        print(f"W1/sd {k:12s}", "  ".join(f"{a}-{b} {w1(X[a][k], X[b][k]) / sd:.2f}"
                                          for i, a in enumerate(TECHS) for b in TECHS[i + 1:]))


if __name__ == "__main__":
    main()
