"""Exploration helper (prints only, nothing for RESULTS): per held-out type, per-problem outcome of the
leave-out MoTT runs (seeds 0-2) vs modal_rule and E0+, against the bare-board features of
runs/analysis/problem_features.json. Used to choose the grouping that collect_results.py tables.
Run: .venv/bin/python scripts/explore_learn_vs_rule.py [rung]   (rung r2 default)"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def recs():
    by = defaultdict(dict)
    for f in glob.glob(os.path.join(ROOT, "runs/**/eval/*.json"), recursive=True):
        d = json.load(open(f))
        if int(d["size"]) == 10:
            by[d["method"]][d["set"]] = d
    return by


def main():
    rung = sys.argv[1] if len(sys.argv) > 1 else "r2"
    by = recs()
    feat = json.load(open(os.path.join(ROOT, "runs/analysis/problem_features.json")))
    for t in ("ip", "pcb", "pkg"):
        sets = sorted(s for s in feat if s.startswith(f"eval_{t}10_"))
        F = [r for s in sets for r in feat[s]]
        def get(m, k):
            return np.concatenate([np.asarray(by[m][s][k], float) for s in sets]) if m in by and all(s in by[m] for s in sets) else None
        rule_ok, rule_c = get("modal_rule", "ok"), get("modal_rule", "count")
        ta_ok, ta_c = get("gp_type_aware", "ok"), get("gp_type_aware", "count")
        gp_c = get("greedy+prune", "count")
        print(f"=== {t}: rule ok {rule_ok.mean():.2f}, E0+ ok {ta_ok.mean():.2f}")
        names = [f"lo_{t}_{rung}_rl", f"lo_{t}_{rung}_s1_rl", f"lo_{t}_{rung}_s2_rl"]
        L = [(get(n, "ok"), get(n, "count")) for n in names]
        L = [x for x in L if x[0] is not None]
        print("seeds", len(L), [f"{o.mean():.2f}" for o, _ in L])
        # per-problem: win = learn solves & rule fails, or both solve and fewer decaps; loss symmetric
        score = np.zeros(len(F))
        for o, c in L:
            win = (o.astype(bool) & ~rule_ok.astype(bool)) | (o.astype(bool) & rule_ok.astype(bool) & (c < rule_c))
            loss = (~o.astype(bool) & rule_ok.astype(bool)) | (o.astype(bool) & rule_ok.astype(bool) & (c > rule_c))
            score += win.astype(float) - loss.astype(float)
        score /= max(len(L), 1)
        keys = ["n_valid", "n_keepout", "n_placeable", "bare_worst", "log_f_worst", "n_viol_peaks", "viol_frac"]
        X = {k: np.array([r[k] for r in F], float) for k in keys}
        X["ta_count"] = ta_c
        X["rule_count"] = rule_c
        X["gp_count"] = gp_c
        for k, v in X.items():
            good = np.isfinite(v)
            rho = np.corrcoef(np.argsort(np.argsort(v[good])), np.argsort(np.argsort(score[good])))[0, 1]
            print(f"  {k:13s} spearman(score) {rho:+.2f}   range {np.nanmin(v):.2f}..{np.nanmax(v):.2f}")
        for cat in ("shape", "target"):
            for val in sorted(set(r[cat] for r in F)):
                m = np.array([r[cat] == val for r in F])
                print(f"  {cat}={val:6s} n={m.sum():2d} mean score {score[m].mean():+.2f}  rule ok {rule_ok[m].mean():.2f}  "
                      f"learn ok {np.mean([o[m].mean() for o, _ in L]):.2f}")
        print("  score dist", np.round(np.percentile(score, [10, 25, 50, 75, 90]), 2), "mean", round(score.mean(), 3))


if __name__ == "__main__":
    main()
