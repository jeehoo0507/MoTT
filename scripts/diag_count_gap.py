"""Count gap under the same repair: which decap kinds each method uses (round 32 diagnosis).

Reads runs/repair10/eval/*+rep4.json (same boards, same repair for all methods). For problems where
every listed method passes the exact check, reports per tech: mean count, kind mix (fraction of
placed decaps of kind 0/1/2, kind 0 = largest C), and the paired count ratio vs r0_coord_rl+rep4
with a bootstrap 95% CI (ratio of means over resampled problems).

Usage: .venv/bin/python scripts/diag_count_gap.py  -> runs/analysis/diag_count_gap.json
"""
import glob
import json
import os

import numpy as np

METHODS = ["r2_mott_rl+rep4", "r0_coord_rl+rep4", "modal_rule+rep4", "greedy+prune+rep4"]
REF = "r0_coord_rl+rep4"


def load(tech):
    recs = {}
    for m in METHODS:
        ok, cnt, mix = [], [], []
        for f in sorted(glob.glob(f"runs/repair10/eval/eval_{tech}10_*__{m}.json")):
            r = json.load(open(f))
            ok += r["ok"]
            cnt += r["count"]
            for occ in r["occ"]:
                o = np.asarray(occ)
                mix.append([int((o == t).sum()) for t in range(3)])
        recs[m] = (np.array(ok), np.array(cnt, float), np.array(mix, float))
    return recs


def occs(tech, m):
    out = []
    for f in sorted(glob.glob(f"runs/repair10/eval/eval_{tech}10_*__{m}.json")):
        out += [np.asarray(o) for o in json.load(open(f))["occ"]]
    return out


def jaccard(a, b):
    sa = {(i, int(t)) for i, t in enumerate(a) if t >= 0}
    sb = {(i, int(t)) for i, t in enumerate(b) if t >= 0}
    return len(sa & sb) / max(1, len(sa | sb))


def boot_ratio(a, b, rng, B=2000):
    n = len(a)
    idx = rng.integers(0, n, (B, n))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    rng = np.random.default_rng(0)
    out = {}
    for tech in ["ip", "pcb", "pkg"]:
        recs = load(tech)
        both = np.all([recs[m][0] for m in METHODS], axis=0)
        row = {"n_all_ok": int(both.sum()), "n": int(len(both))}
        for m in METHODS:
            ok, cnt, mix = recs[m]
            tot = mix[both].sum(0)
            row[m] = {"mean_count": float(cnt[both].mean()),
                      "kind_mix": [round(float(x), 3) for x in tot / tot.sum()],
                      "mean_per_kind": [round(float(x), 2) for x in mix[both].mean(0)]}
            if m != REF:
                row[m]["ratio_vs_r0"] = [round(x, 3) for x in boot_ratio(cnt[both], recs[REF][1][both], rng)]
        # placement overlap with modal_rule: Jaccard of (port, kind) sets, and identical solutions
        rule_occ = occs(tech, "modal_rule+rep4")
        for m in ["r2_mott_rl+rep4", "r0_coord_rl+rep4"]:
            mo = occs(tech, m)
            jac = [jaccard(a, b) for a, b, k in zip(mo, rule_occ, both) if k]
            row[m]["jaccard_vs_rule"] = round(float(np.mean(jac)), 3)
            row[m]["identical_to_rule"] = int(sum(j == 1.0 for j in jac))
        out[tech] = row
        print(tech, f"all-ok {row['n_all_ok']}/{row['n']}")
        for m in METHODS:
            r = row[m]
            print(f"  {m:20s} count {r['mean_count']:6.2f}  per kind {r['mean_per_kind']}  mix {r['kind_mix']}"
                  + (f"  ratio vs r0 {r['ratio_vs_r0']}" if "ratio_vs_r0" in r else "")
                  + (f"  jaccard vs rule {r['jaccard_vs_rule']} identical {r['identical_to_rule']}"
                     if "jaccard_vs_rule" in r else ""))
    os.makedirs("runs/analysis", exist_ok=True)
    json.dump(out, open("runs/analysis/diag_count_gap.json", "w"), indent=1)


if __name__ == "__main__":
    main()
