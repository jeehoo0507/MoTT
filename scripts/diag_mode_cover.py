"""Diagnosis (11/03 round 34): do MoTT's mode tokens miss violating frequency bands?

MoTT (sim.Env.features_mott) gives the policy K_MODES = 8 mode tokens = the 8 highest local maxima of the loaded
max-over-probes log10(|Z_pp|/Zt) on the sorted working grid. diag_difficulty found tens of violating local maxima
per bare pkg10 board (eval median 48), so the 8 highest may be ripples of one broad hump.

Hypothesis (written before running): on the bare board, the top-8 maxima often leave whole violating bands
(maximal runs of grid points with level > FEAS_TOL) without a token, and r2_mott's pcb/pkg failures relative to
r0_coord concentrate on boards with an uncovered band.
Decision rule (written before running): call a board "uncovered" if >= 1 violating band holds none of the 8 tokens.
If (a) >= 20 % of pcb10 or pkg10 eval boards are uncovered, and (b) in that tech the paired success difference
r2 - r0 (same seed, same problems) is lower on uncovered than on covered boards in all 3 seeds, the next MoTT
candidate is a token-selection change (one token per violating band before filling by height) - an input
representation change, written up in PROPOSALS before any training. Otherwise this candidate is dropped.
Bare board only (the first state of every episode); later states are not checked here.
Run: .venv/bin/python scripts/diag_mode_cover.py   -> runs/analysis/diag_mode_cover.json.  CPU, 1 thread, ~1 min.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
K = 8
RUNS = {0: ("runs/2026-10-28/r2_mott_rl", "r2_mott_rl", "runs/2026-10-28/r0_coord_rl", "r0_coord_rl"),
        1: ("runs/2026-10-29/r2_mott_s1_rl", "r2_mott_s1_rl", "runs/2026-10-29/r0_coord_s1_rl", "r0_coord_s1_rl"),
        2: ("runs/2026-10-29/r2_mott_s2_rl", "r2_mott_s2_rl", "runs/2026-10-29/r0_coord_s2_rl", "r0_coord_s2_rl")}


def cover(path):
    from decap_loop.bank import load_bank
    from decap_loop.sim import EPS, FEAS_TOL, probe_impedance
    b, _ = load_bank(path)
    inst = torch.arange(b.B)
    occ = torch.full((b.B, b.n), -1, dtype=torch.long)
    zpp = probe_impedance(b, inst, occ, b.zd("cpu"))
    lv = (torch.log10(zpp.abs() + EPS) - torch.log10(b.Zt)[..., None]).amax(-1).numpy()
    rows = []
    for i in range(b.B):
        o = np.argsort(b.freqs[i])
        f, v = b.freqs[i][o], lv[i][o]
        pad = np.concatenate([[-np.inf], v, [-np.inf]])
        peak = (v >= pad[:-2]) & (v >= pad[2:])            # same rule as features_mott
        cand = np.where(peak)[0]
        tok = cand[np.argsort(-v[cand])[:K]]
        viol = v > FEAS_TOL
        bands, j = [], 0
        while j < len(v):
            if viol[j]:
                k = j
                while k + 1 < len(v) and viol[k + 1]:
                    k += 1
                bands.append((j, k))
                j = k + 1
            else:
                j += 1
        unc = [(a, c) for a, c in bands if not ((tok >= a) & (tok <= c)).any()]
        rows.append(dict(n_bands=len(bands), n_uncov=len(unc),
                         uncov_worst=float(max(v[a:c + 1].max() for a, c in unc)) if unc else None,
                         tok_logf_span=float(np.ptp(np.log10(f[tok]))) if len(tok) else 0.0,
                         n_viol_peaks=int((peak & viol).sum())))
    return rows


def ok(run, meth, s):
    p = os.path.join(ROOT, run, "eval", f"{s}__{meth}.json")
    return np.array(json.load(open(p))["ok"], bool) if os.path.exists(p) else None


def main():
    torch.set_num_threads(1)
    out = {}
    for tech in ("pcb", "pkg", "ip"):
        rows, oks = [], {sd: ([], []) for sd in RUNS}
        for p in sorted(glob.glob(os.path.join(ROOT, f"runs/pool_v1/eval_{tech}10_*.pkl"))):
            s = os.path.basename(p)[:-4]
            rows += cover(p)
            for sd, (r2d, r2m, r0d, r0m) in RUNS.items():
                a, c = ok(r2d, r2m, s), ok(r0d, r0m, s)
                oks[sd][0].append(a)
                oks[sd][1].append(c)
        unc = np.array([r["n_uncov"] > 0 for r in rows])
        res = dict(n=len(rows), share_uncovered=round(float(unc.mean()), 3),
                   n_bands_q=[float(x) for x in np.quantile([r["n_bands"] for r in rows], [0.25, 0.5, 0.75, 1])],
                   n_uncov_q=[float(x) for x in np.quantile([r["n_uncov"] for r in rows], [0.25, 0.5, 0.75, 1])],
                   uncov_worst_q=[round(float(x), 3) for x in np.quantile(
                       [r["uncov_worst"] for r in rows if r["uncov_worst"] is not None] or [np.nan], [0.5, 0.9, 1])],
                   tok_logf_span_q=[round(float(x), 3) for x in np.quantile([r["tok_logf_span"] for r in rows],
                                                                             [0.25, 0.5, 0.75])])
        for sd in RUNS:
            if any(x is None for x in oks[sd][0] + oks[sd][1]):
                continue
            a, c = np.concatenate(oks[sd][0]), np.concatenate(oks[sd][1])
            d = a.astype(float) - c.astype(float)
            res[f"seed{sd}"] = dict(
                r2_unc=round(float(a[unc].mean()), 3) if unc.any() else None,
                r2_cov=round(float(a[~unc].mean()), 3) if (~unc).any() else None,
                r0_unc=round(float(c[unc].mean()), 3) if unc.any() else None,
                r0_cov=round(float(c[~unc].mean()), 3) if (~unc).any() else None,
                diff_unc=round(float(d[unc].mean()), 3) if unc.any() else None,
                diff_cov=round(float(d[~unc].mean()), 3) if (~unc).any() else None)
        sds = [res.get(f"seed{sd}") for sd in RUNS]
        res["rule_a"] = res["share_uncovered"] >= 0.2
        res["rule_b"] = all(x is not None and x["diff_unc"] is not None and x["diff_cov"] is not None
                            and x["diff_unc"] < x["diff_cov"] for x in sds)
        out[tech] = res
        print(tech, json.dumps(res), flush=True)
    out["candidate"] = any(out[t]["rule_a"] and out[t]["rule_b"] for t in ("pcb", "pkg"))
    print("candidate:", out["candidate"])
    os.makedirs(os.path.join(ROOT, "runs/analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "runs/analysis/diag_mode_cover.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
