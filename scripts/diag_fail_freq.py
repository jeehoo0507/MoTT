"""Diagnosis D7 (11/03 round 36, PROPOSALS D7; rule written there before running).

Where is the exact-check worst frequency of the grid-feasible but exact-failing solutions of the r1 / r2 distill
policies (base working grid, one greedy rollout with allow_prune = the D4 "prune" path)?
For each such solution, with fw = certified worst frequency of the final placement:
  d_tok    : min |log10 fw - log10 f_k| over the 8 mode-token frequencies of the final state (the K highest local
             maxima of max_probe log10 |Z_pp| / Zt on the sorted working grid, same rule as sim.features_mott)
  s        : log10 spacing of the two working-grid points around fw; covered = d_tok <= s
  underres : certified worst - working-grid level at the grid point nearest to fw (log10)
Rule (pkg): H-uncov if in all 3 seeds r2's uncovered fraction >= 0.5 and above r1's; H-underres if in all 3 seeds r2's
covered fraction >= 0.5 and median underres >= 0.02; a seed with < 5 failures counts as not holding. pcb reported only.
Run: .venv/bin/python scripts/diag_fail_freq.py -> runs/analysis/diag_fail_freq.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from diag_imit import CKPTS  # noqa: E402

TOL1 = np.log10(1.01)
K_TOK = 8
# --refine (P1 pre-queue check, PROPOSALS 11/03 P1): r2 pkg only; per failure also the peak_env refined excess of the
# nearest token; writes runs/analysis/diag_fail_freq_refine.json
REFINE = "--refine" in sys.argv
# --refine_u (PROPOSALS 11/04 P3 "같이 보고", no rule): --refine plus the same excess with the parabola through the
# nearest *distinct*-frequency neighbours (decap_loop/pkdd_env.py); writes runs/analysis/diag_fail_freq_refine_u.json
REFINE_U = "--refine_u" in sys.argv
REFINE |= REFINE_U
# --rank (D8, PROPOSALS 11/04 D8): r1/r2 pkg; per uncovered failure the level rank of the nearest working-grid local
# maximum within s of fw (A = rank > K_TOK, "token budget"; B = none within s, "invisible on grid");
# writes runs/analysis/diag_fail_rank.json
RANK = "--rank" in sys.argv
# --ckpt NAME=PATH (P1/P2 "같이 보고", PROPOSALS 11/03 P1 / 11/04 P2): D7 + D8b summaries of one more distill checkpoint
# on pkg (report only, no rule); writes runs/analysis/diag_fail_rank_<NAME>.json
EXTRA = dict(a.split("=", 1) for a in sys.argv[sys.argv.index("--ckpt") + 1:]) if "--ckpt" in sys.argv else {}
if EXTRA:
    RANK = True


def tokens_and_grid(b, ib, occ, allpk=None, dedup=False):
    """Sorted working-grid log10 f [E,F], level [E,F] and the K_TOK token log10 f [E,<=K].
    allpk (list) gets every episode's local-maximum indices sorted by level (descending).
    dedup: tokens as decap_loop/dedup_env.py picks them (feat mott_dd: peaks of the distinct-frequency curve)."""
    from decap_loop.sim import probe_impedance
    b.bank.ensure(ib, torch.where(occ >= 0, torch.arange(b.n, device=occ.device), -1))
    z = probe_impedance(b, ib, occ, b.zd()[ib])
    lv = (torch.log10(z.abs() + 1e-30).amax(2) - torch.log10(b.Zt[ib])).cpu().numpy()     # [E,F]
    lf = np.log10(torch.as_tensor(b.freqs)[ib].double().cpu().numpy())
    o = np.argsort(lf, 1)
    lf, lv = np.take_along_axis(lf, o, 1), np.take_along_axis(lv, o, 1)
    pad = np.full((lv.shape[0], 1), -np.inf)
    peak = (lv >= np.concatenate([pad, lv[:, :-1]], 1)) & (lv >= np.concatenate([lv[:, 1:], pad], 1))
    toks, tj = [], []
    for e in range(lv.shape[0]):
        j = np.flatnonzero(peak[e])
        if allpk is not None:
            allpk.append(j[np.argsort(-lv[e, j])])
        if dedup:
            _, iu = np.unique(np.round(lf[e], 9), return_index=True)
            vu = lv[e, iu]
            ju = np.flatnonzero((vu >= np.r_[-np.inf, vu[:-1]]) & (vu >= np.r_[vu[1:], -np.inf]))
            j = iu[ju]
        tj.append(j[np.argsort(-lv[e, j])[:K_TOK]])
        toks.append(lf[e, tj[-1]])
    return lf, lv, toks, tj


def refined_excess(lf, lv, j):
    """peak_env.refine at sorted grid index j of one episode."""
    from decap_loop.peak_env import refine
    exc, _ = refine(torch.from_numpy(lv[None]), torch.from_numpy(lf[None]), torch.tensor([[int(j)]]))
    return float(exc)


def refined_excess_u(lf, lv, j):
    """pkdd_env.refine_at at the run of sorted grid index j, neighbours = nearest distinct frequencies."""
    from decap_loop.pkdd_env import distinct_neighbours, refine_at
    jf = int(np.flatnonzero(lf == lf[int(j)])[0])                 # first member of j's equal-frequency run
    lvt, lft, jt = torch.from_numpy(lv[None]), torch.from_numpy(lf[None]), torch.tensor([[jf]])
    jl, jr = distinct_neighbours(lft, jt)
    exc, _ = refine_at(lvt, lft, jl, jt, jr)
    return float(exc)


@torch.no_grad()
def run(pol, feat, b):
    from decap_loop.modal_env import make_env
    from decap_loop.rollout import rollout
    from decap_loop.safe_verify import certify_parallel
    ib = torch.arange(b.B)
    env = make_env(b, ib, b.cfg, feat, allow_prune=True)
    out = rollout(pol, env, greedy=True, amp=False)
    occ = out["occ"]
    wv, fw, _ = certify_parallel(b, ib, occ, b.cfg, workers=1)
    feas = out["feasible"].numpy()
    allpk = [] if RANK else None
    lf, lv, toks, tj = tokens_and_grid(b, ib, occ, allpk, dedup=feat == "mott_dd")
    rows = []
    for e in np.flatnonzero(feas & (wv > TOL1)):
        x = np.log10(fw[e])
        g = lf[e]
        i = int(np.clip(np.searchsorted(g, x), 1, len(g) - 1))
        s = g[i] - g[i - 1]
        near = i if abs(g[i] - x) < abs(x - g[i - 1]) else i - 1
        d_tok = float(np.min(np.abs(toks[e] - x))) if len(toks[e]) else float("inf")
        rows.append(dict(worst=float(wv[e]), lfw=float(x), s=float(s), d_tok=d_tok, covered=bool(d_tok <= s),
                         underres=float(wv[e] - lv[e, near]), count=int(out["count"][e])))
        if REFINE and len(toks[e]):
            rows[-1]["refined_excess"] = refined_excess(lf[e], lv[e], tj[e][int(np.argmin(np.abs(toks[e] - x)))])
            if REFINE_U:
                rows[-1]["refined_excess_u"] = refined_excess_u(lf[e], lv[e], tj[e][int(np.argmin(np.abs(toks[e] - x)))])
        if RANK:
            pk = allpk[e]
            dist = np.abs(g[pk] - x)
            within = np.flatnonzero(dist <= s)
            rows[-1]["n_max"] = int(len(pk))
            rows[-1]["rank"] = int(within[np.argmin(dist[within])]) + 1 if len(within) else -1   # 1-based, -1 = none
            # D8b: same on the de-duplicated curve (the padded working grid repeats frequencies, pdn.add_freqs)
            gu, iu = np.unique(np.round(g, 9), return_index=True)
            vu = lv[e][iu]
            pad = np.array([-np.inf])
            pku = np.flatnonzero((vu >= np.r_[pad, vu[:-1]]) & (vu >= np.r_[vu[1:], pad]))
            pku = pku[np.argsort(-vu[pku])]
            du = np.abs(gu[pku] - x)
            wu = np.flatnonzero(du <= s)
            rows[-1]["n_max_u"] = int(len(pku))
            rows[-1]["rank_u"] = int(wu[np.argmin(du[wu])]) + 1 if len(wu) else -1
            rows[-1]["n_tok_distinct"] = int(len(np.unique(np.round(toks[e], 9))))
    return rows, int(feas.sum()), int(((wv <= TOL1) & feas).sum()), b.B


def rank_summ(rows):
    unc = [r for r in rows if not r["covered"]]
    a = sum(r["rank"] > K_TOK for r in unc)
    bb = sum(r["rank"] < 0 for r in unc)
    au = sum(1 <= r["rank_u"] <= K_TOK for r in unc)
    return dict(n_fail=len(rows), n_uncov=len(unc),
                A_budget=round(a / len(unc), 4) if unc else None, B_invisible=round(bb / len(unc), 4) if unc else None,
                other=len(unc) - a - bb,
                D_dup=round(au / len(unc), 4) if unc else None,       # D8b: a top-8 distinct peak, pushed out by copies
                Bu_invisible=round(sum(r["rank_u"] < 0 for r in unc) / len(unc), 4) if unc else None,
                n_tok_distinct_med=float(np.median([r["n_tok_distinct"] for r in rows])) if rows else None,
                n_max_u_med=float(np.median([r["n_max_u"] for r in rows])) if rows else None,
                n_max_med=float(np.median([r["n_max"] for r in rows])) if rows else None,
                frac_gt_K=round(float(np.mean([r["n_max"] > K_TOK for r in rows])), 4) if rows else None)


def summ(rows):
    if not rows:
        return dict(n_fail=0)
    cov = np.array([r["covered"] for r in rows])
    return dict(n_fail=len(rows), covered=round(float(cov.mean()), 4), uncovered=round(float(1 - cov.mean()), 4),
                d_tok_med=round(float(np.median([r["d_tok"] for r in rows])), 4),
                s_med=round(float(np.median([r["s"] for r in rows])), 4),
                underres_med=round(float(np.median([r["underres"] for r in rows])), 4),
                worst_med=round(float(np.median([r["worst"] for r in rows])), 4))


def main():
    torch.set_num_threads(1)
    from train import make_policy
    from decap_loop.bank import load_bank
    if EXTRA:
        CKPTS.clear()
        CKPTS.update(EXTRA)
    names = [n for n in CKPTS if n.startswith("r2" if REFINE else ("r1", "r2"))]
    res = {n: {} for n in names}
    for tech in (("pkg",) if REFINE or RANK else ("pkg", "pcb")):
        for name in names:
            ck = torch.load(os.path.join(ROOT, CKPTS[name]), map_location="cpu", weights_only=False)
            a = ck["args"]
            rows, nf, nok, nb = [], 0, 0, 0
            for p in sorted(glob.glob(os.path.join(ROOT, f"runs/pool_v1/eval_{tech}10_*.pkl"))):
                b, _ = load_bank(p)
                pol = make_policy(b.cfg, a)
                pol.load_state_dict(ck["model"])
                pol.eval()
                r, f, o, n = run(pol, a["feat"], b)
                rows += r; nf += f; nok += o; nb += n
            s = summ(rows)
            s.update(grid_feasible=round(nf / nb, 4), exact_pass=round(nok / nb, 4), n=nb)
            if RANK:
                s.update(rank_summ(rows))
            res[name][tech] = dict(summary=s, rows=rows)
            print(name, tech, s, flush=True)
    if REFINE:
        ex = [r["refined_excess"] for n in names for r in res[n]["pkg"]["rows"] if r["covered"] and "refined_excess" in r]
        res["P1_check"] = dict(n_covered=len(ex), frac_gt_002=round(float(np.mean(np.array(ex) > 0.02)), 4),
                               median=round(float(np.median(ex)), 4))
        res["P1_queue"] = res["P1_check"]["frac_gt_002"] >= 0.20
        print("P1 check", res["P1_check"], "-> queue" if res["P1_queue"] else "-> cancel")
        if REFINE_U:
            cov = [r for n in names for r in res[n]["pkg"]["rows"] if r["covered"] and "refined_excess_u" in r]
            eu = np.array([r["refined_excess_u"] for r in cov])
            e0 = np.array([r["refined_excess"] for r in cov])
            res["P3_report"] = dict(n_covered=len(cov), frac_gt_002_raw=round(float(np.mean(e0 > 0.02)), 4),
                                    frac_gt_002_distinct=round(float(np.mean(eu > 0.02)), 4),
                                    median_raw=round(float(np.median(e0)), 4),
                                    median_distinct=round(float(np.median(eu)), 4),
                                    underres_med=round(float(np.median([r["underres"] for r in cov])), 4))
            print("P3 report", res["P3_report"])
            json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_fail_freq_refine_u.json"), "w"), indent=1)
            return
        json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_fail_freq_refine.json"), "w"), indent=1)
        return
    if EXTRA:
        for name in names:
            out = os.path.join(ROOT, f"runs/analysis/diag_fail_rank_{name}.json")
            json.dump({name: res[name], "ckpt": CKPTS[name]}, open(out, "w"), indent=1)
            print("wrote", out)
        return
    if RANK:
        hk = []
        for sfx in ("", "_s1", "_s2"):
            s2 = res[f"r2_mott{sfx}"]["pkg"]["summary"]
            hk.append(bool(s2["n_uncov"] >= 5 and s2["A_budget"] >= 0.5))
        res["H_K_per_seed"], res["H_K"] = hk, all(hk)
        print("H_K", hk, all(hk))
        hd = []
        for sfx in ("", "_s1", "_s2"):
            s2 = res[f"r2_mott{sfx}"]["pkg"]["summary"]
            hd.append(bool(s2["n_uncov"] >= 5 and s2["D_dup"] >= 0.5))
        res["H_dup_per_seed"], res["H_dup"] = hd, all(hd)
        print("H_dup (D8b)", hd, all(hd))
        json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_fail_rank.json"), "w"), indent=1)
        print("wrote runs/analysis/diag_fail_rank.json")
        return
    hu, hr = [], []
    for sfx in ("", "_s1", "_s2"):
        s2, s1 = res[f"r2_mott{sfx}"]["pkg"]["summary"], res[f"r1_modal{sfx}"]["pkg"]["summary"]
        ok = s2["n_fail"] >= 5
        hu.append(bool(ok and s2["uncovered"] >= 0.5 and s2["uncovered"] > s1.get("uncovered", 0.0)))
        hr.append(bool(ok and s2["covered"] >= 0.5 and s2["underres_med"] >= 0.02))
    res["H_uncov_per_seed"], res["H_uncov"] = hu, all(hu)
    res["H_underres_per_seed"], res["H_underres"] = hr, all(hr)
    print("H_uncov", hu, all(hu), "H_underres", hr, all(hr))
    json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_fail_freq.json"), "w"), indent=1)
    print("wrote runs/analysis/diag_fail_freq.json")


if __name__ == "__main__":
    main()
