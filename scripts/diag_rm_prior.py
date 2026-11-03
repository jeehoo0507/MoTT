"""Diagnosis D5 (11/03 round 35, after D4): does MoTT's built-in physics prior drive the post-feasibility pruning?

D4 (diag_prune_feasible, rule not met): on the eval_pkg10 base grid r2's distill policy removes ~1.2 decaps after first
grid feasibility (drop 1.14 / 1.18, seeds 0 / 1) and loses exact passes (cost 0.06 / 0.16), while r1 hardly prunes
(drop 0.00 / 0.56).  MoTT logits = residual + w_dv*dV - w_mn*mn on every channel including remove; the stop logit has no
prior.  At a grid-feasible state dV = 0 for every legal remove (V stays 0) and mn <= 0, so the prior adds
-w_mn[T]*mn >= 0 to removals only: a push toward pruning to the grid edge that r0 / r1 (no prior) do not have.

H-rmprior (written before running, round 35): same measurement as D4 prune variant (one greedy rollout, base grid,
eval_pkg10_{0,1}, 50 boards, exact check), r2_mott distill best.pt seeds 0-2, with the remove-channel prior weights
w_dv[T], w_mn[T] set to 0 at inference only (inference surgery, a diagnosis, never a reported method).
Rule: holds if in all 3 seeds drop (count(noprune, from D4) - count(surgery)) is at most half of D4's drop AND exact
pass rises by >= 0.04 over D4's prune variant.  Then the next candidate is "MoTT prior on the add / switch channels
only" (architecture change, trained, written in PROPOSALS before training).  Otherwise the prior is not the pruning
driver and that candidate is dropped.
Run: .venv/bin/python scripts/diag_rm_prior.py -> runs/analysis/diag_rm_prior.json (needs diag_prune_feasible.json)
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
from diag_prune_feasible import run  # noqa: E402


def main():
    torch.set_num_threads(1)
    from train import make_policy
    from decap_loop.bank import load_bank
    jp = os.path.join(ROOT, "runs/analysis/diag_prune_feasible.json")
    if os.path.exists(jp):
        d4 = json.load(open(jp))
    else:                                    # D4 still on pcb: its pkg lines are already in the log
        import ast
        d4 = {}
        for line in open(os.path.join(ROOT, "runs/analysis/diag_prune_feasible.log")):
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[1] == "pkg" and parts[2].startswith("{"):
                d4.setdefault(parts[0], {})["pkg"] = ast.literal_eval(parts[2])
    res, h = {}, []
    for name in ("r2_mott", "r2_mott_s1", "r2_mott_s2"):
        ck = torch.load(os.path.join(ROOT, CKPTS[name]), map_location="cpu", weights_only=False)
        a = ck["args"]
        ok, cnt = [], []
        for p in sorted(glob.glob(os.path.join(ROOT, "runs/pool_v1/eval_pkg10_*.pkl"))):
            b, _ = load_bank(p)
            pol = make_policy(b.cfg, a)
            pol.load_state_dict(ck["model"])
            pol.eval()
            T = b.cfg.n_types
            with torch.no_grad():
                pol.w_dv[T] = 0.0
                pol.w_mn[T] = 0.0
            _, o, c = run(pol, a["feat"], b, True)
            ok += o.tolist(); cnt += c.tolist()
        ref = d4[name]["pkg"]
        r = dict(ok_s=round(float(np.mean(ok)), 4), cnt_s=round(float(np.mean(cnt)), 4), ok_p=ref["ok_p"],
                 cnt_p=ref["cnt_p"], cnt_n=ref["cnt_n"], drop_d4=ref["drop"])
        r["drop_s"] = round(ref["cnt_n"] - r["cnt_s"], 4)
        r["d_ok"] = round(r["ok_s"] - ref["ok_p"], 4)
        res[name] = r
        h.append(r["drop_s"] <= 0.5 * ref["drop"] and r["d_ok"] >= 0.04)
        print(name, r, flush=True)
    res["H_rmprior_per_seed"] = h
    res["H_rmprior"] = all(h)
    print("H_rmprior per seed:", h, "->", res["H_rmprior"])
    json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_rm_prior.json"), "w"), indent=1)
    print("wrote runs/analysis/diag_rm_prior.json")


if __name__ == "__main__":
    main()
