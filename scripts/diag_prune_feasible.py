"""Diagnosis (11/03 round 35, follow-up of D1): does MoTT's distill policy prune to the edge of the working grid?

D1 failure shape (round 34): r2's distill-only pkg failures place few decaps (11-12, vs r0 22-28 / r1 23-40 when they
fail) and satisfy the working grid but fail the exact check.  The env lets a policy remove / switch decaps once the
working grid is satisfied (allow_prune); the teacher trajectories end with a greedy prune too.

H-prune (written before running, round 35): after first reaching grid feasibility, the r2 distill policy keeps
removing decaps down to the grid boundary, and that removal is what fails the exact check on pkg.
Measurement (no training, no cutting-plane rounds: one greedy rollout on the base working grid of eval_pkg10_{0,1},
50 boards; CPU, 1 thread, certify with 1 worker): for each of the 9 distill best.pt (r0 / r1 / r2 x seeds 0-2)
  prune    : rollout with allow_prune (what every eval does)
  noprune  : same policy, allow_prune off -> the episode ends at the first grid-feasible state; up to that state the
             legal set and hence the greedy actions are identical, so this is the prune run's state at first
             feasibility
  per variant: grid-feasible rate, exact pass rate (safe_verify.certify_parallel <= log10 1.01, as in train.validate),
  mean count; cost = exact(noprune) - exact(prune) over the 50 boards; drop = mean count(noprune) - count(prune).
Rule: H-prune holds if in all 3 seeds r2's cost is > 0.04 (more than 2 boards) and above both r0's and r1's cost.
  Then the D1 gap is (at least partly) post-feasibility pruning, and the next candidate targets the remove decision
  (written in PROPOSALS before training).  Otherwise pruning is not the D1 mechanism and is dropped as a target.
  pcb reported only.
Run: .venv/bin/python scripts/diag_prune_feasible.py -> runs/analysis/diag_prune_feasible.json
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


@torch.no_grad()
def run(pol, feat, b, allow_prune):
    from decap_loop.modal_env import make_env
    from decap_loop.rollout import rollout
    from decap_loop.safe_verify import certify_parallel
    ib = torch.arange(b.B)
    env = make_env(b, ib, b.cfg, feat, allow_prune=allow_prune)
    out = rollout(pol, env, greedy=True, amp=False)
    wv, _, _ = certify_parallel(b, ib, out["occ"], b.cfg, workers=1)
    return out["feasible"].numpy(), (wv <= TOL1) & out["feasible"].numpy(), out["count"].numpy()


def main():
    torch.set_num_threads(1)
    from train import make_policy
    from decap_loop.bank import load_bank
    res = {}
    for name, ck_path in CKPTS.items():
        ck = torch.load(os.path.join(ROOT, ck_path), map_location="cpu", weights_only=False)
        a = ck["args"]
        res[name] = {}
        for tech in ("pkg", "pcb"):
            acc = {k: [] for k in ("feas_p", "ok_p", "cnt_p", "feas_n", "ok_n", "cnt_n")}
            for p in sorted(glob.glob(os.path.join(ROOT, f"runs/pool_v1/eval_{tech}10_*.pkl"))):
                b, _ = load_bank(p)
                pol = make_policy(b.cfg, a)
                pol.load_state_dict(ck["model"])
                pol.eval()
                for sfx, ap in (("p", True), ("n", False)):
                    f, o, c = run(pol, a["feat"], b, ap)
                    acc[f"feas_{sfx}"] += f.tolist(); acc[f"ok_{sfx}"] += o.tolist(); acc[f"cnt_{sfx}"] += c.tolist()
            r = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
            r["cost"] = round(r["ok_n"] - r["ok_p"], 4)
            r["drop"] = round(r["cnt_n"] - r["cnt_p"], 4)
            r["frac_dropped"] = round(float(np.mean(np.array(acc["cnt_n"]) > np.array(acc["cnt_p"]))), 4)
            r["n"] = len(acc["ok_p"])
            res[name][tech] = r
            print(name, tech, r, flush=True)
    h = []
    for s in ("", "_s1", "_s2"):
        c0, c1, c2 = (res[f"{m}{s}"]["pkg"]["cost"] for m in ("r0_coord", "r1_modal", "r2_mott"))
        h.append(c2 > 0.04 and c2 > max(c0, c1))
    res["H_prune_per_seed"] = h
    res["H_prune"] = all(h)
    print("H_prune per seed:", h, "->", res["H_prune"])
    json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_prune_feasible.json"), "w"), indent=1)
    print("wrote runs/analysis/diag_prune_feasible.json")


if __name__ == "__main__":
    main()
