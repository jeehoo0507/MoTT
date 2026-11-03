"""Diagnosis (11/03 round 34, follow-up of PROPOSALS 11/03 D1): how well does each distill-only checkpoint fit the
teacher it imitated, per tech?  Replays the stored teacher solution (data/teacher/<pool>.npz, the same trajectory_to
replay train.py --stage distill uses) on the size-10 training pools and scores the checkpoint's logits at every
teacher step:
  nll      mean -log pi(teacher action)
  top1     argmax action == teacher action
  cell     argmax over cells of logsumexp over channels == teacher cell
  type|cell  argmax channel at the teacher cell == teacher channel (type choice given the right cell)
Checkpoints: r0_coord / r1_modal / r2_mott distill best.pt, seeds 0-2.

Use (written before running; applies only if D1 holds, i.e. r2_distill pkg eval success < r0 and r1 in all 3 seeds):
  if r2's pkg train nll is above both r0's and r1's in all 3 seeds -> "MoTT underfits the pkg teacher in the same
  distill budget" (fit), and the next candidate targets distill fit; if not -> the D1 gap is not a fit gap
  (generalisation / selection), and the candidate must not be a fit fix. type|cell vs cell says which part.

MoTT only (r2), same replay, hypothesis H-prior (written before running, round 34): MoTT's logits = learned residual
+ physics prior w_dv*dV - w_mn*mn, and w_dv / w_mn stay at their init 20 / 3 in every checkpoint (Adam step ~lr on a
scalar of 20), so the prior scale is effectively fixed; on pkg the prior dominates and the policy follows one-step
greedy instead of the teacher.  Extra metrics: prior_top1 (prior-only argmax == teacher), agree_prior (full argmax ==
prior-only argmax), prior_sd / resid_sd (std over legal actions, mean over steps).
Rule: H-prior holds if, in all 3 seeds, pkg has agree_prior above pcb's and ip's AND pkg prior_sd / resid_sd above
pcb's and ip's AND pkg prior_top1 < pkg top1 (following the prior costs teacher agreement).  Then the next candidate
is "prior scale learnable in log space" (one architecture change), written up in PROPOSALS before training.
Otherwise the candidate is dropped.
Run: .venv/bin/python scripts/diag_imit.py  -> runs/analysis/diag_imit.json.  CPU, 1 thread.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CKPTS = {f"{r}{'' if s == 0 else f'_s{s}'}": f"runs/2026-10-{28 if s == 0 else 29}/{r}{'' if s == 0 else f'_s{s}'}_distill/best.pt"
         for s in (0, 1, 2) for r in ("r0_coord", "r1_modal", "r2_mott")}


@torch.no_grad()
def score(pol, feat, path):
    from decap_loop.bank import load_bank
    from decap_loop.baselines import trajectory_to
    from decap_loop.modal_env import make_env
    from decap_loop.rollout import _logits, rollout
    b, _ = load_bank(path)
    cfg = b.cfg
    d = np.load(os.path.join(ROOT, "data/teacher", os.path.basename(path)[:-4] + ".npz"))
    ib = torch.arange(b.B)
    teach = trajectory_to(b, cfg, ib, torch.from_numpy(d["occ"]).long(), max_steps=64)
    tb = torch.nonzero(teach["feasible"]).squeeze(-1)
    env = make_env(b, ib[tb], cfg, feat, max_steps=64, allow_prune=True)
    rep = rollout(pol, env, forced=teach["actions"][tb], store=True, amp=False)
    T1 = cfg.n_types + 1
    out = dict(nll=[], top1=[], cell=[], tcell=[])
    mott = hasattr(pol, "w_dv")
    if mott:
        out.update(prior_top1=[], agree_prior=[], prior_sd=[], resid_sd=[])
    for s in rep["traj"]:
        lg = _logits(pol, s["x"], s["valid"], b.H, b.W, s["mask"], False)
        lp = torch.log_softmax(lg, -1)
        a = s["a"]
        out["nll"] += (-lp.gather(1, a[:, None]).squeeze(1)).tolist()
        out["top1"] += (lg.argmax(-1) == a).tolist()
        if mott:
            T = cfg.n_types
            node = s["x"][..., :pol.dn]
            nb = 4 + T + 3
            pr = pol.w_dv * node[..., nb:nb + T + 1] - pol.w_mn * node[..., nb + T + 1:nb + 2 * (T + 1)]
            pr = torch.cat([pr.reshape(pr.shape[0], -1), torch.zeros_like(pr[:, :1, 0])], 1).float()
            m = s["mask"]
            prm = pr.masked_fill(~m, float("-inf"))
            out["prior_top1"] += (prm.argmax(-1) == a).tolist()
            out["agree_prior"] += (prm.argmax(-1) == lg.argmax(-1)).tolist()
            res = lg - pr
            for v, key in ((pr, "prior_sd"), (res, "resid_sd")):
                vv = torch.where(m, v, torch.zeros_like(v))
                k = m.sum(1).clamp(min=2).float()
                mu = vv.sum(1) / k
                sd = (((vv - mu[:, None]) ** 2) * m).sum(1) / (k - 1)
                out[key] += sd.sqrt().tolist()
        node = a < env.stop_idx
        n = env.n
        cl = lp[:, :n * T1].view(-1, n, T1)
        cell_pred = torch.logsumexp(cl, -1).argmax(-1)
        tc, tch = a // T1, a % T1
        out["cell"] += (cell_pred == tc)[node].tolist()
        at = cl[torch.arange(a.shape[0]), tc.clamp(max=n - 1)]          # [k, T+1] logits at the teacher cell
        out["tcell"] += (at.argmax(-1) == tch)[node].tolist()
    return {k: (float(np.sum(v)), len(v)) for k, v in out.items()}


def main():
    torch.set_num_threads(1)
    from train import make_policy
    from decap_loop.bank import load_bank
    res = {}
    for name, ck_path in CKPTS.items():
        ck = torch.load(os.path.join(ROOT, ck_path), map_location="cpu", weights_only=False)
        a = ck["args"]
        res[name] = {}
        for tech in ("pcb", "pkg", "ip"):
            acc = {}
            for p in sorted(glob.glob(os.path.join(ROOT, f"runs/pool_v1/train_{tech}10_*.pkl"))):
                cfg = load_bank(p)[0].cfg
                pol = make_policy(cfg, a)
                pol.load_state_dict(ck["model"])
                pol.eval()
                for k, (sm, n) in score(pol, a["feat"], p).items():
                    s0, n0 = acc.get(k, (0.0, 0))
                    acc[k] = (s0 + sm, n0 + n)
            res[name][tech] = {k: round(sm / max(n, 1), 4) for k, (sm, n) in acc.items()}
            res[name][tech]["n_steps"] = acc["nll"][1]
            print(name, tech, res[name][tech], flush=True)
    rows = [res[f"r2_mott{'' if s == 0 else f'_s{s}'}"] for s in (0, 1, 2)]
    res["H_prior"] = all(r["pkg"]["agree_prior"] > max(r["pcb"]["agree_prior"], r["ip"]["agree_prior"])
                         and r["pkg"]["prior_sd"] / r["pkg"]["resid_sd"]
                         > max(r[t]["prior_sd"] / r[t]["resid_sd"] for t in ("pcb", "ip"))
                         and r["pkg"]["prior_top1"] < r["pkg"]["top1"] for r in rows)
    print("H_prior:", res["H_prior"])
    os.makedirs(os.path.join(ROOT, "runs/analysis"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_imit.json"), "w"), indent=1)
    print("wrote runs/analysis/diag_imit.json")


if __name__ == "__main__":
    main()
