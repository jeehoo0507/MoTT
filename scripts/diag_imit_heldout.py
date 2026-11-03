"""Diagnosis (11/03 round 35, follow-up of D1 / diag_imit): is MoTT's distill-stage pkg weakness a generalisation gap?

diag_imit showed r2 fits the pkg *training* teacher at least as well as r1 (nll seed 0/1: r2 2.65 / 2.03, r1 2.81 / 2.67)
while r2's distill-only eval pkg success is lower in all 3 seeds (D1).  Not measured yet: how well each distill
checkpoint imitates the same kind of teacher on boards it never saw.

Held-out teacher (same rule as data/teacher: local search where it passes the exact check, else greedy+prune):
on the size-10 eval pools (runs/pool_v1/eval_<tech>10_{0,1}, 50 boards per tech), teacher solution = the stored
`local_search` eval solution if ok, else the `greedy+prune` one if ok, else the board is skipped
(runs/2026-10-28/baselines/eval).  Replay + scoring is diag_imit.score with that occupancy (same trajectory_to,
max_steps 64, only boards whose teacher is feasible on the base working grid).  Same 9 checkpoints as diag_imit.

Hypothesis H-gen and rule (written before running, round 35):
  gap = held-out pkg nll - training pkg nll (training value from runs/analysis/diag_imit.json).
  H-gen holds if in all 3 seeds r2's held-out pkg nll is above both r0's and r1's AND r2's gap is above both r0's and
  r1's gaps.  Then: "MoTT memorises the pkg training teacher; the D1 gap is generalisation" -> next candidate is a
  single regularisation change for MoTT (written in PROPOSALS before training, seeds 0 and 1, same-kind + lo_pcb).
  If not: the D1 gap is not an imitation-generalisation gap either (with D2/diag_imit: not fit, not prior, not update
  count) -> regularisation is dropped as a candidate; the gap is in how the policy's own rollouts compound / stop.
  pcb and ip are reported only.
Run: .venv/bin/python scripts/diag_imit_heldout.py  -> runs/analysis/diag_imit_heldout.json.  CPU, 1 thread.
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

BASE = os.path.join(ROOT, "runs/2026-10-28/baselines/eval")


def heldout_occ(set_name, n):
    """[n, cells] teacher occupancy (local search if ok, else greedy+prune if ok) and a keep mask."""
    ls = json.load(open(os.path.join(BASE, f"{set_name}__local_search.json")))
    gp = json.load(open(os.path.join(BASE, f"{set_name}__greedy+prune.json")))
    occ, keep, src = [], [], []
    for i in range(n):
        if ls["ok"][i]:
            occ.append(ls["occ"][i]); keep.append(True); src.append("ls")
        elif gp["ok"][i]:
            occ.append(gp["occ"][i]); keep.append(True); src.append("gp")
        else:
            occ.append(gp["occ"][i]); keep.append(False); src.append("-")
    return torch.tensor(np.array(occ)).long(), torch.tensor(keep), src


@torch.no_grad()
def score(pol, feat, path):
    from decap_loop.bank import load_bank
    from decap_loop.baselines import trajectory_to
    from decap_loop.modal_env import make_env
    from decap_loop.rollout import _logits, rollout
    b, _ = load_bank(path)
    cfg = b.cfg
    occ, keep, _ = heldout_occ(os.path.basename(path)[:-4], b.B)
    ib = torch.arange(b.B)
    teach = trajectory_to(b, cfg, ib, occ, max_steps=64)
    tb = torch.nonzero(teach["feasible"] & keep).squeeze(-1)
    env = make_env(b, ib[tb], cfg, feat, max_steps=64, allow_prune=True)
    rep = rollout(pol, env, forced=teach["actions"][tb], store=True, amp=False)
    T1 = cfg.n_types + 1
    out = dict(nll=[], top1=[], cell=[], tcell=[])
    for s in rep["traj"]:
        lg = _logits(pol, s["x"], s["valid"], b.H, b.W, s["mask"], False)
        lp = torch.log_softmax(lg, -1)
        a = s["a"]
        out["nll"] += (-lp.gather(1, a[:, None]).squeeze(1)).tolist()
        out["top1"] += (lg.argmax(-1) == a).tolist()
        node = a < env.stop_idx
        n = env.n
        cl = lp[:, :n * T1].view(-1, n, T1)
        tc, tch = a // T1, a % T1
        out["cell"] += (torch.logsumexp(cl, -1).argmax(-1) == tc)[node].tolist()
        at = cl[torch.arange(a.shape[0]), tc.clamp(max=n - 1)]
        out["tcell"] += (at.argmax(-1) == tch)[node].tolist()
    return {k: (float(np.sum(v)), len(v)) for k, v in out.items()}, int(tb.numel())


def main():
    torch.set_num_threads(1)
    from train import make_policy
    from decap_loop.bank import load_bank
    train = json.load(open(os.path.join(ROOT, "runs/analysis/diag_imit.json")))
    res = {}
    for name, ck_path in CKPTS.items():
        ck = torch.load(os.path.join(ROOT, ck_path), map_location="cpu", weights_only=False)
        a = ck["args"]
        res[name] = {}
        for tech in ("pkg", "pcb", "ip"):
            acc, nb = {}, 0
            for p in sorted(glob.glob(os.path.join(ROOT, f"runs/pool_v1/eval_{tech}10_*.pkl"))):
                cfg = load_bank(p)[0].cfg
                pol = make_policy(cfg, a)
                pol.load_state_dict(ck["model"])
                pol.eval()
                sc, k = score(pol, a["feat"], p)
                nb += k
                for key, (sm, n) in sc.items():
                    s0, n0 = acc.get(key, (0.0, 0))
                    acc[key] = (s0 + sm, n0 + n)
            r = {k: round(sm / max(n, 1), 4) for k, (sm, n) in acc.items()}
            r.update(n_steps=acc["nll"][1], n_boards=nb, train_nll=train[name][tech]["nll"])
            r["gap"] = round(r["nll"] - r["train_nll"], 4)
            res[name][tech] = r
            print(name, tech, r, flush=True)
    sfx = ["", "_s1", "_s2"]
    h = []
    for s in sfx:
        r0, r1, r2 = (res[f"{m}{s}"]["pkg"] for m in ("r0_coord", "r1_modal", "r2_mott"))
        h.append(r2["nll"] > max(r0["nll"], r1["nll"]) and r2["gap"] > max(r0["gap"], r1["gap"]))
    res["H_gen_per_seed"] = h
    res["H_gen"] = all(h)
    print("H_gen per seed:", h, "->", res["H_gen"])
    json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_imit_heldout.json"), "w"), indent=1)
    print("wrote runs/analysis/diag_imit_heldout.json")


if __name__ == "__main__":
    main()
