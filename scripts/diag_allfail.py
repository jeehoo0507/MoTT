"""Diagnosis (11/03 round 27, QUEUE 11/03 11:40 KST observations 1-2): how much of the RL signal comes from groups in
which all M samples of an instance fail, and what that signal rewards.

train.py RL reward R = -count/n_ref - lam * (fail + V), advantage = (R - mean_M) / std_M per instance. V is the mean
positive part of log10(|Z|/Zt) over the working grid (sim.py), so for a near miss V ~ 0 and, when every sample of an
instance fails, the normalised advantage is driven by -count alone: the update pushes a failing instance toward FEWER
decaps, i.e. away from feasibility. This script measures, on the training pools (working grid as stored, no exact
check -> grid failures only, a lower bound on the failures train.py sees), for a checkpoint:
  - share of instances whose M samples all fail (by kind and by teacher g+p count tercile = hardness)
  - in those groups: corr(count, V) within the group (negative = fewer decaps go with more violation) and the share of
    the advantage variance explained by the count term (|corr(adv, -count)|^2)
Run: .venv/bin/python scripts/diag_allfail.py <ckpt> <config.json of that run> [M] [lam]
Writes runs/analysis/diag_allfail_<run>.json.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.modal_env import make_env  # noqa: E402
from decap_loop.pdn import PDNConfig  # noqa: E402
from decap_loop.rollout import rollout  # noqa: E402
from train import load_teacher, make_policy  # noqa: E402


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    ckpt, cfg_path = sys.argv[1], sys.argv[2]
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    a = json.load(open(cfg_path))["args"]
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else 1.4   # r2_mott_rl ended at lam ~1.4
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    policy = make_policy(PDNConfig(), a).to(dev)
    policy.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    policy.eval()
    if hasattr(policy, "set_eval_R"):
        policy.set_eval_R(a.get("loop_eval", 8))
    files = sorted({f for g in a["pool_glob"].split(",") for f in glob.glob(os.path.join(ROOT, a["pool"], g))})
    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        pb, _ = load_bank(f, dev, dense=True, z_dtype=torch.complex64)
        tch = load_teacher(os.path.join(ROOT, a["teacher_dir"], name + ".npz"), pb, dev, lambda *x: None)
        n_ref = tch["gp"].clamp(min=1).float()
        for c0 in range(0, pb.B, 4):
            ib = torch.arange(c0, min(c0 + 4, pb.B), device=dev)
            env = make_env(pb, ib.repeat_interleave(M), pb.cfg, a["feat"], max_steps=a["max_steps"],
                           allow_prune=not a.get("no_prune", False))
            with torch.no_grad():
                out = rollout(policy, env, amp=not a.get("no_amp", False))
            nb = ib.numel()
            cnt = out["count"].float().view(nb, M).cpu().numpy()
            fail = (~out["feasible"]).view(nb, M).cpu().numpy()
            V = out["V"].float().view(nb, M).cpu().numpy()
            for k in range(nb):
                b = int(ib[k])
                R = -cnt[k] / float(n_ref[b]) - lam * (fail[k] + V[k])
                adv = (R - R.mean()) / (R.std() + 1e-6)
                rows.append(dict(pool=name, tech=pb.cfg.name if hasattr(pb.cfg, "name") else name.split("_")[1][:-2],
                                 inst=b, gp=int(tch["gp"][b]), teach_ok=bool(tch["ok"][b]), n_fail=int(fail[k].sum()),
                                 cnt_mean=float(cnt[k].mean()), V_mean=float(V[k].mean()),
                                 c_cntV=corr(cnt[k], V[k]) if fail[k].all() else None,
                                 r2_cnt=corr(adv, -cnt[k]) ** 2 if fail[k].all() else None))
        print(f"{name}: all-fail {np.mean([r['n_fail'] == M for r in rows if r['pool'] == name]):.2f}", flush=True)
        del pb
        torch.cuda.empty_cache()
    techs = sorted({r["pool"].split("_")[1].rstrip("0123456789") for r in rows})
    summ = {}
    for t in techs + ["all"]:
        rs = [r for r in rows if t == "all" or r["pool"].split("_")[1].rstrip("0123456789") == t]
        g = np.array([r["gp"] for r in rs])
        q = np.quantile(g, [1 / 3, 2 / 3])
        for lab, sel in (("all", rs), ("gp low", [r for r in rs if r["gp"] <= q[0]]),
                         ("gp mid", [r for r in rs if q[0] < r["gp"] <= q[1]]), ("gp high", [r for r in rs if r["gp"] > q[1]])):
            af = [r for r in sel if r["n_fail"] == M]
            cs = [r["c_cntV"] for r in af if r["c_cntV"] is not None and not np.isnan(r["c_cntV"])]
            r2 = [r["r2_cnt"] for r in af if r["r2_cnt"] is not None and not np.isnan(r["r2_cnt"])]
            summ[f"{t} | {lab}"] = dict(n=len(sel), all_fail=len(af) / max(1, len(sel)),
                                        any_fail=float(np.mean([r["n_fail"] > 0 for r in sel])) if sel else None,
                                        sample_fail=float(np.mean([r["n_fail"] / M for r in sel])) if sel else None,
                                        allfail_teacher_ok=float(np.mean([r["teach_ok"] for r in af])) if af else None,
                                        corr_cnt_V=float(np.median(cs)) if cs else None,
                                        r2_adv_cnt=float(np.median(r2)) if r2 else None,
                                        V_allfail=float(np.median([r["V_mean"] for r in af])) if af else None)
    run = os.path.basename(os.path.dirname(ckpt))
    out = os.path.join(ROOT, f"runs/analysis/diag_allfail_{run}_{os.path.splitext(os.path.basename(ckpt))[0]}.json")
    json.dump(dict(ckpt=ckpt, M=M, lam=lam, summary=summ, rows=rows), open(out, "w"), indent=1)
    for k, v in summ.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
