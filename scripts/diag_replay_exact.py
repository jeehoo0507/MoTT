"""Diagnosis D6 (11/03 round 35): do the distill targets pass the exact check?

train.py --stage distill imitates trajectory_to(working grid, stored teacher solution): add the teacher's decaps in
greedy order until the pool's *base working grid* is satisfied, then greedy-prune.  The working grid never grows in
distill (only RL adds exact-check failure frequencies, train.py "without this the policy learns to prune into
resonance peaks that fall between grid points").  So every rung imitates a base-grid-edge subset of the teacher
solution, which may fail the exact check even though the stored teacher passed it.  D3-D5 found r2's distill pkg
solutions grid-feasible but exact-failing far more often than r1's at similar counts (base grid: 0.18 / 0.20 / 0.10 vs
0.34 / 0.34 / 0.38).

Measurement (no policy): for every training pool file (pkg: sizes 8-12, all 9 files; pcb / ip: size 10 only, report),
replay = trajectory_to(base grid, teacher occ) exactly as in distill (max_steps 64); for boards whose replay is
grid-feasible, exact pass of the replay final (safe_verify.certify_parallel <= log10 1.01, 1 worker) and of the stored
teacher occ; counts of both.

Rule (written before running, round 35): if the pkg replay-final exact pass rate is <= 0.80 (>= 20 % of the pkg distill
targets fail the exact check) -> candidate C-cp "distill-stage cutting plane": before distill, grow each training pool's
working grid with the exact-check failure frequencies of its replayed teacher (as eval / RL do, same cap as RL), so
the distill targets are exact-passing; distill-only comparison per seed (distill is deterministic per seed), with r0
for fairness (it is a learning-method change).  Written in PROPOSALS before training.  If > 0.80, distill targets are
not the issue and C-cp is dropped.
Run: .venv/bin/python scripts/diag_replay_exact.py -> runs/analysis/diag_replay_exact.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TOL1 = np.log10(1.01)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    from decap_loop.bank import load_bank
    from decap_loop.baselines import trajectory_to
    from decap_loop.safe_verify import certify_parallel
    res = {}
    for tech, pat in (("pkg", "train_pkg*.pkl"), ("pcb", "train_pcb10_*.pkl"), ("ip", "train_ip10_*.pkl")):
        acc = dict(rep_ok=[], tch_ok=[], rep_cnt=[], tch_cnt=[], files={})
        for p in sorted(glob.glob(os.path.join(ROOT, "runs/pool_v1", pat))):
            b, _ = load_bank(p)
            name = os.path.basename(p)[:-4]
            d = np.load(os.path.join(ROOT, "data/teacher", name + ".npz"))
            occ = torch.from_numpy(d["occ"]).long()
            ib = torch.arange(b.B)
            teach = trajectory_to(b, b.cfg, ib, occ, max_steps=64)
            tb = torch.nonzero(teach["feasible"] & torch.from_numpy(d["ok"]).bool()).squeeze(-1)
            if not tb.numel():
                continue
            wr, _, _ = certify_parallel(b, ib[tb], teach["occ"][tb], b.cfg, workers=1)
            wt, _, _ = certify_parallel(b, ib[tb], occ[tb], b.cfg, workers=1)
            r_ok, t_ok = (wr <= TOL1).tolist(), (wt <= TOL1).tolist()
            acc["rep_ok"] += r_ok; acc["tch_ok"] += t_ok
            acc["rep_cnt"] += teach["count"][tb].tolist(); acc["tch_cnt"] += (occ[tb] >= 0).sum(1).tolist()
            acc["files"][name] = dict(n=int(tb.numel()), rep_ok=float(np.mean(r_ok)), F=int(b.freqs.shape[1]))
            print(tech, name, acc["files"][name], flush=True)
        r = {k: round(float(np.mean(acc[k])), 4) for k in ("rep_ok", "tch_ok", "rep_cnt", "tch_cnt")}
        r.update(n=len(acc["rep_ok"]), files=acc["files"])
        res[tech] = r
        print(tech, {k: v for k, v in r.items() if k != "files"}, flush=True)
    res["C_cp"] = res["pkg"]["rep_ok"] <= 0.80
    print("C_cp (pkg replay exact pass <= 0.80):", res["C_cp"])
    json.dump(res, open(os.path.join(ROOT, "runs/analysis/diag_replay_exact.json"), "w"), indent=1)
    print("wrote runs/analysis/diag_replay_exact.json")


if __name__ == "__main__":
    main()
