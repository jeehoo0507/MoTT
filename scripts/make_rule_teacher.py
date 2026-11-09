"""Diagnosis + candidate teacher (11/03 round 27, QUEUE 11/03 11:40 KST observation 3: MoTT solutions look like the
non-learned modal rule's, and the rule is 1.00 on pcb eval).

For every train_*.pkl: run `modal_heuristic` (the `modal_rule` baseline) once on the stored working grid, exact-check
its solution the same way scripts/make_teacher.py checks LS (verify.certify, worst <= 1% over Zt), and compare with the
stored teacher (data/teacher: LS where certified and not worse, else greedy+prune). Writes
  data/teacher_rbest/<pool>.npz   occ / count / ok / gp_count (the train.py teacher keys) = per instance the certified
                                  solution with fewer decaps among {stored teacher, rule} (ties: stored teacher);
                                  plus rule_occ, rule_count, rule_ok, rule_worst, src (0 stored, 1 rule)
  data/teacher_rbest/<pool>.json  per-pool summary
gp_count is unchanged, so the RL count normalisation (n_ref) is the same as with data/teacher.
One process, single-threaded torch (CPU slot rule, QUEUE 10/28 14:20).
Run: .venv/bin/python scripts/make_rule_teacher.py [--pool runs/pool_v1] [--src data/teacher] [--out data/teacher_rbest]
     --cp 4 --out data/teacher_rcp: the rule inside cutting-plane (4 rounds, as in eval) before the exact check.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="runs/pool_v1")
    ap.add_argument("--src", default="data/teacher")
    ap.add_argument("--out", default="data/teacher_rbest")
    ap.add_argument("--glob", default="train_*.pkl")
    ap.add_argument("--cp", type=int, default=0, help="cutting-plane rounds for the rule (0: stored grid only; "
                    "4 = scripts/eval_policy.py --rounds). Rounds refine an in-memory copy of the grid only.")
    a = ap.parse_args()
    torch.set_num_threads(1)
    from decap_loop.bank import load_bank
    from decap_loop.baselines import modal_heuristic
    from decap_loop.verify import certify, cutting_plane
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.pool, a.glob)))
    agg = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(a.out, name + ".npz")
        t = time.time()
        b, _ = load_bank(f)
        d = dict(np.load(os.path.join(a.src, name + ".npz")))
        if os.path.exists(dst):
            e = dict(np.load(dst))
            r_occ, r_ok, r_cnt, r_w = e["rule_occ"], e["rule_ok"], e["rule_count"], e["rule_worst"]
        else:
            if a.cp:
                r, _ = cutting_plane(lambda bb: modal_heuristic(bb, bb.cfg), b, b.cfg, max_rounds=a.cp)
            else:
                r = modal_heuristic(b, b.cfg)
            r_occ = r["occ"].cpu().numpy()
            r_w, _, _ = certify(b, torch.arange(b.B), r["occ"].cpu(), b.cfg)
            r_ok = np.asarray(r_w) <= np.log10(1.01)
            r_cnt = (r_occ >= 0).sum(1)
            n = d["occ"].shape[1]
            if r_occ.shape[1] != n:   # same [B, n_ports] layout as the stored teacher
                raise SystemExit(f"{name}: rule occ shape {r_occ.shape} vs teacher {d['occ'].shape}")
        t_ok, t_cnt = d["ok"].astype(bool), d["count"]
        use_r = r_ok & (~t_ok | (r_cnt < t_cnt))
        occ = np.where(use_r[:, None], r_occ, d["occ"])
        cnt = np.where(use_r, r_cnt, t_cnt)
        ok = t_ok | r_ok
        np.savez(dst, occ=occ, count=cnt, ok=ok, gp_count=d["gp_count"], rule_occ=r_occ, rule_count=r_cnt,
                 rule_ok=r_ok, rule_worst=np.asarray(r_w), src=use_r.astype(np.int8))
        both = t_ok & r_ok
        meta = dict(name=name, B=int(b.B), teacher_ok=float(t_ok.mean()), rule_ok=float(r_ok.mean()),
                    rule_over_teacher=float(np.mean(r_cnt[both] / np.maximum(t_cnt[both], 1))) if both.any() else None,
                    rule_fewer=float(np.mean(r_cnt[both] < t_cnt[both])) if both.any() else None,
                    use_rule=float(use_r.mean()), new_ok=float(ok.mean()),
                    teacher_count=float(t_cnt[t_ok].mean()) if t_ok.any() else None,
                    new_count=float(cnt[ok].mean()) if ok.any() else None, t_total=time.time() - t)
        json.dump(meta, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
        tech = name.split("_")[1].rstrip("0123456789")
        g = agg.setdefault(tech, dict(t_ok=[], r_ok=[], use=[], ratio=[], fewer=[], new_ok=[]))
        g["t_ok"] += list(t_ok)
        g["r_ok"] += list(r_ok)
        g["use"] += list(use_r)
        g["new_ok"] += list(ok)
        g["ratio"] += list(r_cnt[both] / np.maximum(t_cnt[both], 1))
        g["fewer"] += list(r_cnt[both] < t_cnt[both])
        print(f"{name}: teacher ok {meta['teacher_ok']:.2f} rule ok {meta['rule_ok']:.2f} rule/teacher "
              f"{meta['rule_over_teacher']} use rule {meta['use_rule']:.2f} ({meta['t_total']:.0f} s)", flush=True)
    summ = {t: dict(n=len(g["t_ok"]), teacher_ok=float(np.mean(g["t_ok"])), rule_ok=float(np.mean(g["r_ok"])),
                    new_ok=float(np.mean(g["new_ok"])), use_rule=float(np.mean(g["use"])),
                    rule_over_teacher_mean=float(np.mean(g["ratio"])) if g["ratio"] else None,
                    rule_over_teacher_median=float(np.median(g["ratio"])) if g["ratio"] else None,
                    rule_fewer=float(np.mean(g["fewer"])) if g["fewer"] else None) for t, g in agg.items()}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for t, s in summ.items():
        print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
