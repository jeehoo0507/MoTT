"""Export the numbers behind the report's board figures (Fig. 1 one policy on three board kinds, Fig. 11 rule vs
MoTT+P1 on one board) so they can be drawn off-server (Korean font, IEEE style). Nothing is re-solved: placements
come from the eval records (occ), the curves from the same exact check (verify._probe_z_exact) as fig_board.py.
Instance choice is the rule fig_board.py / fig_rule_vs_mott.py already used (fixed before this export):
  Fig. 1  median count among the policy's solved problems (--pick median)
  Fig. 11 the problem whose (rule - MoTT+P1) count difference is the median among problems both solve
Run (server 1): .venv/bin/python scripts/dump_board_data.py  -> reports/fig_data/boards.json"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decap_loop.bank import load_bank  # noqa: E402
from decap_loop.pdn import target_at  # noqa: E402
from decap_loop.verify import _inst_data, _probe_z_exact  # noqa: E402

POOL = "runs/pool_v1"
R = "runs/2026-10-29/r2pk_rl/eval"


def find(set_, method):
    for d in (R, "runs/2026-10-28/baselines/eval", "runs/repair10/eval", "runs/repair16/eval"):
        p = os.path.join(d, f"{set_}__{method}.json")
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(f"{set_}__{method}")


def board(batch, rec, b):
    torch.set_num_threads(1)
    cfg = batch.cfg
    mask, ent, P = _inst_data(batch, b)
    occ = np.asarray(rec["occ"][b])
    f = cfg.check_freqs()
    zt = target_at(batch.zt_par[b:b + 1], f)[0]
    bare = (_probe_z_exact(mask, ent, P, None, cfg, f) / zt[:, None]).max(1)
    after = (_probe_z_exact(mask, ent, P, occ, cfg, f) / zt[:, None]).max(1)
    return dict(set=rec["set"], tech=rec["tech"], method=rec["method"], inst=int(b), H=int(batch.H), W=int(batch.W),
                mask=np.asarray(mask).reshape(-1).astype(int).tolist(),
                keepout=batch.keepout[b].reshape(-1).numpy().astype(int).tolist(),
                probes=np.asarray(P).astype(int).tolist(), entries=np.asarray(ent).astype(int).tolist(),
                occ=occ.astype(int).tolist(), count=int((occ >= 0).sum()), ok=bool(rec["ok"][b]),
                decaps=[[float(c), float(esl), float(esr)] for c, esl, esr in cfg.decaps],
                f=np.asarray(f).tolist(), bare=np.asarray(bare).tolist(), after=np.asarray(after).tolist(),
                worst=float(np.max(after)))


def median_pick(rec):
    ok = np.asarray(rec["ok"], bool); cnt = np.asarray(rec["count"])
    idx = np.flatnonzero(ok)
    return int(idx[np.argsort(cnt[idx], kind="stable")[len(idx) // 2]])


def main():
    out = {"fig1": [], "fig11": []}
    for t in ("pcb", "pkg", "ip"):
        rec = find(f"eval_{t}10_0", "r2pk_rl")
        batch, _ = load_bank(os.path.join(POOL, rec["set"] + ".pkl"))
        b = median_pick(rec)
        out["fig1"].append(board(batch, rec, b))
        print("fig1", rec["set"], "inst", b, "count", out["fig1"][-1]["count"], "worst", round(out["fig1"][-1]["worst"], 3))
    rule, mott = find("eval_ip16_0", "modal_rule"), find("eval_ip16_0", "r2pk_rl")
    ok = np.asarray(rule["ok"], bool) & np.asarray(mott["ok"], bool)
    idx = np.flatnonzero(ok)
    d = np.asarray(rule["count"])[idx] - np.asarray(mott["count"])[idx]
    b = int(idx[np.argmin(np.abs(d - np.median(d)))])
    batch, _ = load_bank(os.path.join(POOL, "eval_ip16_0.pkl"))
    out["fig11"] = [board(batch, rule, b), board(batch, mott, b)]
    out["fig11_note"] = dict(both_ok=int(len(idx)), median_diff=float(np.median(d)), inst=b)
    print("fig11 inst", b, "rule", out["fig11"][0]["count"], "mott", out["fig11"][1]["count"])
    os.makedirs("reports/fig_data", exist_ok=True)
    json.dump(out, open("reports/fig_data/boards.json", "w"))
    print("wrote reports/fig_data/boards.json")


if __name__ == "__main__":
    main()
