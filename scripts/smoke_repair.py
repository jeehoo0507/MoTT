"""Smoke for the repair stage (PROPOSALS 11/03 T2, eval_policy.py --repair) before its eval jobs are queued:
a 6-problem 8-cell pkg bank (cutting plane with g+p, like build_pools.py) in runs/analysis/smoke_repair/, then
eval_policy.py with --rounds 1 --repair 2 (few rounds so that some solutions still fail and the repair path runs)
for greedy+prune, modal_rule and r2_mott_rl. Checks only that the repair path runs and that it never touches an
instance that already passed; the numbers are not reported. Writes runs/analysis/smoke_repair.ok on success.
Run: .venv/bin/python scripts/smoke_repair.py
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = "runs/analysis/smoke_repair"


def build():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from decap_loop.bank import save_bank
    from decap_loop.baselines import greedy
    from decap_loop.pdn import sample_batch
    from decap_loop.techs import DIFFICULTY, PKG
    from decap_loop.verify import cutting_plane
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(7)
    b = sample_batch(6, 8, 8, PKG, np.random.default_rng(7), difficulty=DIFFICULTY["pkg"])
    r, info = cutting_plane(lambda bb: greedy(bb, PKG), b, PKG, max_rounds=10)
    ok = info["worst"] <= np.log10(1.01)
    cnt = (r["occ"] >= 0).sum(1).cpu().numpy()
    save_bank(os.path.join(OUT, "eval_pkg8_0.pkl"), b,
              {"greedy+prune": dict(occ=r["occ"].cpu().numpy(), worst=info["worst"], ok=ok, count=cnt)})
    print("built", ok.tolist(), cnt.tolist(), "rounds", info["rounds"], flush=True)


def main():
    os.chdir(ROOT)
    if not os.path.exists(os.path.join(OUT, "eval_pkg8_0.pkl")):
        build()
    cmd = [".venv/bin/python", "scripts/eval_policy.py", "--pool", OUT, "--sets", "eval_pkg8_0", "--methods", "gp,modal_rule",
           "--policy", "r2_mott_rl=runs/2026-10-28/r2_mott_rl/best.pt", "--rounds", "1", "--repair", "2",
           "--workers", "1", "--out", OUT]
    subprocess.run(cmd, check=True)
    bad = []
    for f in sorted(glob.glob(os.path.join(OUT, "eval", "*+rep2.json"))):
        r = json.load(open(f))
        same = all(c == c0 for c, c0, o0 in zip(r["count"], r["count_pre"], r["ok_pre"]) if o0)
        print(os.path.basename(f), "ok pre", sum(r["ok_pre"]), "-> post", sum(r["ok"]), "| count pre", r["count_pre"],
              "post", r["count"], "| repair rounds used", r["repair_used"], "| passed untouched", same, flush=True)
        if not same or sum(r["ok"]) < sum(r["ok_pre"]):
            bad.append(f)
    assert len(glob.glob(os.path.join(OUT, "eval", "*+rep2.json"))) == 3, "missing records"
    assert not bad, bad
    os.rename(os.path.join(OUT, "eval"), os.path.join(OUT, "smoke_records"))   # keep them out of the RESULTS glob
    open("runs/analysis/smoke_repair.ok", "w").write("ok\n")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
