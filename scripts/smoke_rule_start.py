"""Smoke for the rule-start stage (PROPOSALS 11/03 T4, eval_policy.py --start rule) before its eval jobs are queued:
on the 6-problem 8-cell pkg bank of smoke_repair.py, one cutting-plane round (same grid for both) of modal_rule and
modal_rule>r2_mott_rl. Checks only that the path runs and that, wherever the rule met the target on the grid, the
policy never ends with more decaps than the rule (on a met target the env only allows remove / change type / stop).
The numbers are not reported. Writes runs/analysis/smoke_rule_start.ok on success.
Run: .venv/bin/python scripts/smoke_rule_start.py   (after scripts/smoke_repair.py has built the bank)
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BANK = "runs/analysis/smoke_repair"
OUT = "runs/analysis/smoke_rule_start"


def main():
    os.chdir(ROOT)
    assert os.path.exists(os.path.join(BANK, "eval_pkg8_0.pkl")), "run scripts/smoke_repair.py first"
    shutil.rmtree(os.path.join(OUT, "eval"), ignore_errors=True)
    cmd = [".venv/bin/python", "scripts/eval_policy.py", "--pool", BANK, "--sets", "eval_pkg8_0", "--methods", "modal_rule",
           "--policy", "r2_mott_rl=runs/2026-10-28/r2_mott_rl/best.pt", "--start", "rule", "--rounds", "1",
           "--workers", "1", "--out", OUT]
    subprocess.run(cmd, check=True)
    rule = json.load(open(os.path.join(OUT, "eval", "eval_pkg8_0__modal_rule.json")))
    pol = json.load(open(os.path.join(OUT, "eval", "eval_pkg8_0__modal_rule>r2_mott_rl.json")))
    print("rule  ok", rule["ok"], "count", rule["count"], flush=True)
    print("rule> ok", pol["ok"], "count", pol["count"], flush=True)
    # grid-feasible rule solutions: the policy can only remove / swap, so never more decaps
    import numpy as np
    from decap_loop.bank import load_bank
    from decap_loop.verify import grid_worst
    import torch
    b, _ = load_bank(os.path.join(BANK, "eval_pkg8_0.pkl"))
    occ = torch.tensor(rule["occ"], device=b.valid.device)
    gw = grid_worst(b, torch.arange(b.B), occ, b.cfg)
    bad = [i for i in range(b.B) if gw[i] <= 1e-6 and pol["count"][i] > rule["count"][i]]
    assert not bad, f"policy added decaps on a grid-feasible rule start: {bad}"
    os.rename(os.path.join(OUT, "eval"), os.path.join(OUT, "smoke_records"))   # keep them out of the RESULTS glob
    open("runs/analysis/smoke_rule_start.ok", "w").write("ok\n")
    print("smoke ok", int(np.sum(gw <= 1e-6)), "grid-feasible rule starts", flush=True)


if __name__ == "__main__":
    main()
