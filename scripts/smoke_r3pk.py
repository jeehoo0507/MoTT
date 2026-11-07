"""Smoke for the PROPOSALS 11/07 G1 flag combinations before their runs (memory: tiny GPU fwd+bwd first): train.py
distill 30 iters + rl 3 iters on the GPU with the run_rung COMMON args and train pools.
  (no arg) r3pk   = --model loop --peak_refine            -> runs/analysis/smoke_r3pk.ok
  fc       r2pkfc = --model mott --peak_refine --fc_bias  -> runs/analysis/smoke_r2pkfc.ok
Run: .venv/bin/python scripts/smoke_r3pk.py [fc]
"""
import os
import shutil
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.chdir(ROOT)
    fc = sys.argv[1:] == ["fc"]
    tag = "r2pkfc" if fc else "r3pk"
    out = f"runs/analysis/smoke_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    common = "--d 128 --layers 4 --heads 8 --B 16 --M 8 --lr 3e-4 --val_every 100 --max_steps 64".split()
    arm = ["--model", "mott", "--peak_refine", "--fc_bias"] if fc else ["--model", "loop", "--peak_refine"]
    py = ".venv/bin/python"
    pools = ["--pool", "runs/pool_v1", "--teacher_dir", "data/teacher", "--seed", "0"]
    subprocess.run([py, "train.py", "--stage", "distill", *arm, *common, "--iters", "30", *pools, "--out", f"{out}/d"],
                   check=True)
    subprocess.run([py, "train.py", "--stage", "rl", *arm, *common, "--iters", "3", *pools, "--init", f"{out}/d/last.pt",
                    "--out", f"{out}/r"], check=True)
    ck = torch.load(f"{out}/r/last.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    assert a["feat"] == "mott_pk" and a["peak_refine"] and a["model"] == arm[1] and a["fc_bias"] == fc
    open(f"runs/analysis/smoke_{tag}.ok", "w").write("ok\n")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
