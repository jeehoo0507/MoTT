"""Smoke for QUEUE 11/07 10:45 KST candidate C3 `r3pk` (`--model loop --peak_refine`) before it is queued (memory: tiny
GPU fwd+bwd first): train.py distill 30 iters + rl 3 iters on the GPU with the run_rung COMMON args and train pools.
Writes runs/analysis/smoke_r3pk.ok on success.
Run: .venv/bin/python scripts/smoke_r3pk.py
"""
import os
import shutil
import subprocess

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "runs/analysis/smoke_r3pk"


def main():
    os.chdir(ROOT)
    shutil.rmtree(OUT, ignore_errors=True)
    common = "--d 128 --layers 4 --heads 8 --B 16 --M 8 --lr 3e-4 --val_every 100 --max_steps 64".split()
    arm = ["--model", "loop", "--peak_refine"]
    py = ".venv/bin/python"
    pools = ["--pool", "runs/pool_v1", "--teacher_dir", "data/teacher", "--seed", "0"]
    subprocess.run([py, "train.py", "--stage", "distill", *arm, *common, "--iters", "30", *pools, "--out", f"{OUT}/d"],
                   check=True)
    subprocess.run([py, "train.py", "--stage", "rl", *arm, *common, "--iters", "3", *pools, "--init", f"{OUT}/d/last.pt",
                    "--out", f"{OUT}/r"], check=True)
    ck = torch.load(f"{OUT}/r/last.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["feat"] == "mott_pk" and ck["args"]["peak_refine"] and ck["args"]["model"] == "loop"
    open("runs/analysis/smoke_r3pk.ok", "w").write("ok\n")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
