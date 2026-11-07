"""GPU smoke for `--peak_gate` (QUEUE 11/07 14:30 KST step 2, PROPOSALS 11/07 C1'): the same as step 3 of
scripts/smoke_peak_refine.py (train.py distill 30 iters + rl 3 iters on the GPU, run_rung COMMON-like, train pools)
with --model mott --peak_gate; the checkpoint must carry feat mott_pkgate (gate on) and peak_refine (implied).
Writes runs/analysis/smoke_peak_gate_gpu.ok on success. CPU part: scripts/smoke_peak_gate.py (human).
Run: .venv/bin/python scripts/smoke_peak_gate_gpu.py
"""
import os
import shutil
import subprocess

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "runs/analysis/smoke_peak_gate_gpu"


def main():
    os.chdir(ROOT)
    assert torch.cuda.is_available()
    shutil.rmtree(OUT, ignore_errors=True)
    common = "--d 128 --layers 4 --heads 8 --B 16 --M 8 --lr 3e-4 --val_every 100 --max_steps 64".split()
    py = ".venv/bin/python"
    pools = ["--pool", "runs/pool_v1", "--teacher_dir", "data/teacher", "--seed", "0"]
    subprocess.run([py, "train.py", "--stage", "distill", "--model", "mott", "--peak_gate", *common, "--iters", "30",
                    *pools, "--out", f"{OUT}/d"], check=True)
    subprocess.run([py, "train.py", "--stage", "rl", "--model", "mott", "--peak_gate", *common, "--iters", "3",
                    *pools, "--init", f"{OUT}/d/last.pt", "--out", f"{OUT}/r"], check=True)
    ck = torch.load(f"{OUT}/r/last.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["feat"] == "mott_pkgate" and ck["args"]["peak_refine"] and ck["args"]["peak_gate"], ck["args"]
    open("runs/analysis/smoke_peak_gate_gpu.ok", "w").write("ok\n")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
