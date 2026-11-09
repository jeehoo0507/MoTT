"""Smoke for `--peak_refine` (PROPOSALS 11/03 P1) before its runs are queued (memory: tiny GPU fwd+bwd first).
1. refine() on a synthetic concave sampled peak recovers the vertex; flat / edge / convex cases give 0.
2. On eval_pkg10_0 (bare boards): PkEnv features = sim features_mott with 2 channels inserted per token (all other
   channels bit-identical), extra channels finite; reports how often a token gets a positive refinement (not judged).
3. train.py distill 30 iters + rl 3 iters with --model mott --peak_refine on the GPU (COMMON of run_rung, train pools).
Writes runs/analysis/smoke_peak_refine.ok on success.
Run: .venv/bin/python scripts/smoke_peak_refine.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = "runs/analysis/smoke_peak_refine"


def main():
    os.chdir(ROOT)
    from decap_loop.peak_env import refine, PkEnv, mott_pk_dims
    from decap_loop.sim import Env, mott_dims, K_MODES
    from decap_loop.bank import load_bank
    # 1. synthetic: y = 1 - 30 (x - 0.013)^2 sampled at x = 0, 0.02, 0.04 ... ; peak index 1 (x=0.02 closest)
    x = torch.arange(0, 0.2, 0.02, dtype=torch.float64)[None]
    y = 1 - 30 * (x - 0.013) ** 2
    j = torch.tensor([[1]])
    exc, yr = refine(y, x, j)
    assert abs(float(yr) - 1.0) < 1e-9, (float(yr), float(exc))
    e0, _ = refine(y, x, torch.tensor([[0]]))
    assert float(e0) == 0.0
    e1, _ = refine(-y, x, torch.tensor([[1]]))           # convex -> 0
    assert float(e1) == 0.0
    print("1. refine ok: synthetic vertex", float(yr), "excess", float(exc), flush=True)
    # 2. features on bare boards
    b, _ = load_bank("runs/pool_v1/eval_pkg10_0.pkl")
    ib = torch.arange(b.B)
    e_ref, e_pk = Env(b, ib, b.cfg, feat_mode="mott"), PkEnv(b, ib, b.cfg, feat_mode="mott_pk")
    T = b.cfg.n_types
    dn, dm, de, dc = mott_dims(T)
    _, dm2, _, _ = mott_pk_dims(T)
    xr, xp = e_ref.observe()["x"], e_pk.observe()["x"]
    o = dn + K_MODES * de
    assert xp.shape[-1] == xr.shape[-1] + K_MODES * (dm2 - dm), (xp.shape, xr.shape)
    assert torch.equal(xp[..., :o], xr[..., :o]) and torch.equal(xp[..., o + K_MODES * dm2:], xr[..., o + K_MODES * dm:])
    mp = xp[:, 0, o:o + K_MODES * dm2].view(-1, K_MODES, dm2)
    mr = xr[:, 0, o:o + K_MODES * dm].view(-1, K_MODES, dm)
    assert torch.equal(mp[..., :dm], mr)
    ext = mp[..., dm:]
    assert torch.isfinite(ext).all()
    v = mr[..., 3] > 0.5
    print(f"2. features ok: tokens {int(v.sum())}, refined excess > 0.01 on {float((ext[..., 0][v] > 0.01).float().mean()):.2f}, "
          f"median excess {float(ext[..., 0][v].median()):.3f}, refined-violating {float(ext[..., 1][v].mean()):.2f} "
          f"vs grid-violating {float(mr[..., 2][v].mean()):.2f}", flush=True)
    # 3. GPU distill + rl
    shutil.rmtree(OUT, ignore_errors=True)
    common = "--d 128 --layers 4 --heads 8 --B 16 --M 8 --lr 3e-4 --val_every 100 --max_steps 64".split()
    py = ".venv/bin/python"
    pools = ["--pool", "runs/pool_v1", "--teacher_dir", "data/teacher", "--seed", "0"]
    subprocess.run([py, "train.py", "--stage", "distill", "--model", "mott", "--peak_refine", *common, "--iters", "30",
                    *pools, "--out", f"{OUT}/d"], check=True)
    subprocess.run([py, "train.py", "--stage", "rl", "--model", "mott", "--peak_refine", *common, "--iters", "3",
                    *pools, "--init", f"{OUT}/d/last.pt", "--out", f"{OUT}/r"], check=True)
    ck = torch.load(f"{OUT}/r/last.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["feat"] == "mott_pk" and ck["args"]["peak_refine"]
    open("runs/analysis/smoke_peak_refine.ok", "w").write("ok\n")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
