"""Smoke for `--peak_refine --dedup_modes` (PROPOSALS 11/04 P3, conditional) before any run is queued.
1. refine_at + distinct_neighbours on synthetic sorted curves: without repeated frequencies it equals peak_env.refine
   bit for bit; on a curve whose peak frequency is repeated it gives the vertex excess where refine gives 0.
2. On eval_pkg10_0 / eval_pcb10_0 (bare boards): PkDdEnv features = DedupEnv features with 2 channels inserted per
   token (all other channels bit-identical), shape = PkEnv's; reports the share of valid tokens with excess > 0.01
   for PkEnv vs PkDdEnv (not judged).
3. train.py distill 5 iters + rl 2 iters with --model mott --peak_refine --dedup_modes on the CPU.
Writes runs/analysis/smoke_pkdd.ok on success.  Run: .venv/bin/python scripts/smoke_pkdd.py
"""
import os
import shutil
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = "runs/analysis/smoke_pkdd"


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    from decap_loop.bank import load_bank
    from decap_loop.dedup_env import DedupEnv
    from decap_loop.peak_env import PkEnv, mott_pk_dims, refine
    from decap_loop.pkdd_env import PkDdEnv, distinct_neighbours, refine_at
    from decap_loop.sim import K_MODES, mott_dims
    # 1. synthetic
    g = torch.rand(4, 40, dtype=torch.float64)
    lf = torch.cumsum(torch.rand(4, 40, dtype=torch.float64) + 0.01, 1)
    j = torch.randint(0, 40, (4, 8))
    jl, jr = distinct_neighbours(lf, j)
    assert torch.equal(jl, (j - 1).clamp(min=0)) and torch.equal(jr, (j + 1).clamp(max=39))
    e1, y1 = refine(g, lf, j)
    e2, y2 = refine_at(g, lf, jl, j, jr)
    assert torch.equal(e1, e2) and torch.equal(y1, y2)
    lf = torch.tensor([[0., 1., 2., 2., 2., 3., 4.]], dtype=torch.float64)
    lv = torch.tensor([[0., 1., 2., 2., 2., 1.5, 0.]], dtype=torch.float64)
    jj = torch.tensor([[2]])
    jl, jr = distinct_neighbours(lf, jj)
    assert jl.item() == 1 and jr.item() == 5, (jl, jr)
    assert refine(lv, lf, jj)[0].item() == 0.0                  # raw neighbour is a copy -> no vertex
    assert refine_at(lv, lf, jl, jj, jr)[0].item() > 0.0        # distinct neighbours: vertex right of 2
    print("1. refine_at / distinct_neighbours ok", flush=True)
    # 2. bare boards
    for tech in ("pkg", "pcb"):
        b, _ = load_bank(f"runs/pool_v1/eval_{tech}10_0.pkl")
        ib = torch.arange(b.B)
        T = b.cfg.n_types
        dn, dm, de, dc = mott_dims(T)
        _, dm2, _, _ = mott_pk_dims(T)
        e_dd, e_pk = DedupEnv(b, ib, b.cfg, feat_mode="mott_dd"), PkEnv(b, ib, b.cfg, feat_mode="mott_pk")
        e_pd = PkDdEnv(b, ib, b.cfg, feat_mode="mott_pkdd")
        xd, xp, xq = e_dd.observe()["x"], e_pk.observe()["x"], e_pd.observe()["x"]
        assert xq.shape == xp.shape, (xq.shape, xp.shape)
        o = dn + K_MODES * de
        assert torch.equal(xq[..., :o], xd[..., :o]), "node/edge part differs from DedupEnv"
        assert torch.equal(xq[..., o + K_MODES * dm2:], xd[..., o + K_MODES * dm:]), "decap part differs"
        mq = xq[:, 0, o:o + K_MODES * dm2].view(-1, K_MODES, dm2)
        md = xd[:, 0, o:o + K_MODES * dm].view(-1, K_MODES, dm)
        mp = xp[:, 0, o:o + K_MODES * dm2].view(-1, K_MODES, dm2)
        assert torch.equal(mq[..., :dm], md), "first 4 mode channels differ from DedupEnv"
        vq, vp = mq[..., 3] > 0.5, mp[..., 3] > 0.5
        sq = (mq[..., 4][vq] > 0.01).float().mean().item()
        sp = (mp[..., 4][vp] > 0.01).float().mean().item()
        print(f"2. {tech}: features ok; valid tokens with refined excess > 0.01: PkEnv {sp:.2f} -> PkDdEnv {sq:.2f}",
              flush=True)
    # 3. CPU distill + rl
    shutil.rmtree(OUT, ignore_errors=True)
    common = "--d 32 --layers 2 --heads 2 --B 4 --M 4 --lr 3e-4 --val_every 1000 --max_steps 24 --device cpu".split()
    py = ".venv/bin/python"
    pools = ["--pool", "runs/pool_v1", "--teacher_dir", "data/teacher", "--seed", "0"]
    flags = ["--model", "mott", "--peak_refine", "--dedup_modes"]
    subprocess.run([py, "train.py", "--stage", "distill", *flags, *common, "--iters", "5", *pools, "--out", f"{OUT}/d"],
                   check=True)
    subprocess.run([py, "train.py", "--stage", "rl", *flags, *common, "--iters", "2", *pools,
                    "--init", f"{OUT}/d/last.pt", "--out", f"{OUT}/r"], check=True)
    ck = torch.load(f"{OUT}/r/last.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["feat"] == "mott_pkdd" and ck["args"]["peak_refine"] and ck["args"]["dedup_modes"]
    open("runs/analysis/smoke_pkdd.ok", "w").write("ok\n")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
