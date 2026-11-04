"""Smoke for `--dedup_modes` (PROPOSALS 11/04 P2) before its runs are queued.
1. distinct_peaks on synthetic sorted curves: a duplicated peak counts once, a duplicated point on a slope is no peak,
   and without duplicates the mask equals the sim rule (>= both neighbours).
2. On eval_pkg10_0 / eval_pcb10_0 (bare boards): DedupEnv features vs sim features_mott — node and decap parts
   bit-identical, valid token frequencies distinct; episodes whose sim token set is already distinct give bit-identical
   features; reports distinct valid tokens per state before / after (not judged).
3. train.py distill 5 iters + rl 2 iters with --model mott --dedup_modes on the CPU (no new model op, so no GPU-specific
   risk; the GPU is busy with P1). Writes runs/analysis/smoke_dedup_modes.ok on success.
Run: .venv/bin/python scripts/smoke_dedup_modes.py
"""
import os
import shutil
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = "runs/analysis/smoke_dedup_modes"


def sim_rule(lvs):
    ninf = torch.full_like(lvs[:, :1], -float("inf"))
    return (lvs >= torch.cat([ninf, lvs[:, :-1]], 1)) & (lvs >= torch.cat([lvs[:, 1:], ninf], 1))


def main():
    os.chdir(ROOT)
    torch.set_num_threads(1)
    from decap_loop.dedup_env import DedupEnv, distinct_peaks
    from decap_loop.sim import Env, mott_dims, K_MODES
    from decap_loop.bank import load_bank
    # 1. synthetic
    f = torch.tensor([[1., 2., 3., 3., 3., 4., 5., 6., 6., 7.]], dtype=torch.float64)
    v = torch.tensor([[0., 1., 2., 2., 2., 1., 0., 3., 3., 4.]], dtype=torch.float64)
    pk = distinct_peaks(v, f)[0].tolist()
    assert pk == [False, False, True, False, False, False, False, False, False, True], pk   # 3 once; 6 on a slope: no
    assert sim_rule(v)[0].sum() > 2                                                         # sim counts the copies
    g = torch.rand(4, 50, dtype=torch.float64)
    ff = torch.arange(50, dtype=torch.float64).expand(4, 50)
    assert torch.equal(distinct_peaks(g, ff), sim_rule(g))
    print("1. distinct_peaks ok", flush=True)
    # 2. features on bare boards
    T = None
    for tech in ("pkg", "pcb"):
        b, _ = load_bank(f"runs/pool_v1/eval_{tech}10_0.pkl")
        ib = torch.arange(b.B)
        e_ref, e_dd = Env(b, ib, b.cfg, feat_mode="mott"), DedupEnv(b, ib, b.cfg, feat_mode="mott_dd")
        T = b.cfg.n_types
        dn, dm, de, dc = mott_dims(T)
        xr, xd = e_ref.observe()["x"], e_dd.observe()["x"]
        assert xr.shape == xd.shape
        o = dn + K_MODES * de
        assert torch.equal(xr[..., :dn], xd[..., :dn]), "node part differs"
        assert torch.equal(xr[..., o + K_MODES * dm:], xd[..., o + K_MODES * dm:]), "decap part differs"
        mr = xr[:, 0, o:o + K_MODES * dm].view(-1, K_MODES, dm)
        md = xd[:, 0, o:o + K_MODES * dm].view(-1, K_MODES, dm)
        nd_r, nd_d, same = [], [], 0
        for e in range(b.B):
            fr = mr[e, mr[e, :, 3] > 0.5, 0]
            fd = md[e, md[e, :, 3] > 0.5, 0]
            assert len(torch.unique(fd)) == len(fd), ("dedup tokens repeat", e, fd)
            nd_r.append(len(torch.unique(fr)))
            nd_d.append(len(fd))
            if len(torch.unique(fr)) == len(fr):
                assert torch.equal(xr[e], xd[e]), ("distinct sim token set but features differ", e)
                same += 1
        print(f"2. {tech}: features ok; distinct valid tokens per bare state sim mean {sum(nd_r) / b.B:.2f} -> dedup "
              f"{sum(nd_d) / b.B:.2f}; sim token set already distinct (bit-identical) in {same}/{b.B}", flush=True)
    # 3. CPU distill + rl
    shutil.rmtree(OUT, ignore_errors=True)
    common = "--d 32 --layers 2 --heads 2 --B 4 --M 4 --lr 3e-4 --val_every 1000 --max_steps 24 --device cpu".split()
    py = ".venv/bin/python"
    pools = ["--pool", "runs/pool_v1", "--teacher_dir", "data/teacher", "--seed", "0"]
    subprocess.run([py, "train.py", "--stage", "distill", "--model", "mott", "--dedup_modes", *common, "--iters", "5",
                    *pools, "--out", f"{OUT}/d"], check=True)
    subprocess.run([py, "train.py", "--stage", "rl", "--model", "mott", "--dedup_modes", *common, "--iters", "2",
                    *pools, "--init", f"{OUT}/d/last.pt", "--out", f"{OUT}/r"], check=True)
    ck = torch.load(f"{OUT}/r/last.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["feat"] == "mott_dd" and ck["args"]["dedup_modes"]
    open("runs/analysis/smoke_dedup_modes.ok", "w").write("ok\n")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
