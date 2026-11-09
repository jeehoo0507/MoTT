"""Unit check of decap_loop/modal_env.py: feature size matches feat_dim_any, all finite, policy
rollout + backward works for feat modal / modal_only, on a small PCB and IP batch (GPU if present).

Run: .venv/bin/python scripts/test_modal_env.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decap_loop.modal_env import feat_dim_any, make_env  # noqa: E402
from decap_loop.model import GridPolicy  # noqa: E402
from decap_loop.pdn import sample_batch  # noqa: E402
from decap_loop.rollout import backward_logp, rollout  # noqa: E402
from decap_loop.techs import DIFFICULTY, TECHS  # noqa: E402

torch.set_num_threads(1)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for tech in ("pcb", "ip"):
    cfg = TECHS[tech]
    t = time.time()
    b = sample_batch(3, 8, 8, cfg, np.random.default_rng(7), difficulty=DIFFICULTY[tech], device=dev)
    print(f"{tech}: batch 3x8x8 in {time.time()-t:.1f}s, F={b.freqs.shape[1]}")
    for fm in ("modal", "modal_only", "coord", "phys"):
        env = make_env(b, torch.arange(b.B, device=dev).repeat_interleave(2), cfg, fm, max_steps=12)
        obs = env.observe()
        D = feat_dim_any(cfg, fm)
        assert obs["x"].shape[-1] == D, (fm, obs["x"].shape, D)
        assert torch.isfinite(obs["x"]).all(), fm
        for prior in (False, True):
            pol = GridPolicy(D, cfg.n_types, d=32, layers=1, heads=2, prior=prior).to(dev)
            env = make_env(b, torch.arange(b.B, device=dev).repeat_interleave(2), cfg, fm, max_steps=12)
            out = rollout(pol, env, store=True, amp=False)
            loss = backward_logp(pol, out["traj"], torch.randn(env.E, device=dev), b.H, b.W, amp=False, scale=1 / env.E)
            assert np.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in pol.parameters())
        x = obs["x"][0, env.valid[0]]
        print(f"  {fm:10s} D={D:3d} ok; feature range [{x.min().item():.2f}, {x.max().item():.2f}] "
              f"steps {len(out['traj'])} count {out['count'].tolist()}")
print("MODAL ENV OK")
