"""Unit check of decap_loop/loop_model.py (rung 3): rollout + backward with sampled R, eval at R=2..16.
Run: .venv/bin/python scripts/test_loop_model.py"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decap_loop.loop_model import LoopMoTT  # noqa: E402
from decap_loop.modal_env import make_env  # noqa: E402
from decap_loop.pdn import sample_batch  # noqa: E402
from decap_loop.rollout import backward_logp, rollout  # noqa: E402
from decap_loop.sim import K_MODES, mott_dims  # noqa: E402
from decap_loop.techs import DIFFICULTY, TECHS  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = TECHS["pkg"]
t = time.time()
b = sample_batch(3, 8, 8, cfg, np.random.default_rng(3), difficulty=DIFFICULTY["pkg"], device=dev)
print(f"batch in {time.time()-t:.1f}s")
pol = LoopMoTT(cfg.n_types, K_MODES, mott_dims(cfg.n_types), d=32, heads=2, r_train=(2, 8), r_eval=8).to(dev)
rng = np.random.default_rng(0)
for it in range(2):
    R = pol.sample_R(rng)
    env = make_env(b, torch.arange(b.B, device=dev).repeat_interleave(2), cfg, "mott", max_steps=10)
    out = rollout(pol, env, store=True, amp=False)
    loss = backward_logp(pol, out["traj"], torch.randn(env.E, device=dev), b.H, b.W, amp=False, scale=1 / env.E)
    assert np.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in pol.parameters())
    print(f"train it {it}: R={R} steps {len(out['traj'])} loss {loss:.3f} ok")
pol.eval()
for R in (2, 4, 8, 16):
    pol.set_eval_R(R)
    env = make_env(b, torch.arange(b.B, device=dev), cfg, "mott", max_steps=10)
    with torch.no_grad():
        out = rollout(pol, env, greedy=True, amp=False)
    print(f"eval R={R:2d}: count {out['count'].tolist()} feasible {out['feasible'].tolist()}")
print("LOOP MODEL OK")
