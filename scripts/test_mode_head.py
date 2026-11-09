"""Unit check of the --mode_head add logit (arch c', decap_loop/model.py MoTT.mode_add), CPU only:
(1) at init the mode-head policy gives the same logits as its parent (same shared weights),
(2) rollout + backward on a small batch gives finite grads that reach the new head, for mott and loop.
Run: .venv/bin/python scripts/test_mode_head.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decap_loop.loop_model import LoopMoTT  # noqa: E402
from decap_loop.modal_env import make_env  # noqa: E402
from decap_loop.model import MoTT  # noqa: E402
from decap_loop.pdn import sample_batch  # noqa: E402
from decap_loop.rollout import backward_logp, rollout  # noqa: E402
from decap_loop.sim import K_MODES, mott_dims  # noqa: E402
from decap_loop.techs import DIFFICULTY, TECHS  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(0)
dev = torch.device("cpu")
cfg = TECHS["pkg"]
b = sample_batch(2, 8, 8, cfg, np.random.default_rng(3), difficulty=DIFFICULTY["pkg"], device=dev)
dims = mott_dims(cfg.n_types)
for name, mk in (("mott", lambda mh: MoTT(cfg.n_types, K_MODES, dims, d=32, layers=2, heads=2, fc=True, mode_head=mh)),
                 ("loop", lambda mh: LoopMoTT(cfg.n_types, K_MODES, dims, d=32, heads=2, fc=True, mode_head=mh))):
    parent, child = mk(False), mk(True)
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("mh_") for k in missing), (missing, unexpected)
    parent.eval(), child.eval()
    with torch.no_grad():
        o1 = rollout(parent, make_env(b, torch.arange(b.B, device=dev), cfg, "mott", max_steps=4), greedy=True, amp=False)
        o2 = rollout(child, make_env(b, torch.arange(b.B, device=dev), cfg, "mott", max_steps=4), greedy=True, amp=False)
    assert o1["count"].tolist() == o2["count"].tolist() and o1["feasible"].tolist() == o2["feasible"].tolist()
    child.train()
    env = make_env(b, torch.arange(b.B, device=dev).repeat_interleave(2), cfg, "mott", max_steps=6)
    out = rollout(child, env, store=True, amp=False)
    loss = backward_logp(child, out["traj"], torch.randn(env.E, device=dev), b.H, b.W, amp=False, scale=1 / env.E)
    assert np.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in child.parameters())
    g = child.mh_phi[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "no gradient reaches the mode head"
    print(f"{name}: init == parent (greedy count {o1['count'].tolist()}), train loss {loss:.3f}, "
          f"mode-head grad {g.abs().sum():.2e} ok")
print("MODE HEAD OK")
