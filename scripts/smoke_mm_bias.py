"""Smoke for `--mm_bias` (PROPOSALS 11/07 M1, decap_loop/model.py MoTT.mm_bias), CPU, synthetic boards (no pool).
1. With tau -> inf the mm model equals the same-weights model without mm (the bias is exactly zero); with tau = 1
   decade the logits differ (the bias is live).
2. Gradient reaches mm_logtau (8 parameters, one per head); no other new parameters.
3. A greedy rollout with an untrained MoTT+P1+G1+M1 runs end to end (train.py make_policy with --mm_bias).
Run: python scripts/smoke_mm_bias.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from decap_loop.pdn import PDNConfig, sample_batch  # noqa: E402
from decap_loop.peak_env import PkEnv  # noqa: E402
from decap_loop.rollout import rollout  # noqa: E402
import train  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)
cfg = PDNConfig()
b = sample_batch(6, 8, 8, cfg, rng)
ib = torch.arange(b.B)
env = PkEnv(b, ib, cfg, feat_mode="mott_pk", max_steps=40)
obs = env.observe()
x, valid = obs["x"], env.valid
base = dict(model="mott", d=32, layers=2, heads=4, peak_refine=True, fc_bias=True, feat="mott_pk")
torch.manual_seed(1); p0 = train.make_policy(cfg, dict(base))
torch.manual_seed(1); p1 = train.make_policy(cfg, dict(base, mm_bias=True))
missing, unexpected = p1.load_state_dict(p0.state_dict(), strict=False)
assert missing == ["mm_logtau"] and not unexpected, (missing, unexpected)
assert p1.mm_logtau.numel() == 4, p1.mm_logtau.shape
n0, n1 = sum(q.numel() for q in p0.parameters()), sum(q.numel() for q in p1.parameters())
assert n1 - n0 == 4, (n0, n1)
p0.eval(); p1.eval()


def logits(pol):
    node, stop = pol(x, valid, b.H, b.W)
    return torch.cat([node.reshape(node.shape[0], -1), stop.reshape(-1, 1)], 1)


with torch.no_grad():
    l0 = logits(p0)
    p1.mm_logtau.fill_(30.0)            # tau = e^30: bias ~ 0
    l_inf = logits(p1)
    p1.mm_logtau.fill_(0.0)             # tau = 1 decade
    l_1 = logits(p1)
assert torch.allclose(l0, l_inf, atol=1e-5), float((l0 - l_inf).abs().max())
assert not torch.allclose(l0, l_1, atol=1e-5), "bias has no effect at tau = 1 decade"
print(f"1. tau->inf equals no-mm model (max |diff| {float((l0 - l_inf).abs().max()):.2e}); tau=1 differs "
      f"(max |diff| {float((l0 - l_1).abs().max()):.3f})", flush=True)
p1.train()
logits(p1).sum().backward()
assert p1.mm_logtau.grad is not None and bool(p1.mm_logtau.grad.abs().sum() > 0), "no grad to mm_logtau"
print("2. grad reaches mm_logtau:", p1.mm_logtau.grad.tolist(), "| new params:", n1 - n0, flush=True)
p1.eval()
out = rollout(p1, PkEnv(b, ib, cfg, feat_mode="mott_pk", max_steps=40), greedy=True, amp=False)
print("3. greedy rollout ok: count", out["count"].tolist(), "feasible", out["feasible"].tolist(), flush=True)
open("runs/analysis/smoke_mm_bias.ok", "w").write("ok\n") if os.path.isdir("runs/analysis") else None
print("OK")
