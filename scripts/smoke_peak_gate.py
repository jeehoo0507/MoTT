"""Smoke for `--peak_gate` (PROPOSALS 11/07 C1', decap_loop/pkgate_env.py), CPU, synthetic boards (no pool needed).
1. PkGateEnv features == PkEnv features on bare boards (same 6-wide tokens).
2. Gate is at least as strict as the grid test: gated feas => grid feas; count boards where it differs.
3. A random-action rollout in the gated env ends (done) and its final boards obey the rule; a greedy rollout with
   an untrained MoTT runs end to end (train.py make_policy with --peak_gate).
Run: python scripts/smoke_peak_gate.py
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
from decap_loop.pkgate_env import PkGateEnv  # noqa: E402
from decap_loop.rollout import rollout  # noqa: E402
from decap_loop.sim import FEAS_TOL  # noqa: E402
import train  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)
cfg = PDNConfig()
b = sample_batch(6, 8, 8, cfg, rng)
ib = torch.arange(b.B)
e_pk, e_g = PkEnv(b, ib, cfg, feat_mode="mott_pk", max_steps=40), PkGateEnv(b, ib, cfg, feat_mode="mott_pkgate", max_steps=40)
o1, o2 = e_pk.observe(), e_g.observe()
assert torch.equal(o1["x"], o2["x"]), "features differ on bare boards"
assert torch.equal(o1["mask"], o2["mask"]), "bare-board masks differ"   # bare boards: not met either way
print("1. features/mask identical on bare boards", o1["x"].shape, flush=True)
# 2/3. random rollout in the gated env; check the rule at every step
E = b.B
gen = torch.Generator().manual_seed(0)
stricter = 0
for t in range(60):
    obs = e_g.observe()
    grid_met = obs["m"] <= FEAS_TOL
    assert bool((~obs["feas"] | grid_met).all()), "gated feas without grid feas"
    stricter += int((grid_met & ~obs["feas"]).sum())
    if bool(e_g.done.all()):
        break
    mask = obs["mask"].clone(); mask[e_g.done] = False; mask[e_g.done, e_g.stop_idx] = True
    p = mask.float() / mask.float().sum(1, keepdim=True)
    a = torch.multinomial(p, 1, generator=gen).squeeze(-1)
    e_g.step(torch.where(e_g.done, torch.full_like(a, e_g.stop_idx), a))
print(f"2. random rollout ok: done {int(e_g.done.sum())}/{E} after {t} steps, steps where the gate was stricter than the grid: {stricter}", flush=True)
a = dict(model="mott", d=32, layers=2, heads=4, peak_refine=True, peak_gate=True, feat="mott_pkgate")
pol = train.make_policy(cfg, a)
env = PkGateEnv(b, ib, cfg, feat_mode="mott_pkgate", max_steps=40)
out = rollout(pol, env, greedy=True, amp=False)
print("3. greedy rollout ok: count", out["count"].tolist(), "feasible", out["feasible"].tolist(), flush=True)
open("runs/analysis/smoke_peak_gate.ok", "w").write("ok\n") if os.path.isdir("runs/analysis") else None
print("OK")
