"""Smoke for `--model manoharan2` (paper-faithful B-Manoharan v2, baselines_ml/manoharan2.py), CPU, synthetic boards.
1. iterative port order is a permutation of the placeable cells (placeable first) and differs from the one-shot
   order (same f0) on at least one board; the cached order equals a fresh computation; brute-force check of the
   first two ranks against an explicit Z with ports shorted (solve, not rank-1).
2. the f0 rule gives a base-grid frequency inside the band (and inside the mid band when inductive).
3. with P > 1 IC ports, L is the mean over IC ports (checked against a per-probe recomputation).
4. an untrained policy's greedy rollout runs end to end (train.make_policy, make_env); one rl-style sampled rollout +
   backward_logp gives finite gradients; the paper reward (--reward manoharan) is finite.
5. the history self-attention is live: changing a PAST history row changes the logits, a FUTURE row does not.
6. parameter count; `--model manoharan` (v1) still builds and rolls out unchanged.
Run: python scripts/smoke_manoharan2.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from baselines_ml.manoharan2 import (ManoharanEnv2, board_order, f0_index, iterative_order, manoharan2_teacher,  # noqa: E402
                                     manoharan_reward, n_dz, oneshot_order, z0_at)
from decap_loop.modal_env import feat_dim_any, make_env  # noqa: E402
from decap_loop.pdn import PDNConfig, sample_batch  # noqa: E402
from decap_loop.rollout import backward_logp, rollout  # noqa: E402
from decap_loop.sim import _probe_diag  # noqa: E402
import train  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)
cfg = PDNConfig()
b = sample_batch(6, 8, 8, cfg, rng, n_probe=(2, 3))
ib = torch.arange(b.B)
F0 = cfg.n_freq
print(f"boards: {b.B} x {b.H}x{b.W}, P={b.P}, working grid F={b.freqs.shape[1]} (base {F0})", flush=True)

# 1. iterative order ------------------------------------------------------------------------------------------
res = board_order(b, ib.tolist())
place = b.placeable()
n_diff = 0
P = b.probes
zpp0 = _probe_diag(b.bank.cols(ib, P)[:, :F0], P)
fi, ok = f0_index(zpp0, cfg)
Z = torch.stack([z0_at(b.bank, i, int(fi[i])) for i in range(b.B)])
w0 = torch.from_numpy(2 * np.pi * b.freqs[np.arange(b.B), fi.numpy()])
order_fresh, Lsel = iterative_order(Z, P, place, w0)
one = oneshot_order(Z, P, place, w0)
for i, (order, L, f, _) in enumerate(res):
    npl = int(place[i].sum())
    first = order[:npl]
    assert sorted(first.tolist()) == sorted(torch.nonzero(place[i]).squeeze(-1).tolist()), "not a permutation of placeable"
    assert sorted(order.tolist()) == list(range(b.n)), "order is not a permutation of all cells"
    assert torch.equal(order, order_fresh[i]), "cache differs from fresh computation"
    diff = int((first != one[i, :npl]).sum())
    n_diff += diff > 0
    print(f"  board {i}: {npl} placeable, f0 idx {f} ({b.freqs[i, f] / 1e9:.3f} GHz), iterative vs one-shot: "
          f"{diff} positions differ; first 6 iterative {first[:6].tolist()} one-shot {one[i, :6].tolist()}")
assert n_diff >= 1, "iterative order equals the one-shot order on every board"

# brute force first two ranks on board 0 with explicit shorts (Z of the remaining network by removing shorted nodes)
i = 0
Zf = Z[i]
pl = torch.nonzero(place[i]).squeeze(-1).tolist()
pr = P[i].tolist()


def L_short(shorted):
    """mean over probes Im Z_pp with `shorted` nodes at 0 V: Z_pp - Z_pS Z_SS^-1 Z_Sp."""
    S = torch.tensor(shorted)
    zps = Zf[pr][:, S]
    z = torch.diagonal(Zf[pr][:, pr]) - torch.diagonal(zps @ torch.linalg.solve(Zf[S][:, S], zps.T))
    return float(z.imag.mean() / w0[i])


j1 = min(pl, key=lambda j: L_short([j]))
j2 = min([j for j in pl if j != j1], key=lambda j: L_short([j1, j]))
assert [j1, j2] == res[0][0][:2].tolist(), ([j1, j2], res[0][0][:2].tolist())
assert abs(L_short([j1, j2]) - float(res[0][1][j2])) < 1e-6 * abs(L_short([j1, j2])) + 1e-18
print(f"1. iterative order: permutation of placeable cells on all {b.B} boards, differs from one-shot on {n_diff}; "
      f"brute-force first two ranks {[j1, j2]} match", flush=True)

# 2. f0 rule ----------------------------------------------------------------------------------------------------
fz = b.freqs[np.arange(b.B), fi.numpy()]
u = (np.log(fz) - np.log(cfg.f_min)) / (np.log(cfg.f_max) - np.log(cfg.f_min))
assert ((fz >= cfg.f_min) & (fz <= cfg.f_max)).all()
assert ((u >= 0.25 - 1e-9) & (u <= 0.75 + 1e-9))[ok.numpy()].all()
ph = torch.rad2deg(torch.angle(zpp0[torch.arange(b.B), fi])).mean(-1)
print(f"2. f0 inside band [{cfg.f_min:.0e}, {cfg.f_max:.0e}]: {[f'{x / 1e9:.3f}' for x in fz]} GHz, "
      f"band position {np.round(u, 2).tolist()}, inductive rule used {ok.tolist()}, phase at f0 "
      f"{[round(float(x), 1) for x in ph]} deg", flush=True)

# 3. L averaged over IC ports ------------------------------------------------------------------------------------
assert b.P > 1
zpj = Z.gather(1, P[:, :, None].expand(-1, -1, b.n))
zpp = zpj.gather(2, P[:, :, None])[..., 0]
per_probe = (zpp[:, :, None] - zpj ** 2 / torch.diagonal(Z, dim1=1, dim2=2)[:, None, :]).imag / w0[:, None, None]
j0 = res[0][0][0]
Lmean = float(per_probe[0, :, j0].mean())
assert abs(Lmean - float(res[0][1][j0])) < 1e-9 * abs(Lmean), (Lmean, float(res[0][1][j0]))
print(f"3. P={b.P}: first-ranked L on board 0 = mean over IC ports {Lmean:.3e} H "
      f"(per port {[f'{x:.3e}' for x in per_probe[0, :, j0].tolist()]})", flush=True)

# 4. rollouts -----------------------------------------------------------------------------------------------------
a = dict(model="manoharan2", feat="manoharan2", d=128, heads=8, layers=4, glob=False)
torch.manual_seed(1)
pol = train.make_policy(cfg, a)
env = make_env(b, ib, cfg, "manoharan2")
assert isinstance(env, ManoharanEnv2)
x = env.observe()["x"]
assert x.shape[-1] == feat_dim_any(cfg, "manoharan2"), (x.shape, feat_dim_any(cfg, "manoharan2"))
pol.eval()
out = rollout(pol, make_env(b, ib, cfg, "manoharan2"), greedy=True, amp=False)
print(f"4. greedy rollout ok: count {out['count'].tolist()} feasible {out['feasible'].tolist()} "
      f"steps {(out['actions'] >= 0).sum(1).tolist()}", flush=True)
pol.train()
M = 4
ie = ib.repeat_interleave(M)
env = make_env(b, ie, cfg, "manoharan2", max_steps=64)
o = rollout(pol, env, store=True, amp=False)
R = manoharan_reward(b, ie, o["occ"], ~o["feasible"])
assert torch.isfinite(R).all()
Rg = R.view(b.B, M)
adv = ((Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-6)).view(-1)
pol.zero_grad()
backward_logp(pol, o["traj"], adv, b.H, b.W, amp=False, scale=1 / len(ie))
g = [q.grad for q in pol.parameters() if q.grad is not None]
assert g and all(torch.isfinite(t).all() for t in g)
tch = manoharan2_teacher(b, cfg, ib, out["occ"])
rep = rollout(pol, make_env(b, ib, cfg, "manoharan2"), forced=tch["actions"], amp=False)
assert torch.equal(rep["occ"], out["occ"]), "teacher replay along the iterative order does not reproduce occ"
print(f"   sampled rollout + backward_logp: grads finite on {len(g)} tensors; paper reward range "
      f"[{float(R.min()):.2f}, {float(R.max()):.2f}]; teacher replay reproduces the greedy solution", flush=True)

# 5. history attention --------------------------------------------------------------------------------------------
env = make_env(b, ib, cfg, "manoharan2")
for _ in range(3):                                   # 3 steps of "no decap" so that rows 0..2 are history
    obs = env.observe()
    cur = env.current()
    env.step(torch.where(cur >= 0, cur * (cfg.n_types + 1) + cfg.n_types, env.stop_idx))
obs = env.observe()
x = obs["x"].clone()
act = torch.nonzero((env.current() >= 0) & ~env.done).squeeze(-1)
assert act.numel() and bool((env.ptr[act] == 3).all())
pol.eval()
with torch.no_grad():
    l0, _ = pol(x, env.valid, b.H, b.W)
    xp = x.clone()
    xp[act, 0, :n_dz(cfg)] += 0.5                    # a PAST step's dZ
    lp, _ = pol(xp, env.valid, b.H, b.W)
    xf = x.clone()
    xf[act, 5, :n_dz(cfg)] += 0.5                    # a FUTURE row (not yet valid)
    lf, _ = pol(xf, env.valid, b.H, b.W)
dp, df = float((l0 - lp)[act].abs().max()), float((l0 - lf)[act].abs().max())
assert dp > 1e-6, "past history does not change the logits"
assert df == 0.0, "future rows leak into the logits"
print(f"5. history self-attention: past-row change moves logits by {dp:.2e}, future-row change by {df:.1e}", flush=True)

# 6. params, v1 untouched -------------------------------------------------------------------------------------------
n2 = sum(q.numel() for q in pol.parameters())
p1 = train.make_policy(cfg, dict(a, model="manoharan", feat="manoharan"))
n1 = sum(q.numel() for q in p1.parameters())
o1 = rollout(p1.eval(), make_env(b, ib, cfg, "manoharan"), greedy=True, amp=False)
print(f"6. params: manoharan2 {n2:,} ({n2 / 1e6:.3f}M); v1 manoharan {n1:,}; v1 greedy rollout ok "
      f"(count {o1['count'].tolist()})", flush=True)
open("runs/analysis/smoke_manoharan2.ok", "w").write("ok\n") if os.path.isdir("runs/analysis") else None
print("OK")
