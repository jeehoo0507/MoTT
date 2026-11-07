"""Smoke for `--model devformer` (B-DevFormer, baselines_ml/devformer.py, PROPOSALS_DRAFT_devformer.md), CPU, synthetic
boards (no pool, no exact check).
1. Forward shapes on coord features (logits [E,n,T+1], stop [E]); parameter count.
2. Permutation invariance of the encoder: shuffling the cells permutes the per-cell embeddings the same way
   (no positional order), and the policy logits likewise; the stop logit is unchanged.
3. AP augmentation: P = 4 permuted copies of a greedy+prune teacher trajectory, P distinct add orders different
   from the teacher's, the same decap set; every copy that replays without divergence ends in the teacher's solution.
4. distill_step (EE P = 4 + SE lambda = 5) gives finite gradients; with P = 0, lambda = 0 it equals the plain
   distill loss of train.py (same weights, same value).
5. Greedy rollout of the untrained policy runs end to end; train.py --stage distill --model devformer runs 3 iters;
   --stage rl with --model devformer is refused (the paper has no RL).
Run: python scripts/smoke_devformer.py
"""
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from baselines_ml.devformer import add_phase_len, distill_step, permute_adds, replay_ok  # noqa: E402
from decap_loop.baselines import greedy  # noqa: E402
from decap_loop.modal_env import make_env  # noqa: E402
from decap_loop.pdn import PDNConfig, sample_batch  # noqa: E402
from decap_loop.rollout import backward_logp, rollout  # noqa: E402
import train  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)
cfg = PDNConfig()
T = cfg.n_types
b = sample_batch(4, 7, 7, cfg, rng, n_probe=(2, 2))
ib = torch.arange(b.B)
env = make_env(b, ib, cfg, "coord", max_steps=40)
obs = env.observe()
x, valid = obs["x"], env.valid
a = dict(model="devformer", feat="coord", d=64, layers=4, heads=4)
pol = train.make_policy(cfg, a).eval()
n_par = sum(q.numel() for q in pol.parameters())
full = train.make_policy(cfg, dict(a, d=128, heads=8))
n_full = sum(q.numel() for q in full.parameters())
park = train.make_policy(cfg, dict(model="park_am", feat="coord", d=128, layers=4, heads=8, glob=False))
n_park = sum(q.numel() for q in park.parameters())
assert len(pol.layers) == 3, "df_layers default must be the paper's L = 3"
with torch.no_grad():
    lg, st = pol(x, valid, b.H, b.W)
E, n = x.shape[:2]
assert lg.shape == (E, n, T + 1) and st.shape == (E,), (lg.shape, st.shape)
assert torch.isfinite(lg).all() and torch.isfinite(st).all()
print(f"1. forward ok: x {tuple(x.shape)} -> logits {tuple(lg.shape)}, stop {tuple(st.shape)} | params: smoke "
      f"(d 64) {n_par / 1e6:.3f}M, run size (d 128, L 3, FF 512) {n_full / 1e6:.3f}M (B-Park d 128 L 4: "
      f"{n_park / 1e6:.3f}M)", flush=True)

# 2. permutation invariance (equivariance of per-cell outputs) -------------------------------------------------
g = torch.Generator().manual_seed(3)
pi = torch.randperm(n, generator=g)
with torch.no_grad():
    h0, *_ = pol.encode(x, valid)
    h1, *_ = pol.encode(x[:, pi], valid[:, pi])
    lg1, st1 = pol(x[:, pi], valid[:, pi], b.H, b.W)
vm = valid[..., None]
d_h = float(((h1 - h0[:, pi]) * vm[:, pi]).abs().max())
d_l = float(((lg1 - lg[:, pi]) * vm[:, pi]).abs().max())
d_s = float((st1 - st).abs().max())
assert d_h < 1e-4 and d_l < 1e-4 and d_s < 1e-4, (d_h, d_l, d_s)
print(f"2. permutation invariance: shuffled cells -> embeddings max |diff| {d_h:.1e}, logits {d_l:.1e}, "
      f"stop {d_s:.1e} (valid cells)", flush=True)

# 3. AP augmentation ----------------------------------------------------------------------------------------
tch = greedy(b, cfg, max_steps=40)
tb = torch.nonzero(tch["feasible"]).squeeze(-1)
assert tb.numel() >= 2, "need feasible teacher instances"
acts = tch["actions"][tb]
P = 4
perm, k = permute_adds(acts, b.n, T, P, np.random.default_rng(1))
assert perm.shape == (P, *acts.shape)
for e in range(acts.shape[0]):
    ke = int(k[e])
    orders = {tuple(perm[p, e, :ke].tolist()) for p in range(P)}
    if ke >= 3:   # 3! = 6 >= P + 1 orders exist
        assert len(orders) == P and tuple(acts[e, :ke].tolist()) not in orders, (ke, orders)
    for p in range(P):
        assert sorted(perm[p, e, :ke].tolist()) == sorted(acts[e, :ke].tolist())
        assert torch.equal(perm[p, e, ke:], acts[e, ke:])
n_ok, n_tot = 0, 0
for p in range(P):
    rep = rollout(pol, make_env(b, ib[tb], cfg, "coord", max_steps=40), forced=perm[p], store=True, amp=False)
    ok = replay_ok(rep, perm[p])
    same = (rep["occ"] == tch["occ"][tb]).all(1).cpu()
    assert bool(same[ok].all()), "a non-diverged permuted replay must end in the teacher's solution"
    n_ok, n_tot = n_ok + int(ok.sum()), n_tot + ok.numel()
print(f"3. AP augmentation: add-phase lengths k {k.tolist()}, P = {P} distinct permuted orders per instance "
      f"(k >= 3), same set and suffix; replay valid {n_ok}/{n_tot}, each valid one ends in the teacher solution",
      flush=True)

# 4. distill_step ------------------------------------------------------------------------------------------
args = SimpleNamespace(df_perm=4, df_self=5.0, max_steps=40, no_prune=False, feat="coord")
pol.train()
pol.zero_grad()
loss, stt = distill_step(pol, b, cfg, ib[tb], acts, args, False, np.random.default_rng(2), b.H, b.W)
gn = sum(float(q.grad.abs().sum()) for q in pol.parameters() if q.grad is not None)
assert np.isfinite(loss) and gn > 0 and np.isfinite(gn), (loss, gn)
print(f"4a. distill_step EE P=4 + SE lambda=5: loss {loss:.3f}, |grad| {gn:.3f}, stats {stt}", flush=True)
args0 = SimpleNamespace(df_perm=0, df_self=0.0, max_steps=40, no_prune=False, feat="coord")
pol.zero_grad()
l0, _ = distill_step(pol, b, cfg, ib[tb], acts, args0, False, np.random.default_rng(2), b.H, b.W)
g0 = [q.grad.clone() for q in pol.parameters()]
pol.zero_grad()
rep = rollout(pol, make_env(b, ib[tb], cfg, "coord", max_steps=40), forced=acts, store=True, amp=False)
ts = (rep["actions"] >= 0).sum(1).clamp(min=1).float()
l1 = backward_logp(pol, rep["traj"], 1.0 / ts, b.H, b.W, False, scale=1.0 / tb.numel())
dg = max(float((q.grad - g).abs().max()) for q, g in zip(pol.parameters(), g0))
assert abs(l0 - l1) < 1e-5 and dg < 1e-6, (l0, l1, dg)
print(f"4b. P=0, lambda=0 equals the plain distill step: loss {l0:.4f} vs {l1:.4f}, max |grad diff| {dg:.1e}", flush=True)

# 5. rollout + train.py integration -------------------------------------------------------------------------
pol.eval()
out = rollout(pol, make_env(b, ib, cfg, "coord", max_steps=40), greedy=True, amp=False)
print("5a. greedy rollout ok: count", out["count"].tolist(), "feasible", out["feasible"].tolist(), flush=True)
with tempfile.TemporaryDirectory() as tmp:
    cmd = [sys.executable, "train.py", "--stage", "distill", "--model", "devformer", "--df_perm", "4", "--df_self", "5",
           "--device", "cpu", "--no_amp", "--iters", "3", "--val_every", "3", "--B", "3", "--sizes", "6-6",
           "--val_sizes", "6", "--val_n", "3", "--n_probe", "1-2", "--max_steps", "40", "--d", "32", "--layers", "4",
           "--heads", "4", "--out", tmp]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200,
                       env=dict(os.environ, OMP_NUM_THREADS="1"))
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    assert r.returncode == 0 and os.path.exists(os.path.join(tmp, "metrics.json")), "\n".join(tail)
    log = open(os.path.join(tmp, "log.txt")).read()
    assert "model: devformer" in log and "feat: coord" in log, log[:500]
print("5b. train.py --stage distill --model devformer --df_perm 4 --df_self 5 (3 iters, CPU) ok:", tail[-1], flush=True)
with tempfile.TemporaryDirectory() as tmp:
    r = subprocess.run([sys.executable, "train.py", "--stage", "rl", "--model", "devformer", "--device", "cpu", "--out", tmp],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "distill-only" in r.stderr, (r.returncode, r.stderr[-300:])
print("5c. train.py --stage rl --model devformer is refused (distill-only):", r.stderr.strip().splitlines()[-1], flush=True)
open("runs/analysis/smoke_devformer.ok", "w").write("ok\n") if os.path.isdir("runs/analysis") else None
print("OK")
