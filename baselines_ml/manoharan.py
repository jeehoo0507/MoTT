"""B-Manoharan: prior-work re-implementation (QUEUE 2026-10-29 03:30, A) — Manoharan, Wang, Hwang 2026
(TSIPI; PhD thesis Paper I ch. 2-3), as described there:

  port order : fixed per board before the episode — candidate ports ranked by the loop inductance seen from the
               worst probe with port i shorted, at one frequency f0:  L_i = Im(Z_pp - Z_ip^2 / Z_ii)(f0) / (2 pi f0)
               (lowest first). f0 = the bare board's worst-violation frequency on its working grid (our choice:
               the paper's f0 is a fixed design frequency; ours varies per board kind).
  action     : for the current port only, a decap type or "empty" (skip); then the next port. Episode ends when the
               target is met (only stop is legal) or the order is exhausted. No remove / change-type.
  state      : delta-Z on the fixed feature-frequency grid (max-probe log10 |Z_pp| / Zt at the 8 feature
               frequencies + the rest of the global block) -> MLP, plus an embedding of the current port's L.
  decap      : tokens (ESL, ESR, C, log-normalised) + an "empty" token; one self-attention layer, then the state
               query scores the T + 1 tokens (attention-style pointer over types).
  update     : every placement is the env's rank-1 (Schur / Woodbury) update of the loaded network (sim.Env).
Training (train.py --model manoharan): our pipeline unchanged (distill 0.75 h on the teacher solutions replayed in
the fixed port order, then rl with the group baseline + imitation); the paper's own RL details differ (RESULTS.md).
Implemented as an Env subclass + policy here, so decap_loop/sim.py stays untouched.
"""
import numpy as np
import torch
import torch.nn as nn

from decap_loop.sim import EPS, FEAS_TOL, LOG_CLIP, Env, _probe_diag, violation


def glob_dim(cfg):
    return 2 * cfg.n_feat_freq + 3


def feat_dim(cfg):
    """flags 4, placed type T, global block G, current-port flag, log L, decap tokens 3T."""
    T = cfg.n_types
    return 4 + T + glob_dim(cfg) + 2 + 3 * T


class ManoharanEnv(Env):
    def __init__(self, batch, inst, cfg, occ0=None, max_steps=None, allow_prune=True, feat_mode="manoharan", resync=64):
        place = batch.placeable()[inst]
        # one step per port in the order (+ stop): inherent to the fixed-order traversal, so the caller's
        # max_steps (64 in training) is not used
        super().__init__(batch, inst, cfg, occ0=occ0, max_steps=int(place.sum(1).max()) + 2, allow_prune=False,
                         feat_mode="coord", resync=resync)
        self.feat_mode = "manoharan"
        self._order()

    def _order(self):
        E, n = self.E, self.n
        P = self.b.probes[self.inst]
        z_pp = _probe_diag(self.z_ip, P)                                          # [E,F,P]
        lv = torch.log10(z_pp.abs() + EPS) - self.logZt[..., None]
        worst, pk = lv.max(-1)                                                    # [E,F]
        f0 = worst.argmax(1)                                                      # [E]
        p0 = pk.gather(1, f0[:, None])[:, 0]
        ar = torch.arange(E, device=f0.device)
        zpp = z_pp[ar, f0, p0]                                                    # [E]
        zip_ = self.z_ip[ar, f0, :, p0]                                           # [E,n]
        zii = self.z_ii[ar, f0]                                                   # [E,n]
        w0 = 2 * np.pi * self.freqs_t[ar, f0]
        L = (zpp[:, None] - zip_ ** 2 / zii).imag / w0[:, None]                   # [E,n]
        L = torch.where(self.placeable, L, torch.full_like(L, float("inf")))
        self.order = L.argsort(1)                                                 # placeable cells first
        self.n_place = self.placeable.sum(1)
        Lp = torch.where(self.placeable, L.clamp(min=1e-15), torch.ones_like(L))
        med = torch.stack([Lp[e][self.placeable[e]].median() if self.placeable[e].any() else Lp.new_ones(())
                           for e in range(E)])
        self.logL = (torch.log10(Lp / med[:, None])).clamp(-LOG_CLIP, LOG_CLIP).float() * self.placeable
        self.ptr = torch.zeros(E, dtype=torch.long, device=f0.device)
        dec = [[np.log10(esl) + 11, np.log10(esr) + 1, np.log10(c) + 10] for c, esl, esr in self.b.cfg.decaps]
        self.dec_feat = torch.tensor(dec, dtype=torch.float32, device=f0.device).reshape(-1)   # [3T]

    def current(self):
        """[E] current port (cell index), -1 when the order is exhausted."""
        ex = self.ptr >= self.n_place
        cur = self.order.gather(1, self.ptr.clamp(max=self.n - 1)[:, None])[:, 0]
        return torch.where(ex, torch.full_like(cur, -1), cur)

    def observe(self, feats=True):
        z_pp = _probe_diag(self.z_ip, self.b.probes[self.inst])
        V, m = violation(z_pp, self.logZt)
        feas = m <= FEAS_TOL
        self.feas, self.V = feas, V
        E, n, T = self.E, self.n, self.T
        cur = self.current()
        go = ~feas & (cur >= 0)
        legal = torch.zeros(E, n, T + 1, dtype=torch.bool, device=self.occ.device)
        e = torch.nonzero(go).squeeze(-1)
        legal[e, cur[e]] = True                                                   # types 0..T-1 and skip (T)
        mask = torch.cat([legal.reshape(E, -1), (~go)[:, None]], 1)
        self.done |= (self.steps >= self.max_steps) | ~mask.any(1)
        obs = dict(mask=mask, V=V, m=m, feas=feas)
        if feats:
            obs["x"] = self.features(self.z_ip, self.z_ii, z_pp, V, None, None, feas)
        return obs

    def features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        self.feat_mode = "coord"
        base = Env.features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas)[..., :-4]   # flags, placed type, global
        self.feat_mode = "manoharan"
        E, n = self.E, self.n
        cur = self.current()
        onehot = torch.zeros(E, n, device=base.device)
        e = torch.nonzero(cur >= 0).squeeze(-1)
        onehot[e, cur[e]] = 1.0
        vf = self.valid.float()
        dec = self.dec_feat.expand(E, n, -1)
        return torch.cat([base, onehot[..., None], self.logL[..., None], dec * vf[..., None]], -1)

    def step(self, a):
        act = ~self.done
        stop = act & (a == self.stop_idx)
        super().step(a)        # skip = channel T on an empty cell: admittance 0 -> 0, a no-op rank-1 update
        self.ptr[act & ~stop] += 1


class ManoharanNet(nn.Module):
    def __init__(self, n_types, n_feat_freq, d=128, heads=8):
        super().__init__()
        self.T, self.G = n_types, 2 * n_feat_freq + 3
        self.mlp_z = nn.Sequential(nn.Linear(self.G, d), nn.ReLU(), nn.Linear(d, d))
        self.mlp_L = nn.Sequential(nn.Linear(1, d), nn.ReLU(), nn.Linear(d, d))
        self.embed_dec = nn.Linear(3, d)
        self.empty = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.tok_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.stop_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.d = d
        self.grad_ckpt = False

    def forward(self, x, valid, H, W):
        E, n, _ = x.shape
        T, G = self.T, self.G
        o = 4 + T
        flag = x[..., o + G]
        idx = flag.argmax(1)                                                      # current port (any cell if none)
        xc = x[torch.arange(E, device=x.device), idx]                             # [E,D]
        s = self.mlp_z(xc[:, o:o + G]) + self.mlp_L(xc[:, o + G + 1:o + G + 2])  # state
        dec = xc[:, o + G + 2:].view(E, T, 3)
        tok = torch.cat([self.embed_dec(dec), self.empty.expand(E, 1, self.d).to(s.dtype)], 1)   # [E,T+1,d]
        y, _ = self.tok_attn(tok, tok, tok, need_weights=False)
        tok = self.n1(tok + y)
        # mul-sum, not einsum/bmm: their CUDA backward here goes through triton, which can't build in this image
        # (no Python.h; bm_manoharan crashed 2026-10-29 09:19, scripts/diag_triton.py) — same value
        score = (self.q(s)[:, None] * self.k(tok)).sum(-1) / self.d ** 0.5                     # [E,T+1]
        logits = torch.zeros(E, n, T + 1, dtype=score.dtype, device=x.device)
        logits[torch.arange(E, device=x.device), idx] = score
        return logits, self.stop_head(s)[:, 0]


@torch.no_grad()
def manoharan_teacher(batch, cfg, inst, occ_t, max_steps=None):
    """Imitation target in the fixed port order: walk the order, place the teacher's type where the teacher solution
    has a decap, skip elsewhere, up to the last teacher decap (the env forces stop once the target is met).
    feasible / count come from the usual replay (baselines.trajectory_to) so the same instances are used."""
    from decap_loop.baselines import trajectory_to
    ref = trajectory_to(batch, cfg, inst, occ_t, max_steps=max_steps)
    env = ManoharanEnv(batch, inst, cfg)
    T = cfg.n_types
    o = env.order                                                                 # [E,n]
    typ = occ_t.gather(1, o)                                                      # teacher type along the order
    j = torch.arange(o.shape[1], device=o.device).expand_as(o)
    inside = j < env.n_place[:, None]
    last = torch.where((typ >= 0) & inside, j, torch.full_like(j, -1)).amax(1)    # [E]
    acts = o * (T + 1) + torch.where(typ >= 0, typ, torch.full_like(typ, T))
    acts = torch.where(j <= last[:, None], acts, torch.full_like(acts, -1))
    return dict(actions=acts, feasible=ref["feasible"], count=ref["count"])
