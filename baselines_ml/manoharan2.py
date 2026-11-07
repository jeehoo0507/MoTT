"""B-Manoharan v2 (`--model manoharan2`): paper-faithful re-implementation of Manoharan, Wang, Hwang (PhD thesis
Paper I, "RL for decap optimization in post-layout PDNs using transformer architecture"). Page numbers below are the
PRINTED page numbers of the dissertation (PDF page = printed + 14). `--model manoharan` (baselines_ml/manoharan.py)
is left exactly as it was so the old runs stay reproducible; this file fixes the six audit contradictions:

 1. port order  ITERATIVE greedy (pp.14-15, eq.1): short the port with the lowest loop L seen by the IC port(s), keep it
                shorted, update Z (Schur complement), re-rank the rest, until every placeable port is ranked. Once per
                board (bare board; the order does not depend on the policy), cached per instance.
 2. f0          the paper only says "a chosen mid-band frequency, typically where inductive behavior dominates" (p.14).
                OUR CONCRETE CHOICE FOR AN UNSPECIFIED f0: on the bare board, among the base working-grid points in the
                middle half of the band in log f (25 %..75 %), the point whose IC-port impedance phase (mean over IC
                ports) is closest to +90 deg; if none of them is inductive (phase <= 0), the base-grid point nearest
                the geometric band centre. On our boards this lands on the inductive plateau just below the first
                plane anti-resonance (phase ~ +88 deg).
 3. several IC  L_j = mean over IC ports k of Im Z_{ik ik}(f0 | j shorted) / (2 pi f0) (eqs. 7-8, p.22); the delta-Z
    ports       deviation is aggregated over IC ports the same way (mean) (p.22: "aggregated", operator unspecified).
 4. decoder     one decoder layer (Fig. 1 p.13, pp.18-20): query q_t = e_port + e_dZ (eq.3); self-attention of q_t over
                the HISTORY of decoder inputs q_0..q_t of this episode (causal), cross-attention to the candidate decap
                tokens, then u_tc = (h W_Q).(e_c W_K) / sqrt(d) (eq.4), softmax with temperature tau (eq.5).
                The history lives in the env (one row per decoding step in the [E, n, D] feature tensor), so
                rollout.py / backward_logp keep their interface and the log-probabilities recomputed for the
                gradient see exactly the same history (the past queries are recomputed from the stored inputs).
 5. delta-Z     linear dZ = (|Z_IC| - Zt) / Zt on the cfg.n_freq = 32 base points of the working grid (Fig.1 p.13:
                Linear 201 -> 64 on 201 points; we have 32 base points, identical and sorted on every instance; the
                points appended by the exact check vary per instance and are not used). Normalised by Zt (our choice:
                raw ohms span 1 mOhm..1 Ohm targets), clipped to [-1, DZ_CLIP].
 6. decap token (C, ESL, ESR) log-normalised -> Linear(3->64) ReLU LN -> Linear(64->128) ReLU LN (pp.16-17, Fig.1); the
                paper's 4th feature (package size) does not exist on our boards. "no decap" is a learned row in the
                same 3-feature input space (Fig.1: library "including no decap").
Kept from v1: action = decap type or "no decap" at the current port, then the next port; the episode stops when the
target is met (only stop legal) or the order is exhausted; every placement is the env's exact rank-1 update.
`--reward manoharan` (train.py) plugs in the paper's terminal reward (eqs. 9-15, pp.25-26): manoharan_reward below.
Not reproduced (see PROPOSALS_DRAFT_bm2.md): critic + replay buffer (10k, mini-batch 64), discount 0.99, dropout,
package-size masking, 50k episodes / 24 h.
"""
import numpy as np
import torch
import torch.nn as nn

from baselines_ml.manoharan import ManoharanEnv
from decap_loop.sim import LOG_CLIP, _probe_diag, probe_impedance

DZ_CLIP = 100.0        # (|Z| - Zt) / Zt clip (bare boards reach ~10x the target; 100x leaves room)
MID_BAND = (0.25, 0.75)


def n_dz(cfg):
    return cfg.n_freq


def feat_dim(cfg):
    """history row: dZ (n_freq), log L (1), history-valid (1), current-row flag (1); decap tokens 3T."""
    return n_dz(cfg) + 3 + 3 * cfg.n_types


# ------------------------------------------------------------------ port order (pp.14-15, 22)

def f0_index(z_pp_bare, cfg):
    """z_pp_bare [E, F0, P] complex: bare-board IC-port impedance on the base grid. Returns f0 grid index [E] and
    whether the inductive rule applied [E] (False = geometric-centre fallback). See the module doc, item 2."""
    F0 = z_pp_bare.shape[1]
    u = np.linspace(0, 1, F0)
    mid = torch.from_numpy((u >= MID_BAND[0]) & (u <= MID_BAND[1])).to(z_pp_bare.device)
    ph = torch.rad2deg(torch.angle(z_pp_bare))                              # [E,F0,P]
    dev = (ph - 90.0).abs().mean(-1)                                        # [E,F0]
    ind = (ph.mean(-1) > 0) & mid[None]
    dev = torch.where(ind, dev, torch.full_like(dev, float("inf")))
    k = dev.argmin(1)
    ok = ind.any(1)
    centre = int(np.argmin(np.abs(u - 0.5)))
    return torch.where(ok, k, torch.full_like(k, centre)), ok


def z0_at(bank, b, fi):
    """Bare-board Z0(f) of instance b at working-grid index fi, all cells: [n, n] complex128 (read only)."""
    if bank.dense:
        return bank.Zc[b, fi][:, bank.slot[b]].to(torch.complex128)        # dense: slot = arange
    lu = bank.lus[b][fi]                                                   # sparse: n solves at one frequency
    return torch.from_numpy(lu.solve(np.eye(bank.n, dtype=np.complex128))).to(bank.device)


def iterative_order(Z, probes, placeable, w0):
    """Iterative greedy loop-inductance ranking (pp.14-15, eq.1, eqs.7-8), vectorised over E boards.
    Z [E,n,n] complex128 at f0, probes [E,P], placeable [E,n] bool, w0 [E] = 2 pi f0.
    Step k: for every unranked placeable j, L_j = mean_p Im(Z_pp - Z_pj^2 / Z_jj) / w0 (port j shorted, on top of the
    k already-ranked ports which stay shorted); pick the lowest, short it in Z by the Schur / rank-1 update
    Z <- Z - Z[:,j] Z[j,:] / Z_jj (the dy -> inf limit of sim.Env._rank1, coefficient dy/(1+dy Z_jj) -> 1/Z_jj).
    Returns order [E,n] (placeable cells first, in rank order; the rest after) and Lsel [E,n] (L of each cell at
    the moment it was ranked; +inf for non-placeable)."""
    Z = Z.clone()
    E, n, _ = Z.shape
    ar = torch.arange(E, device=Z.device)
    left = placeable.clone()
    order = torch.full((E, n), -1, dtype=torch.long, device=Z.device)
    Lsel = torch.full((E, n), float("inf"), dtype=torch.float64, device=Z.device)
    pidx = probes[:, :, None].expand(-1, -1, n)
    for k in range(int(placeable.sum(1).max())):
        zpj = Z.gather(1, pidx)                                                       # [E,P,n]
        zpp = zpj.gather(2, probes[:, :, None])[..., 0]                               # [E,P]
        zjj = torch.diagonal(Z, dim1=1, dim2=2)
        zjj_safe = torch.where(left, zjj, torch.ones_like(zjj))
        L = (zpp[:, :, None] - zpj ** 2 / zjj_safe[:, None, :]).imag.mean(1) / w0[:, None]   # [E,n]
        L = torch.where(left, L, torch.full_like(L, float("inf")))
        j = L.argmin(1)
        act = left.any(1)
        order[act, k] = j[act]
        Lsel[ar[act], j[act]] = L[ar[act], j[act]]
        left[ar[act], j[act]] = False
        col = Z[ar, :, j]                                                             # [E,n] (Z symmetric)
        c = torch.where(act, 1.0 / zjj[ar, j], torch.zeros_like(zjj[:, 0]))
        Z = Z - col[:, :, None] * col[:, None, :] * c[:, None, None]
    # non-placeable cells after the ranked ones (never visited: current() stops at n_place)
    rest = torch.where(order >= 0, order, torch.full_like(order, n))
    used = torch.zeros(E, n + 1, dtype=torch.bool, device=Z.device).scatter_(1, rest, True)[:, :n]
    tail = torch.argsort(used.to(torch.int8), dim=1, stable=True)                     # unused cells first
    n_pl = placeable.sum(1)
    pos = torch.arange(n, device=Z.device)[None].expand(E, n)
    order = torch.where(pos < n_pl[:, None], order, tail.gather(1, (pos - n_pl[:, None]).clamp(min=0)))
    return order, Lsel


def oneshot_order(Z, probes, placeable, w0):
    """v1-style one-shot ranking at the same f0 (bare board, every port shorted alone), for the smoke comparison."""
    E, n, _ = Z.shape
    zpj = Z.gather(1, probes[:, :, None].expand(-1, -1, n))
    zpp = zpj.gather(2, probes[:, :, None])[..., 0]
    L = (zpp[:, :, None] - zpj ** 2 / torch.diagonal(Z, dim1=1, dim2=2)[:, None, :]).imag.mean(1) / w0[:, None]
    return torch.where(placeable, L, torch.full_like(L, float("inf"))).argsort(1)


def board_order(batch, insts):
    """Per instance b of `batch` (bare board): (order [n], Lsel [n], f0 index, inductive rule used), cached on the
    batch object (base grid points never move when the exact check appends points, so the cache stays valid)."""
    cache = batch.__dict__.setdefault("_m2_order", {})
    need = sorted({int(b) for b in insts} - set(cache))
    cfg, F0, bank = batch.cfg, batch.cfg.n_freq, batch.bank
    for c0 in range(0, len(need), 16):
        bs = torch.tensor(need[c0:c0 + 16], dtype=torch.long, device=batch.probes.device)
        P = batch.probes[bs]
        zpp = _probe_diag(bank.cols(bs, P)[:, :F0], P)                              # [e,F0,P] bare board
        fi, ok = f0_index(zpp, cfg)
        Z = torch.stack([z0_at(bank, int(b), int(f)) for b, f in zip(bs, fi)])     # [e,n,n]
        w0 = torch.from_numpy(2 * np.pi * batch.freqs[bs.cpu().numpy(), fi.cpu().numpy()]).to(Z.device)
        order, Lsel = iterative_order(Z, P, batch.placeable()[bs], w0)
        for i, b in enumerate(bs.tolist()):
            cache[b] = (order[i], Lsel[i], int(fi[i]), bool(ok[i]))
    return [cache[int(b)] for b in insts]


# ------------------------------------------------------------------ env

class ManoharanEnv2(ManoharanEnv):
    """v1 env (fixed order, type-or-skip at the current port, stop when met) with the iterative order (1-3) and the
    per-step decoder-input history (4-5). Feature row r of an episode = decoder input of its step r (r <= current)."""

    def _order(self):
        E, n, dev = self.E, self.n, self.inst.device
        res = board_order(self.b, self.inst.tolist())
        self.order = torch.stack([r[0] for r in res]).to(dev)
        Lsel = torch.stack([r[1] for r in res]).to(dev)
        self.f0_idx = torch.tensor([r[2] for r in res], device=dev)
        self.f0_inductive = torch.tensor([r[3] for r in res], device=dev)
        self.n_place = self.placeable.sum(1)
        Lp = torch.where(self.placeable, Lsel.clamp(min=1e-15), torch.ones_like(Lsel))
        med = torch.stack([Lp[e][self.placeable[e]].median() if self.placeable[e].any() else Lp.new_ones(())
                           for e in range(E)])
        # "normalized" (p.15): log10(L / median over the board's ranked ports), as v1
        self.logL = (torch.log10(Lp / med[:, None])).clamp(-LOG_CLIP, LOG_CLIP).float() * self.placeable
        self.ptr = torch.zeros(E, dtype=torch.long, device=dev)
        dec = [[np.log10(c) + 10, np.log10(esl) + 11, np.log10(esr) + 1] for c, esl, esr in self.b.cfg.decaps]
        self.dec_feat = torch.tensor(dec, dtype=torch.float32, device=dev).reshape(-1)       # [3T]  (C, ESL, ESR)
        F0 = n_dz(self.b.cfg)
        self.hist = torch.zeros(E, n, F0 + 1, device=dev)                                 # dZ, log L per step
        self.hvalid = torch.zeros(E, n, dtype=torch.bool, device=dev)

    def features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        E, n = self.E, self.n
        F0 = n_dz(self.b.cfg)
        zt = 10 ** self.logZt[:, :F0]
        dz = ((z_pp[:, :F0].abs() - zt[..., None]) / zt[..., None]).mean(-1)               # [E,F0] mean over IC
        dz = dz.clamp(-1.0, DZ_CLIP).float()
        cur = self.current()
        e = torch.nonzero((cur >= 0) & ~self.done & (self.ptr < n)).squeeze(-1)
        r = self.ptr[e]
        self.hist[e, r] = torch.cat([dz[e], self.logL[e, cur[e]][:, None]], 1)
        self.hvalid[e, r] = True
        is_cur = torch.zeros(E, n, device=dz.device)
        is_cur[e, r] = 1.0
        dec = self.dec_feat.expand(E, n, -1)
        return torch.cat([self.hist, self.hvalid.float()[..., None], is_cur[..., None], dec], -1)


# ------------------------------------------------------------------ policy (Fig. 1, p.13)

def _enc(i, d, h=64):
    """Fig. 1 encoder: Linear(i->64) ReLU LayerNorm, Linear(64->d) ReLU LayerNorm."""
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, d), nn.ReLU(), nn.LayerNorm(d))


class ManoharanNet2(nn.Module):
    def __init__(self, n_types, n_freq, d=128, heads=8, tau=1.0):
        super().__init__()
        self.T, self.F0, self.d, self.tau = n_types, n_freq, d, tau
        self.port_enc = _enc(1, d)                     # loop inductance -> 128 (p.15, Fig.1)
        self.dz_enc = _enc(n_freq, d)                  # dZ (n_f points) -> 128 (p.18, Fig.1)
        self.cand_enc = _enc(3, d)                     # (C, ESL, ESR) -> 128 (pp.16-17, Fig.1)
        self.no_decap = nn.Parameter(torch.zeros(3))   # "no decap" row of the library (Fig.1), learned
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)   # W^Q, W^K of eq.4
        self.grad_ckpt = False

    def forward(self, x, valid, H, W):
        E, n, _ = x.shape
        T, F0 = self.T, self.F0
        ar = torch.arange(E, device=x.device)
        dz, logL = x[..., :F0], x[..., F0:F0 + 1]
        hv, cur = x[..., F0 + 1] > 0.5, x[..., F0 + 2]
        idx = cur.argmax(1)                                                           # current step row
        q_all = self.port_enc(logL) + self.dz_enc(dz)                                 # eq.3 for every step [E,n,d]
        q = q_all[ar, idx][:, None]                                                   # [E,1,d]
        kpm = ~hv
        kpm[ar, idx] = False                                                          # the query itself (causal)
        h, _ = self.self_attn(q, q_all, q_all, key_padding_mask=kpm, need_weights=False)
        h = self.n1(q + h)
        dec = torch.cat([x[:, 0, F0 + 3:F0 + 3 + 3 * T].reshape(E, T, 3),
                         self.no_decap.to(x.dtype).expand(E, 1, 3)], 1)              # [E,T+1,3]
        tok = self.cand_enc(dec)                                                      # [E,T+1,d]
        c, _ = self.cross_attn(h, tok, tok, need_weights=False)
        h = self.n2(h + c)[:, 0]                                                      # [E,d]
        # eq.4-5; mul-sum, not einsum/bmm (CUDA backward of those pulls in triton in our image, v1 note)
        score = (self.wq(h)[:, None] * self.wk(tok)).sum(-1) / self.d ** 0.5 / self.tau             # [E,T+1]
        logits = torch.zeros(E, n, T + 1, dtype=score.dtype, device=x.device)
        logits[ar, idx] = score
        # stop is legal only when nothing else is (target met / order exhausted): its logit is irrelevant
        return logits, torch.zeros(E, dtype=score.dtype, device=x.device)


# ------------------------------------------------------------------ teacher replay, reward

@torch.no_grad()
def manoharan2_teacher(batch, cfg, inst, occ_t, max_steps=None):
    """As baselines_ml.manoharan.manoharan_teacher, along the iterative order of ManoharanEnv2."""
    from decap_loop.baselines import trajectory_to
    ref = trajectory_to(batch, cfg, inst, occ_t, max_steps=max_steps)
    env = ManoharanEnv2(batch, inst, cfg)
    T = cfg.n_types
    o = env.order
    typ = occ_t.gather(1, o)
    j = torch.arange(o.shape[1], device=o.device).expand_as(o)
    inside = j < env.n_place[:, None]
    last = torch.where((typ >= 0) & inside, j, torch.full_like(j, -1)).amax(1)
    acts = o * (T + 1) + torch.where(typ >= 0, typ, torch.full_like(typ, T))
    acts = torch.where(j <= last[:, None], acts, torch.full_like(acts, -1))
    return dict(actions=acts, feasible=ref["feasible"], count=ref["count"])


@torch.no_grad()
def manoharan_reward(batch, inst, occ, fail, eps=1e-9):
    """Paper terminal reward (eqs. 9-15, pp.25-26), per episode [E] (float32):
       success: R = N_ports - N_caps * d ;  fail: R = -N_caps * d - sum_f max(|Z(f)| - Zt(f), 0)^2 ;  R / N_ports,
       d = 1 / (mean_f Zt + eps) (eq.12), N_ports = placeable ports of the board. |Z| on the working grid, summed over
       IC ports as well (the paper's single-IC sum over f). `fail` = our failure (working grid or exact check)."""
    zd = batch.zd(inst.device)[inst]                                                    # [E,T,F]
    zpp = probe_impedance(batch, inst, occ, zd).abs()                                  # [E,F,P]
    zt = batch.Zt[inst]                                                                 # [E,F] ohm
    viol = (zpp - zt[..., None]).clamp(min=0).pow(2).sum((1, 2))
    d = 1.0 / (zt.mean(1) + eps)
    npt = batch.placeable()[inst].sum(1).double()
    ncap = (occ >= 0).sum(1).double()
    R = torch.where(fail, -ncap * d - viol, npt - ncap * d)
    return (R / npt).float()
