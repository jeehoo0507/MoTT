"""Rung 3: looped MoTT. One MoTT block, weights shared across R iterations, with input injection
(the token embeddings are added to the state before every iteration: h_{r+1} = Block(h_r + h_0),
h_0 = 0; Universal Transformer / Looped Transformer style). R is sampled per training iteration
(train.py calls sample_R once per iteration so the rollout and the log-prob recomputation use the
same policy) and set explicitly at test time (set_eval_R: R 2 -> 16 extrapolation).

Everything else (tokens, edge bias, heads, physics prior) is the rung-2 MoTT of model.py.
"""
import numpy as np
import torch

from .model import MoTT


class LoopMoTT(MoTT):
    def __init__(self, n_types, k_modes, dims, d=64, heads=4, r_train=(2, 8), r_eval=8, fc=False, mode_head=False):
        super().__init__(n_types, k_modes, dims, d=d, layers=1, heads=heads, fc=fc, mode_head=mode_head)
        self.r_train = (int(r_train[0]), int(r_train[1]))
        self.R_cur = self.r_train[1]
        self.R_eval = int(r_eval)

    def sample_R(self, rng=None):
        lo, hi = self.r_train
        self.R_cur = int(rng.integers(lo, hi + 1)) if rng is not None else int(np.random.randint(lo, hi + 1))
        return self.R_cur

    def set_eval_R(self, R):
        self.R_eval = int(R)

    def forward(self, x, valid, H, W):
        E, n, _ = x.shape
        T, K = self.T, self.K
        o = 0
        node = x[..., o:o + self.dn]; o += self.dn
        edge = x[..., o:o + K * self.de].view(E, n, K, self.de); o += K * self.de
        mode = x[:, 0, o:o + K * self.dm].view(E, K, self.dm); o += K * self.dm
        dec = x[:, 0, o:o + T * self.dc].view(E, T, self.dc)
        mmask = mode[..., 3] > 0.5
        mmask = mmask | ~mmask.any(1, keepdim=True)
        u0, m0, c0 = self.embed_u(node), self.embed_m(mode), self.embed_c(dec)
        bias = self.edge_bias(edge).permute(0, 3, 1, 2)
        ev = self.edge_val(edge)
        fcb = self.fc_bias(mode, dec)
        blk = self.blocks[0]
        R = self.R_cur if self.training else self.R_eval
        m, u, c = torch.zeros_like(m0), torch.zeros_like(u0), torch.zeros_like(c0)
        for _ in range(R):
            m, u, c = blk(m + m0, u + u0, c + c0, bias, ev, mmask, valid, fcb)
        uu = u[:, :, None, :].expand(E, n, T, u.shape[-1])
        cc = c[:, None, :, :].expand(E, n, T, c.shape[-1])
        add = self.add_head(torch.cat([uu, cc, uu * cc], -1))[..., 0]
        add = add + self.mode_add(u, m, c, bias, mmask)
        rm = self.rm_head(u)
        nb = 4 + T + 3
        dV, mn = node[..., nb:nb + T + 1], node[..., nb + T + 1:nb + 2 * (T + 1)]
        prior = (self.w_dv * dV - self.w_mn * mn) if getattr(self, "use_prior", True) else 0.0
        logits = torch.cat([add, rm], -1) + prior
        vm = valid[..., None].to(u.dtype)
        g = (u * vm).sum(1) / vm.sum(1).clamp(min=1)
        mg = (m * mmask[..., None]).sum(1) / mmask.sum(1, keepdim=True).clamp(min=1)
        stop = self.stop_head(torch.cat([g, mg], -1))[:, 0]
        return logits, stop
