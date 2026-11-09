"""B-Park: prior-work re-implementation (QUEUE 2026-10-29 03:30, A) — Park et al. 2022 (TMTT, HBM PDN) /
DevFormer-style attention-model (AM, Kool et al. 2019) pointer policy on coordinate input.

  encoder : cell tokens = rung-0 coordinate input (`--feat coord`: flags, placed type, global violation at the
            8 fixed feature frequencies, normalised (x, y) and offset to the first probe); linear embedding +
            L transformer layers (full self-attention over valid cells, no positional encoding beyond the coords)
  context : [graph embedding (mean over cells), mean embedding of the occupied cells] -> query (Kool's context
            embedding; the "last node" term is replaced by the placed-set mean, the state is a set, not a tour)
  decoder : one multi-head glimpse of the context over the cells, then a single-head pointer score
            C * tanh(q_t . k_i / sqrt(d)), C = 10 (Kool), one query per action channel t (add type t / remove),
            so the type choice uses our action space unchanged; stop logit from the glimpse
Training (train.py): same distill stage; rl with `--baseline rollout` (greedy rollout of a frozen copy, replaced
when validation improves; Kool's paired t-test replaced by the validation score) and `--imit 0` (plain
REINFORCE, no imitation / expert iteration). Differences from the papers are listed in RESULTS.md.
The encoder is re-run every step (the state changes the features), unlike Kool's encode-once.
"""
import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)   # LayerNorm, not Kool's BatchNorm: padded boards
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, h, valid):
        x = self.n1(h)
        x, _ = self.attn(x, x, x, key_padding_mask=~valid, need_weights=False)
        h = h + x
        return h + self.ff(self.n2(h))


class ParkAM(nn.Module):
    def __init__(self, in_dim, n_types, d=128, layers=4, heads=8, clip=10.0):
        super().__init__()
        self.T, self.clip, self.d = n_types, clip, d
        self.embed = nn.Linear(in_dim, d)
        self.layers = nn.ModuleList([EncoderLayer(d, heads) for _ in range(layers)])
        self.ctx = nn.Linear(2 * d, d)
        self.glimpse = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q = nn.Linear(d, (n_types + 1) * d)
        self.k = nn.Linear(d, d)
        self.stop_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.grad_ckpt = False

    def forward(self, x, valid, H, W):
        """x[E,n,D] coord features -> logits [E,n,T+1], stop [E] (masking by the env, as for every policy)."""
        E, n, _ = x.shape
        h = self.embed(x)
        for lyr in self.layers:
            h = lyr(h, valid)
        vm = valid[..., None].to(h.dtype)
        g = (h * vm).sum(1) / vm.sum(1).clamp(min=1)                                   # [E,d]
        T = self.T
        placed = x[..., 4:4 + T].sum(-1, keepdim=True).to(h.dtype) * vm                 # occupied one-hot block
        hp = (h * placed).sum(1) / placed.sum(1).clamp(min=1)
        c = self.ctx(torch.cat([g, hp], -1))[:, None]                                   # [E,1,d]
        gl, _ = self.glimpse(c, h, h, key_padding_mask=~valid, need_weights=False)      # [E,1,d]
        q = self.q(gl).view(E, T + 1, self.d)                                           # one query per channel
        k = self.k(h)                                                                   # [E,n,d]
        u = torch.einsum("etd,end->ent", q, k) / self.d ** 0.5
        logits = self.clip * torch.tanh(u)
        stop = self.stop_head(torch.cat([gl[:, 0], g], -1))[:, 0]
        return logits, stop
