"""B-DevFormer: prior-work re-implementation of DevFormer (Kim et al., ICML 2023, arXiv 2205.13225v3) moved onto
our problem (target impedance, minimum count, 3 decap types, add / remove / change-type / stop, three board kinds).
Proposal and the faithful-vs-adapted table: PROPOSALS_DRAFT_devformer.md. `--model devformer` in train.py.

Model (paper Sec. 3.1, p. 3-4, Fig. 3):
  input   : per cell x_i = (x, y, c) with c one-hot {0 decap allowed, 1 keep-out, 2 probe} (App. A.3, p. 14);
            added for our env: power-entry flag and the placed decap type (one-hot T; the state of our
            add / remove actions). Taken from the `--feat coord` vector (sim.py), the offset-to-first-probe
            channels are dropped (the PPE replaces them). No positional encoding: x, y are input values only.
  PPE     : h0 = phi_node(x_i) + phi_PPE(||x_i - x_probe||) (p. 4); with several probes the distance to the
            nearest probe.
  encoder : L = 3 layers MHA + FF (128 / 512), post-norm residual as Kool / DevFormer, LayerNorm instead of
            BatchNorm (padded boards of mixed sizes, as B-Park). Permutation-equivariant over cells.
  decoder : Q = Linear(PCN(h_probe) + RCN(h_ctx) + SCN(g)) (p. 4):
              PCN = MLP of the probe embedding (mean over probe cells),
              RCN = MLP of the mean embedding of the placed cells (paper: embedding of the last action a_{t-1};
                    our policy interface is Markov in the state, which is a set),
              SCN = linear of the global state vector of the coord features (violation at the 8 feature
                    frequencies, log f / f1, target met, steps left, fill) -- added: the paper has a fixed
                    board and no target, ours varies per instance.
            one MHA glimpse of Q over the cells, then the B-Park pointer: one query per action channel
            (add type t / remove), logits C * tanh(q.k / sqrt d), C = 10; stop logit from the glimpse.
Training (train.py --stage distill only, no RL, as the paper -- offline imitation):
  --df_perm P   expert exploitation (EE, eq. 3, p. 5): the teacher trajectory plus P random permutations of its
                add phase (the decaps placed before the target is met; the remove / stop suffix is the same state
                sequence, so it is kept). A permutation whose prefix already meets the target diverges from our
                env rules (the next add is illegal) and is dropped (weight 0); the drop rate is logged.
  --df_self L   self-exploitation (SE, eq. 4-5, p. 5): sample a' from the current policy (no grad = the copy
                pi_theta~), permute its add phase, and penalise |log pi_theta(t(a')) - log pi_theta~(a')| / k over
                the add-phase steps (k = their number). The paper takes the L1 of sequence probabilities and
                multiplies lambda by 1e32 (p. 24) to bring lambda * L_self to 0.1-1; with our longer sequences
                that scale is not stable, so the same order-bias term (zero iff the policy is AP-symmetric on the
                sample, Theorem 1) is taken in log space. lambda * L_self is logged to check the 0.1-1 range.
"""
import numpy as np
import torch
import torch.nn as nn

from decap_loop.modal_env import make_env
from decap_loop.rollout import _logits, amp_ctx, backward_logp, rollout  # noqa: F401 (amp_ctx: re-export)


class EncoderLayer(nn.Module):
    """h <- Norm(h + MHA(h)); h <- Norm(h + FF(h)) (paper p. 4 eq. for h_{l+1}; Kool et al. 2019)."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, h, valid):
        a, _ = self.attn(h, h, h, key_padding_mask=~valid, need_weights=False)
        h = self.n1(h + a)
        return self.n2(h + self.ff(h))


def mlp(d):
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))


class DevFormer(nn.Module):
    def __init__(self, in_dim, n_types, d=128, layers=3, heads=8, ff=512, clip=10.0):
        super().__init__()
        T = n_types
        self.T, self.clip, self.d = T, clip, d
        self.g_dim = in_dim - 8 - T                  # coord layout: 4 flags | T occ | glob | x, y, dx, dy
        assert self.g_dim > 0, in_dim
        self.node = nn.Linear(2 + 3 + 1 + T, d)      # (x, y) + c one-hot (allowed / keep-out / probe) + entry + type
        self.ppe = nn.Linear(1, d)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ff) for _ in range(layers)])
        self.pcn, self.rcn = mlp(d), mlp(d)
        self.scn = nn.Linear(self.g_dim, d)
        self.qlin = nn.Linear(d, d)
        self.glimpse = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q = nn.Linear(d, (T + 1) * d)
        self.k = nn.Linear(d, d)
        self.stop_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.grad_ckpt = False

    def split(self, x):
        """coord features -> node input [E,n,6+T], probe mask [E,n], global state [E,g_dim], positions [E,n,2]."""
        T = self.T
        valid, keep, probe, entry = (x[..., i] for i in range(4))
        allowed = valid * (1 - keep) * (1 - probe)
        occ = x[..., 4:4 + T]
        pos = x[..., -4:-2]                                                    # normalised (x, y)
        node = torch.cat([pos, allowed[..., None], keep[..., None], probe[..., None], entry[..., None], occ], -1)
        vf = valid[..., None]
        g = (x[:, :, 4 + T:4 + T + self.g_dim] * vf).sum(1) / vf.sum(1).clamp(min=1)   # same on every valid cell
        return node, probe > 0.5, g, pos, occ

    def encode(self, x, valid):
        node, pm, g, pos, occ = self.split(x)
        dist = torch.cdist(pos.float(), pos.float())                          # [E,n,n]
        dist = dist.masked_fill(~pm[:, None, :], float("inf")).amin(-1)     # to the nearest probe
        dist = torch.where(torch.isfinite(dist), dist, torch.zeros_like(dist))
        h = self.node(node) + self.ppe(dist[..., None].to(node.dtype))
        for lyr in self.layers:
            h = lyr(h, valid)
        return h, pm, g, occ

    def forward(self, x, valid, H, W):
        """x[E,n,D] coord features -> logits [E,n,T+1], stop [E] (masking by the env, as for every policy)."""
        E, n, _ = x.shape
        h, pm, g, occ = self.encode(x, valid)
        vm = valid[..., None].to(h.dtype)
        pmf = pm[..., None].to(h.dtype) * vm
        hp = (h * pmf).sum(1) / pmf.sum(1).clamp(min=1)                      # h_probe
        placed = occ.sum(-1, keepdim=True).to(h.dtype) * vm
        hc = (h * placed).sum(1) / placed.sum(1).clamp(min=1)                # placed-set embedding
        Q = self.qlin(self.pcn(hp) + self.rcn(hc) + self.scn(g.to(h.dtype)))[:, None]   # [E,1,d]
        gl, _ = self.glimpse(Q, h, h, key_padding_mask=~valid, need_weights=False)
        q = self.q(gl).view(E, self.T + 1, self.d)
        k = self.k(h)
        u = torch.einsum("etd,end->ent", q, k) / self.d ** 0.5
        logits = self.clip * torch.tanh(u)
        stop = self.stop_head(torch.cat([gl[:, 0], Q[:, 0]], -1))[:, 0]
        return logits, stop


# ---------------------------------------------------------------- action-permutation (AP) transformation
def add_phase_len(actions, n, T):
    """actions [E,L] (flat cell * (T+1) + channel, stop = n * (T+1), -1 pad) -> k [E]: length of the leading add
    phase (adds on cells not touched before; ends at the first stop, remove, type change or pad)."""
    a = actions.cpu().numpy()
    stop = n * (T + 1)
    k = np.zeros(a.shape[0], dtype=np.int64)
    for e in range(a.shape[0]):
        seen = set()
        for t in range(a.shape[1]):
            v = int(a[e, t])
            if v < 0 or v == stop or v % (T + 1) == T or v // (T + 1) in seen:
                break
            seen.add(v // (T + 1))
            k[e] = t + 1
    return torch.from_numpy(k)


def permute_adds(actions, n, T, P, rng):
    """P AP-transformed copies [P,E,L] of actions [E,L]: the add phase of each row is reordered by a random
    permutation, distinct from the identity and from each other while k! allows (else repeats). The rest of the
    row is unchanged. Returns (copies, k)."""
    k = add_phase_len(actions, n, T)
    out = actions.unsqueeze(0).repeat(P, 1, 1).clone()
    for e in range(actions.shape[0]):
        ke = int(k[e])
        if ke < 2:
            continue
        n_perm = 1
        for i in range(2, ke + 1):
            n_perm *= i
            if n_perm > P + 1:
                break
        seen = {tuple(range(ke))}
        for p in range(P):
            for _ in range(64):
                perm = tuple(int(i) for i in rng.permutation(ke))
                if perm not in seen or len(seen) >= n_perm:
                    break
            seen.add(perm)
            out[p, e, :ke] = actions[e, :ke][list(perm)]
    return out, k


def replay_ok(rep, forced):
    """True where a forced replay took every forced action (no divergence from the env rules)."""
    want = (forced >= 0).sum(1)
    return (rep["actions"] >= 0).sum(1).cpu() == want.cpu()


@torch.no_grad()
def traj_logp(policy, traj, E, H, W, amp, keep_step=None, chunk_cells=1 << 16):
    """Per-episode sum of log pi(a_t | s_t) over a stored trajectory [E]; keep_step(t, ep) -> bool mask of the
    steps to include (default all)."""
    out = torch.zeros(E, dtype=torch.float32)
    for t, s in enumerate(traj):
        m = keep_step(t, s["ep"]) if keep_step is not None else torch.ones_like(s["ep"], dtype=torch.bool)
        if not bool(m.any()):
            continue
        x, valid, mask, a, ep = s["x"][m], s["valid"][m], s["mask"][m], s["a"][m], s["ep"][m]
        step = max(1, chunk_cells // max(1, x.shape[1]))
        for i in range(0, x.shape[0], step):
            lg = _logits(policy, x[i:i + step], valid[i:i + step], H, W, mask[i:i + step], amp)
            lp = torch.log_softmax(lg, -1).gather(1, a[i:i + step, None]).squeeze(1)
            out.index_add_(0, ep[i:i + step].cpu(), lp.float().cpu())
    return out


def filter_traj(traj, k):
    """Keep only the add-phase steps t < k[ep] of a stored trajectory."""
    kd = k.to(traj[0]["ep"].device) if traj else k
    res = []
    for t, s in enumerate(traj):
        m = t < kd[s["ep"]]
        if bool(m.any()):
            res.append({key: v[m] for key, v in s.items()})
    return res


def distill_step(policy, batch, cfg, inst, teach_actions, args, amp, rng, H, W):
    """One DevFormer CSE distill update (gradients accumulated, no optimiser step). inst [Et] instances with a
    feasible teacher, teach_actions [Et,L]. Returns (loss value, stats dict)."""
    E, n, T = inst.shape[0], batch.n, cfg.n_types
    P, lam = int(args.df_perm), float(args.df_self)
    kw = dict(max_steps=args.max_steps, allow_prune=not args.no_prune)
    copies = [teach_actions]
    if P > 0:
        perm, _ = permute_adds(teach_actions, n, T, P, rng)
        copies += [perm[p] for p in range(P)]
    scale = 1.0 / (E * len(copies))     # mean over D_aug (dropped copies count as zero: fixed scale per step)
    loss, n_ok, n_aug = 0.0, 0, 0
    for c, forced in enumerate(copies):   # one replay per copy: the same GPU memory as the plain distill step
        rep = rollout(policy, make_env(batch, inst, cfg, args.feat, **kw), forced=forced, store=True, amp=amp)
        steps = (rep["actions"] >= 0).sum(1).clamp(min=1).float()
        wt = 1.0 / steps                  # copy 0 = the teacher order: exactly the plain distill weight
        if c > 0:
            ok = replay_ok(rep, forced).to(steps.device)
            n_ok, n_aug = n_ok + int(ok.sum()), n_aug + E
            wt = wt * ok.float()
        loss += backward_logp(policy, rep["traj"], wt, H, W, amp, scale=scale)
    st = dict(aug_ok=n_ok / max(1, n_aug), self_loss=float("nan"), self_ok=float("nan"))
    if lam > 0:
        # a' ~ pi_theta~ (the current weights without grad), t(a') one random AP transformation of its add phase
        smp = rollout(policy, make_env(batch, inst, cfg, args.feat, **kw), greedy=False, store=True, amp=amp)
        a1 = smp["actions"]
        perm, k = permute_adds(a1, n, T, 1, rng)
        tr = perm[0]
        rep = rollout(policy, make_env(batch, inst, cfg, args.feat, **kw), forced=tr, store=True, amp=amp)
        ok = replay_ok(rep, tr) & (k >= 2)
        kd = k.to(inst.device)
        def keep(t, ep):   # noqa: E306
            return t < kd[ep]
        lp_ref = traj_logp(policy, smp["traj"], E, H, W, amp, keep)          # log pi_theta~(a') (constant)
        lp_new = traj_logp(policy, rep["traj"], E, H, W, amp, keep)          # log pi_theta(t(a')), value only
        d = (lp_new - lp_ref)
        kf = k.clamp(min=1).float()
        val = (d.abs() / kf)[ok]
        st["self_loss"] = lam * float(val.mean()) if ok.any() else 0.0
        st["self_ok"] = float(ok.float().mean())
        if ok.any():
            # d/dtheta lam*|lp_new - c|/k = lam*sign(d)/k * dlp_new; backward_logp minimises -scale*sum w*lp
            w = torch.where(ok, -lam * torch.sign(d) / kf, torch.zeros_like(d)).to(inst.device)
            backward_logp(policy, filter_traj(rep["traj"], kd), w, H, W, amp, scale=1.0 / int(ok.sum()))
        loss += st["self_loss"]           # reported loss = L_Expert + lambda * L_self (values)
    return loss, st
