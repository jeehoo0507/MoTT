"""Rung-1 input: `--feat modal` = rung-0 coordinate input + resonance-participation port features
+ exact one-step effects. Implemented as an Env subclass so sim.py stays untouched.

Per cell (all dimensionless log ratios; nothing tied to board size or frequency band):
  [coord part, identical to Env.features(feat_mode="coord")]
    flags (valid, keepout, probe, entry), placed type one-hot (T), global vector
    (violation at the 8 fixed feature frequencies, log f/f1, feasible, steps left, fraction placed),
    normalised (x, y) and offset to the first probe
  [modal part] for each of the K_PORT highest peaks of max_probe |Z_pp(f)| / Zt(f) on the sorted
  working grid (worst mode first), at the peak frequency f_k and worst probe p_k:
    mode descriptors (same on every cell): log10(f_k / f_c), peak level log10|Z_pp|/Zt, valid
    participation  log10(|Z_ii(f_k)| / |Z_pp(f_k)|)   (ratio of the residues r_ii / r_pp of a
                                                      single dominant mode; see 결과.md "참여비 v2")
    coupling       log10 g,  g = |Z_ip| / sqrt(|Z_pp| |Z_ii|)
    transfer       log10(|Z_ip(f_k)| / Zt(f_k))
    after-add      log10(|Z_pp'(f_k)| / Zt(f_k)) for each decap type t (rank-1 formula)
  [one-step part, last 2(T+1) channels so GridPolicy(prior=True) finds them]
    V - V' and worst margin after every action (add type t / remove), exact on the working grid

`--feat modal_only` drops the coordinate block (ablation: electrical features alone).
"""
import numpy as np
import torch

from .pdn import PDNConfig
from .sim import EPS, FEAS_TOL, LOG_CLIP, Env, _probe_diag, feat_dim

K_PORT = 3   # modes used for the port features


def modal_dims(T):
    """(mode descriptor size, per-cell per-mode size)."""
    return 3, 3 + T


def feat_dim_any(cfg: PDNConfig, mode: str):
    T = cfg.n_types
    if mode in ("modal", "modal_only"):
        dm, de = modal_dims(T)
        base = feat_dim(cfg, "coord") if mode == "modal" else 4 + T + (2 * cfg.n_feat_freq + 3)
        return base + K_PORT * (dm + de) + 2 * (T + 1)
    if mode == "manoharan":   # prior-work baseline B-Manoharan (baselines_ml/manoharan.py)
        from baselines_ml.manoharan import feat_dim as mfd
        return mfd(cfg)
    return feat_dim(cfg, mode)


def make_env(batch, inst, cfg, feat_mode, **kw):
    if feat_mode in ("modal", "modal_only"):
        return ModalEnv(batch, inst, cfg, feat_mode=feat_mode, **kw)
    if feat_mode == "mott_pk":   # MoTT + parabolic peak refinement per mode token (decap_loop/peak_env.py)
        from .peak_env import PkEnv
        return PkEnv(batch, inst, cfg, feat_mode=feat_mode, **kw)
    if feat_mode == "mott_pkgate":   # P1 refined peak also gates target-met (decap_loop/pkgate_env.py, 11/07 C1')
        from .pkgate_env import PkGateEnv
        return PkGateEnv(batch, inst, cfg, feat_mode=feat_mode, **kw)
    if feat_mode == "mott_pkdd":   # P1 channels on the P2 token set (decap_loop/pkdd_env.py)
        from .pkdd_env import PkDdEnv
        return PkDdEnv(batch, inst, cfg, feat_mode=feat_mode, **kw)
    if feat_mode == "mott_dd":   # MoTT with mode tokens on distinct working-grid frequencies (decap_loop/dedup_env.py)
        from .dedup_env import DedupEnv
        return DedupEnv(batch, inst, cfg, feat_mode=feat_mode, **kw)
    if feat_mode == "manoharan":
        from baselines_ml.manoharan import ManoharanEnv
        return ManoharanEnv(batch, inst, cfg, **kw)
    return Env(batch, inst, cfg, feat_mode=feat_mode, **kw)


class ModalEnv(Env):
    def features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        mode = self.feat_mode
        self.feat_mode = "coord"
        base = Env.features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas)       # [E,n,Dc]
        self.feat_mode = mode
        E, n, T = self.E, self.n, self.T
        if mode == "modal_only":                                            # drop the 4 coordinate channels
            base = base[..., :-4]
        vf = self.valid.float()
        modal = self._modal_block(z_ip, z_ii, z_pp)                         # [E,n,K*(dm+de)]
        dV = (V[:, None, None] - Vn).float().clamp(-LOG_CLIP, LOG_CLIP)
        one = torch.cat([dV, mn.clamp(-3, 3).float()], -1)                 # [E,n,2(T+1)]
        return torch.cat([base, modal * vf[..., None], one * vf[..., None]], -1)

    def _modal_block(self, z_ip, z_ii, z_pp):
        E, n, T = self.E, self.n, self.T
        K = min(K_PORT, z_pp.shape[1])
        P = z_pp.shape[2]
        lvp = torch.log10(z_pp.abs() + EPS) - self.logZt[..., None]        # [E,F,P]
        lv, pk = lvp.max(-1)                                                # worst probe per frequency
        lvs = lv.gather(1, self.forder)                                     # sorted by frequency
        ninf = torch.full_like(lvs[:, :1], -float("inf"))
        peak = (lvs >= torch.cat([ninf, lvs[:, :-1]], 1)) & (lvs >= torch.cat([lvs[:, 1:], ninf], 1))
        top, j = torch.where(peak, lvs, -float("inf")).topk(K, 1)          # worst peak first
        mvalid = torch.isfinite(top)
        fk = self.forder.gather(1, j)                                       # grid index [E,K]
        pr = pk.gather(1, fk)                                               # worst probe at that peak
        zpp = z_pp.gather(1, fk[..., None].expand(E, K, P)).gather(2, pr[..., None])[..., 0]           # [E,K]
        zip_ = z_ip.gather(1, fk[:, :, None, None].expand(E, K, n, P))
        zip_ = zip_.gather(3, pr[:, :, None, None].expand(E, K, n, 1))[..., 0]                          # [E,K,n]
        zii = z_ii.gather(1, fk[..., None].expand(E, K, n))                                              # [E,K,n]
        lzt = self.logZt.gather(1, fk)                                                                   # [E,K]
        zdk = self.zd.gather(2, fk[:, None, :].expand(E, T, K))                                          # [E,T,K]
        part = torch.log10(zii.abs() + EPS) - torch.log10(zpp.abs()[..., None] + EPS)
        g = zip_.abs() / torch.sqrt(zpp.abs()[..., None] * zii.abs() + EPS)
        ed = [part, torch.log10(g + 1e-6), torch.log10(zip_.abs() + EPS) - lzt[..., None]]
        for t in range(T):
            after = zpp[..., None] - zip_ ** 2 / (zii + zdk[:, t, :, None])
            ed.append(torch.log10(after.abs() + EPS) - lzt[..., None])
        edge = torch.stack(ed, -1).clamp(-LOG_CLIP, LOG_CLIP).float() * mvalid[..., None, None]        # [E,K,n,de]
        cfg = self.b.cfg
        fc = float(np.sqrt(cfg.f_min * cfg.f_max))
        fkv = self.freqs_t.gather(1, fk)
        md = torch.stack([torch.log10(fkv / fc), top.clamp(-LOG_CLIP, LOG_CLIP), mvalid.double()], -1).float()
        md = torch.where(mvalid[..., None], md, torch.zeros_like(md))                                   # [E,K,dm]
        dm, de = modal_dims(T)
        if K < K_PORT:
            md = torch.cat([md, md.new_zeros(E, K_PORT - K, dm)], 1)
            edge = torch.cat([edge, edge.new_zeros(E, K_PORT - K, n, de)], 1)
        per_cell = edge.permute(0, 2, 1, 3).reshape(E, n, K_PORT * de)
        return torch.cat([md.reshape(E, 1, -1).expand(E, n, K_PORT * dm), per_cell], -1)
