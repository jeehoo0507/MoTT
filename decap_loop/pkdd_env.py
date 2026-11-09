"""`--peak_refine --dedup_modes` (PROPOSALS 11/04 P3, conditional on P2 "win"): P1 channels on the P2 token set.

P1 (`peak_env.PkEnv`) appends to each sim mode token the parabolic-vertex excess of its peak level and the refined
violating flag; P2 (`dedup_env.DedupEnv`) picks the K = 8 tokens on distinct working-grid frequencies.  This env is
DedupEnv's token set with PkEnv's two channels.  The only adaptation: the parabola's three points are the token and its
nearest *distinct*-frequency neighbours (on the padded grid the raw j-1 / j+1 can be copies of j, which PkEnv then
treats as "not inner" and gives 0).  Without repeated frequencies this is PkEnv.refine exactly (checked by
scripts/smoke_pkdd.py).  sim.py, peak_env.py and dedup_env.py are unchanged.
"""
import torch

from .dedup_env import DedupEnv, distinct_peaks
from .peak_env import N_EXTRA
from .sim import EPS, FEAS_TOL, K_MODES, mott_dims


def refine_at(lvs, lfs, jl, j, jr):
    """peak_env.refine with explicit neighbour indices jl < j < jr (same formula)."""
    F = lvs.shape[1]
    x0, x1, x2 = lfs.gather(1, jl), lfs.gather(1, j), lfs.gather(1, jr)
    y0, y1, y2 = lvs.gather(1, jl), lvs.gather(1, j), lvs.gather(1, jr)
    d01, d12, d02 = x1 - x0, x2 - x1, x2 - x0
    inner = (j > 0) & (j < F - 1) & (d01 > 0) & (d12 > 0)
    s01 = (y1 - y0) / d01.clamp(min=EPS)
    s12 = (y2 - y1) / d12.clamp(min=EPS)
    a = (s12 - s01) / d02.clamp(min=EPS)
    xv = 0.5 * (x0 + x1) - s01 / (2 * a.clamp(max=-EPS))
    yv = y0 + s01 * (xv - x0) + a * (xv - x0) * (xv - x1)
    ok = inner & (a < 0) & torch.isfinite(yv)
    exc = torch.where(ok, (yv - y1).clamp(0, 1), torch.zeros_like(y1))
    return exc, y1 + exc


def distinct_neighbours(lfs, j):
    """lfs [E,F] sorted log f; j [E,K] first members of equal-frequency runs -> (jl, jr): last index of the previous
    run and first index of the next run (clamped to the band; a clamped index has the same f, so refine_at's
    d > 0 test rejects it)."""
    E, F = lfs.shape
    first = torch.ones(E, F, dtype=torch.bool, device=lfs.device)
    first[:, 1:] = lfs[:, 1:] != lfs[:, :-1]
    ar = torch.arange(F, device=lfs.device).expand(E, F)
    idx = torch.where(first, ar, torch.full_like(ar, F))
    nf = torch.cat([idx[:, 1:].flip(1).cummin(1).values.flip(1), torch.full_like(idx[:, :1], F)], 1)
    jr = nf.gather(1, j).clamp(max=F - 1)
    return (j - 1).clamp(min=0), jr


class PkDdEnv(DedupEnv):
    def features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        x = DedupEnv.features_mott(self, z_ip, z_ii, z_pp, V, Vn, mn, feas)    # [E,n,D] packed with dm = 4
        E, n, T = self.E, self.n, self.T
        dn, dm, de, dc = mott_dims(T)
        K = min(K_MODES, z_pp.shape[1])
        lvp = torch.log10(z_pp.abs() + EPS) - self.logZt[..., None]
        lvs = lvp.max(-1).values.gather(1, self.forder)
        fs = self.freqs_t.gather(1, self.forder)
        peak = distinct_peaks(lvs, fs)
        top, j = torch.where(peak, lvs, -float("inf")).topk(K, 1)             # same tokens, same order as DedupEnv
        mvalid = torch.isfinite(top)
        lfs = torch.log10(fs)
        jl, jr = distinct_neighbours(lfs, j)
        exc, yr = refine_at(lvs, lfs, jl, j, jr)
        extra = torch.stack([exc, (yr > FEAS_TOL).double()], -1).float() * mvalid[..., None]   # [E,K,2]
        if K < K_MODES:
            extra = torch.cat([extra, extra.new_zeros(E, K_MODES - K, N_EXTRA)], 1)
        o = dn + K_MODES * de
        mode = x[:, 0, o:o + K_MODES * dm].view(E, K_MODES, dm)
        mode = torch.cat([mode, extra.to(mode.dtype)], -1).reshape(E, 1, -1).expand(E, n, -1)
        return torch.cat([x[..., :o], mode, x[..., o + K_MODES * dm:]], -1)
