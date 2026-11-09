"""`--peak_refine` (PROPOSALS 11/03 D7 -> candidate P1): MoTT input with two extra channels per mode token.

D7 (round 36): the grid-feasible but exact-failing solutions of the distill policies fail mostly at a frequency that is
next to one of the 8 mode tokens (covered 0.50-0.75), and the exact worst there is ~0.44 decade above the working-grid
level at the nearest grid point: the token's peak level is read on a grid that is too coarse for the resonance.
This env keeps sim.features_mott unchanged (sim.py is an environment file) and appends, for each token at sorted grid
index j, the vertex of the parabola through (log10 f, level) at j-1, j, j+1:
  ch 4: refined excess  y* - y_j, clipped to [0, 1] (0 when the three points are not concave or j is a band edge)
  ch 5: refined violating flag  y* > FEAS_TOL
so the packed mode descriptor becomes 6 wide (channels 0-3 as in sim, so the valid mask (3) and the fc bias (0) are
unchanged).  Only an input representation change; the model code is the rung-2 MoTT with dm = 6.
"""
import torch

from .sim import EPS, FEAS_TOL, K_MODES, Env, mott_dims

N_EXTRA = 2


def mott_pk_dims(T):
    dn, dm, de, dc = mott_dims(T)
    return dn, dm + N_EXTRA, de, dc


def refine(lvs, lfs, j):
    """lvs, lfs [E,F] sorted level / log10 f; j [E,K] sorted token index -> (excess [E,K], refined level [E,K])."""
    F = lvs.shape[1]
    jl, jr = (j - 1).clamp(min=0), (j + 1).clamp(max=F - 1)
    x0, x1, x2 = lfs.gather(1, jl), lfs.gather(1, j), lfs.gather(1, jr)
    y0, y1, y2 = lvs.gather(1, jl), lvs.gather(1, j), lvs.gather(1, jr)
    d01, d12, d02 = x1 - x0, x2 - x1, x2 - x0
    inner = (j > 0) & (j < F - 1) & (d01 > 0) & (d12 > 0)
    s01 = (y1 - y0) / d01.clamp(min=EPS)
    s12 = (y2 - y1) / d12.clamp(min=EPS)
    a = (s12 - s01) / d02.clamp(min=EPS)                       # second divided difference = parabola's x^2 coeff
    # Newton form p(x) = y0 + s01 (x - x0) + a (x - x0)(x - x1); p'(xv) = 0
    xv = 0.5 * (x0 + x1) - s01 / (2 * a.clamp(max=-EPS))
    yv = y0 + s01 * (xv - x0) + a * (xv - x0) * (xv - x1)
    ok = inner & (a < 0) & torch.isfinite(yv)
    exc = torch.where(ok, (yv - y1).clamp(0, 1), torch.zeros_like(y1))
    return exc, y1 + exc


class PkEnv(Env):
    def features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        self.feat_mode = "mott"
        x = Env.features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas)            # [E,n,D] packed as sim.features_mott
        self.feat_mode = "mott_pk"
        E, n, T = self.E, self.n, self.T
        dn, dm, de, dc = mott_dims(T)
        K = min(K_MODES, z_pp.shape[1])
        lvp = torch.log10(z_pp.abs() + EPS) - self.logZt[..., None]
        lv = lvp.max(-1).values
        lvs = lv.gather(1, self.forder)
        lfs = torch.log10(self.freqs_t.gather(1, self.forder))
        ninf = torch.full_like(lvs[:, :1], -float("inf"))
        peak = (lvs >= torch.cat([ninf, lvs[:, :-1]], 1)) & (lvs >= torch.cat([lvs[:, 1:], ninf], 1))
        top, j = torch.where(peak, lvs, -float("inf")).topk(K, 1)             # same tokens, same order as sim
        mvalid = torch.isfinite(top)
        exc, yr = refine(lvs, lfs, j)
        extra = torch.stack([exc, (yr > FEAS_TOL).double()], -1).float() * mvalid[..., None]   # [E,K,2]
        if K < K_MODES:
            extra = torch.cat([extra, extra.new_zeros(E, K_MODES - K, N_EXTRA)], 1)
        o = dn + K_MODES * de
        mode = x[:, 0, o:o + K_MODES * dm].view(E, K_MODES, dm)
        mode = torch.cat([mode, extra.to(mode.dtype)], -1).reshape(E, 1, -1).expand(E, n, -1)
        return torch.cat([x[..., :o], mode, x[..., o + K_MODES * dm:]], -1)
