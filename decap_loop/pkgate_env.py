"""`--peak_gate` (PROPOSALS 11/07 C1', human): P1's refined peak level used in the feasibility test, not only as input.

Diagnosis: most exact-check failures are near misses — the working grid is met but a resonance peak between grid
points exceeds the target by 0.03-0.06 decade. P1 (peak_env.PkEnv) estimates that true peak level by the parabola
through the three grid points around each of the K mode tokens and feeds it to the model as two extra channels; the
model may still stop on the coarse-grid test. Here the SAME refined level also enters the environment's target-met
test: an episode is "met" only if every grid point AND every refined token peak is <= FEAS_TOL. Consequences follow
sim.py's episode rules unchanged: while not met only "add" is legal, prune/stop open only when met. No exact check,
no new physics (same parabola as P1), no board-kind branch; sim.py untouched (subclass, as peak_env.py).

Everything else is PkEnv: features (6-wide mode tokens), rank-1 updates, action effects. Feature mode "mott_pkgate".
"""
import torch

from .peak_env import PkEnv, refine
from .sim import EPS, FEAS_TOL, K_MODES, action_effects, violation, _probe_diag


class PkGateEnv(PkEnv):
    def refined_met(self, z_pp):
        """[E] bool: every refined token peak <= FEAS_TOL (the K highest grid peaks, as the tokens)."""
        K = min(K_MODES, z_pp.shape[1])
        lvp = torch.log10(z_pp.abs() + EPS) - self.logZt[..., None]
        lv = lvp.max(-1).values
        lvs = lv.gather(1, self.forder)
        lfs = torch.log10(self.freqs_t.gather(1, self.forder))
        ninf = torch.full_like(lvs[:, :1], -float("inf"))
        peak = (lvs >= torch.cat([ninf, lvs[:, :-1]], 1)) & (lvs >= torch.cat([lvs[:, 1:], ninf], 1))
        top, j = torch.where(peak, lvs, -float("inf")).topk(K, 1)
        mvalid = torch.isfinite(top)
        _, yr = refine(lvs, lfs, j.clamp(min=0))
        bad = mvalid & (yr > FEAS_TOL)
        return ~bad.any(1)

    def observe(self, feats=True):
        # sim.Env.observe with one change: feas = grid-met AND refined-peak-met
        z_ip, z_ii = self.z_ip, self.z_ii
        z_pp = _probe_diag(z_ip, self.b.probes[self.inst])
        V, m = violation(z_pp, self.logZt)
        feas = (m <= FEAS_TOL) & self.refined_met(z_pp)
        Vn, mn = action_effects(z_ip, z_ii, z_pp, self.occ, self.zd, self.logZt)
        self.feas, self.V = feas, V
        T = self.T
        empty = (self.occ < 0) & self.placeable
        placed = self.occ >= 0
        legal = torch.zeros(self.E, self.n, T + 1, dtype=torch.bool, device=self.occ.device)
        legal[..., :T] = (~feas)[:, None, None] & empty[..., None]
        if self.allow_prune:
            keep = mn <= 0
            other = torch.arange(T, device=self.occ.device) != self.occ[..., None]
            legal[..., :T] |= feas[:, None, None] & placed[..., None] & keep[..., :T] & ~self.tabu & other
            legal[..., T] = feas[:, None] & placed & keep[..., T]
        mask = torch.cat([legal.reshape(self.E, -1), feas[:, None]], 1)
        self.done |= (self.steps >= self.max_steps) | ~mask.any(1)
        obs = dict(mask=mask, V=V, m=m, feas=feas, Vn=Vn, mn=mn)
        if feats:
            obs["x"] = self.features(z_ip, z_ii, z_pp, V, Vn, mn, feas)
        return obs
